"""Extension Xcore — `ext.mcp_bridge` : pont MCP réel vers des serveurs
lancés en sous-processus via `uvx`.

Ce fichier remplace un gateway LLM générique (pools Ollama, repli à 3
paliers) repris d'un autre projet lors du scaffolding XCore initial — sans
rapport avec les tools malgré son nom. Ce qu'il fait maintenant : un
client MCP (stdio, SDK officiel `mcp`) par serveur déclaré en config,
chacun démarré comme un sous-processus `uvx <paquet> [args...]` séparé —
un serveur par catégorie d'outil (voir `agent/tools/catalog.py` pour la
liste des paquets vérifiés : filesystem-mcp, mcp-pdf, excel-mcp-server,
office-word-mcp-server).

La calculatrice n'est volontairement PAS ici : c'est `tools.builtin.calculate`,
un tool natif — inutile de payer un sous-processus MCP pour de
l'arithmétique. `referentiel_normatif` et `memoire` n'ont pas non plus de
serveur ici : aucun paquet du marché ne les couvre (corpus SYCEBNL/ISA et
mémoire inter-mission propres au cabinet) — à construire nous-mêmes plus
tard, comme serveur MCP maison ou service XCore direct.

Config (integration.yaml) :
    services:
      extensions:
        mcp_bridge:
          module: extensions.mcp_bridge.main:McpBridgeService
          config:
            servers:
              pdf:
                command: uvx
                args: ["mcp-pdf"]
              excel:
                command: uvx
                args: ["excel-mcp-server", "stdio"]
              word:
                command: uvx
                args: ["--from", "office-word-mcp-server", "word_mcp_server"]
                # `cwd`/`env` optionnels — utile pour un serveur lancé via
                # un venv dédié plutôt que `uvx` (ex. un venv déjà présent
                # sur la machine cible), ou qui résout ses propres chemins
                # relatifs (fichiers temporaires, dossier de sortie...)
                # contre un répertoire précis plutôt que le cwd du process
                # parent.
                cwd: /chemin/vers/word
                env: {PYTHONPATH: /chemin/vers/word}
            dynamic_servers:
              filesystem:
                command: uvx
                args: ["filesystem-mcp", "{root}"]
                # `cwd` d'un gabarit dynamique peut aussi contenir `{root}`
                # — substitué de la même façon que dans `args`.

`servers` (ci-dessus) démarre un sous-processus par entrée, une fois pour
toutes, à `init()` — pour un serveur dont la portée est la même pour toute
l'app (pdf/excel/word aujourd'hui). `dynamic_servers` déclare des
GABARITS, jamais démarrés à `init()` : `open_scoped_server(category, key,
root)` en lance un exemplaire à la demande, `{root}` substitué dans `args`
— c'est ce que `filesystem` utilise depuis le passage à l'isolement par
mission (secret professionnel : un serveur filesystem-mcp partagé entre
tous les clients aurait pu laisser un LLM parcourir le dossier d'un autre
client).

Usage depuis un plugin/service XCore :
    bridge = self.get_service("ext.mcp_bridge")
    tools  = bridge.list_tools("pdf")
    result = await bridge.call_tool("pdf", "extract_text", {"path": "..."})

    # Serveur scopé (dynamic_servers) — une racine par appelant :
    name  = await bridge.open_scoped_server("filesystem", "client-x:mission-2026", "/data/storage/client/x/mission/2026")
    schemas = bridge.list_tools_schema_for_agent("redacteur")
    await bridge.close_scoped_server("filesystem", "client-x:mission-2026")

**Le filtrage par agent.** Donna est l'agent principal, il orchestre les
sous-agents. `list_tools_schema_for_agent(agent)` filtre les tools d'un
agent donné via `catalog.mcp_tools_for_agent` — la seule source de vérité
sur « quels tools précis un agent a le droit de voir ». Les sous-agents
(redacteur/analyste) reçoivent ces schémas (nommés `mcp_<serveur>_<tool>`)
dans leur boucle LLM dédiée (`delegate_to_subagent`), et les exécutent via
`call_tool_named()`. Binder un serveur MCP entier envoie tous ses schémas
au LLM à chaque appel : mesuré en pratique, les 54 tools de `mcp-pdf`
coûtent ~11 400 tokens à eux seuls — le filtrage par agent (`tool_names`)
est là pour limiter ça sur les agents secondaires. Donna, lui, ne voit
aucun tool `mcp_*` : il délègue (voir plugin chat).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from xcore.services import BaseService, ServiceStatus
from .catalog import declared_agents, mcp_tools_for_agent, sub_agents_for

logger = logging.getLogger("ext.mcp_bridge")

# Contrainte des providers (OpenAI, Anthropic, Groq...) sur les noms
# d'outils : ^[a-zA-Z0-9_-]{1,64}$. Un serveur scopé `filesystem:<uuid>`
# produisait `mcp_filesystem:<uuid>_read_file` — `:` interdit et > 64
# caractères, rejeté en 400 par les providers cloud.
_TOOL_NAME_MAX = 64
_TOOL_NAME_INVALID = re.compile(r"[^a-zA-Z0-9_-]")


def _is_broken_session(exc: BaseException) -> bool:
    """Session MCP inutilisable (processus arrêté, flux fermé) — distincte d'une
    erreur métier renvoyée par l'outil, qui ne justifie jamais une relance."""
    import anyio

    if isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream, BrokenPipeError, ConnectionError)):
        return True
    return "connection closed" in str(exc).lower()


def _short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


@dataclass
class _ServerHandle:
    name: str
    session: ClientSession
    tools: list[Tool] = field(default_factory=list)


class McpBridgeService(BaseService):
    """Service partagé `ext.mcp_bridge` — voir docstring du module."""

    name = "mcp_bridge"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = config or {}
        self._servers_cfg: dict[str, dict[str, Any]] = self._config.get("servers", {})
        self._dynamic_cfg: dict[str, dict[str, Any]] = self._config.get("dynamic_servers", {})
        self._stack = AsyncExitStack()
        self._servers: dict[str, _ServerHandle] = {}
        # Un `AsyncExitStack` par serveur ouvert via `open_scoped_server` —
        # contrairement à `self._stack` (les serveurs statiques, fermés
        # tous ensemble à `shutdown()`), chacun doit pouvoir être fermé
        # individuellement (`close_scoped_server`) sans affecter les autres.
        self._scoped_stacks: dict[str, AsyncExitStack] = {}
        # Nom exposé au LLM → (serveur, tool natif). Rempli à chaque
        # génération de schéma : c'est la seule façon fiable de retrouver
        # un tool dont le nom a été normalisé (serveur scopé, nom trop long).
        self._tool_index: dict[str, tuple[str, str]] = {}
        # Borne d'un appel de tool : sans elle, un serveur MCP bloqué fige la
        # boucle agentique (et le flux SSE) indéfiniment.
        self._call_timeout = float(self._config.get("call_timeout_seconds", 120))
        # Pile de contexte PAR serveur statique : un serveur tombé (processus
        # arrêté, flux coupé) se relance seul, sans toucher aux autres.
        self._server_stacks: dict[str, AsyncExitStack] = {}
        self._restart_locks: dict[str, asyncio.Lock] = {}
        # Serveurs scopés inactifs fermés après ce délai (0 = jamais).
        self._scoped_idle_seconds = float(self._config.get("scoped_idle_seconds", 900))
        self._scoped_last_used: dict[str, float] = {}
        self._in_flight: dict[str, int] = {}
        self._reaper: asyncio.Task | None = None

    async def init(self) -> None:
        self._status = ServiceStatus.INITIALIZING
        for name, cfg in self._servers_cfg.items():
            await self._start_server_with_retry(name, cfg)
        self._status = ServiceStatus.READY if self._servers else ServiceStatus.DEGRADED
        if self._scoped_idle_seconds > 0:
            self._reaper = asyncio.create_task(self._reap_forever())
        logger.info(
            "ext.mcp_bridge prêt — serveur(s) connecté(s) : %s / %d configuré(s)",
            ", ".join(self._servers) or "aucun",
            len(self._servers_cfg),
        )

    async def _start_server_with_retry(
        self, name: str, cfg: dict[str, Any], *, attempts: int = 2, delay: float = 1.0
    ) -> None:
        """Un serveur MCP fraîchement installé par `uvx` (téléchargement à
        froid) ou coupé par un rechargement à chaud du process parent (dev
        server `reload: true`) peut échouer une première fois de façon
        purement transitoire — `MCPError: Connection closed` pendant
        `session.initialize()`, constaté en pratique, sans rapport avec la
        config. Un seul réessai, pas plus : un échec qui se répète est un
        vrai problème (mauvaise config, paquet cassé, chemin invalide), pas
        quelque chose à masquer en boucle."""

        for attempt in range(1, attempts + 1):
            try:
                await self._start_server(name, cfg)
                return
            except Exception:
                if attempt < attempts:
                    logger.warning(
                        "ext.mcp_bridge : tentative %d/%d échouée pour '%s', réessai dans %.1fs.",
                        attempt, attempts, name, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.exception(
                        "ext.mcp_bridge : échec démarrage serveur '%s' après %d tentative(s) (%s)",
                        name, attempts, cfg.get("args"),
                    )

    async def _start_server(self, name: str, cfg: dict[str, Any]) -> None:
        if "command" not in cfg:
            raise ValueError(f"ext.mcp_bridge : serveur '{name}' sans 'command' (attendu ex. 'uvx').")
        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=self.server_env(cfg),
            cwd=cfg.get("cwd"),
        )
        stack = AsyncExitStack()
        try:
            handle = await self._connect(name, params, stack)
        except BaseException:
            await stack.aclose()
            raise
        self._server_stacks[name] = stack
        self._servers[name] = handle
        logger.info("ext.mcp_bridge : '%s' prêt — %d tool(s).", name, len(handle.tools))

    async def _restart_server(self, name: str) -> bool:
        """Relance un serveur statique dont la session est morte. Un seul
        redémarrage à la fois par serveur ; False si la relance échoue."""
        if name not in self._servers_cfg:
            return False
        lock = self._restart_locks.setdefault(name, asyncio.Lock())
        async with lock:
            old = self._server_stacks.pop(name, None)
            self._servers.pop(name, None)
            if old is not None:
                try:
                    await old.aclose()
                except BaseException as exc:  # flux déjà rompu : fermeture best-effort
                    logger.debug("ext.mcp_bridge : fermeture de '%s' : %r", name, exc)
            try:
                await self._start_server(name, self._servers_cfg[name])
            except Exception:
                logger.exception("ext.mcp_bridge : relance de '%s' échouée", name)
                return False
        logger.warning("ext.mcp_bridge : '%s' relancé après une coupure", name)
        return True

    async def _reap_forever(self) -> None:
        interval = max(30.0, min(self._scoped_idle_seconds / 3, 300.0))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_idle_scoped_servers()
            except Exception:
                logger.exception("ext.mcp_bridge : nettoyage des serveurs scopés échoué")

    async def reap_idle_scoped_servers(self, now: float | None = None) -> list[str]:
        """Ferme les serveurs scopés inutilisés depuis `scoped_idle_seconds`
        (jamais un serveur en cours d'appel). Réouverts à la demande ensuite."""
        now = time.monotonic() if now is None else now
        closed = []
        for name in list(self._scoped_stacks):
            if self._in_flight.get(name):
                continue
            if now - self._scoped_last_used.get(name, now) < self._scoped_idle_seconds:
                continue
            self._servers.pop(name, None)
            self._scoped_last_used.pop(name, None)
            stack = self._scoped_stacks.pop(name)
            try:
                await stack.aclose()
            except BaseException as exc:
                logger.debug("ext.mcp_bridge : fermeture de '%s' : %r", name, exc)
            closed.append(name)
        if closed:
            logger.info("ext.mcp_bridge : %d serveur(s) scopé(s) inactif(s) fermé(s)", len(closed))
        return closed

    @staticmethod
    def server_env(cfg: dict[str, Any]) -> dict[str, str] | None:
        """Environnement du sous-processus. xcore remplace un `${VAR}` absent par
        une chaîne vide (sans valeur par défaut possible) : une valeur vide est
        donc retirée, puis `env_defaults` complète ce qui manque. Ainsi
        `DONATA_TRANSPORT: ${DONATA_TRANSPORT}` + `env_defaults: {DONATA_TRANSPORT: memory}`
        vaut « la variable si elle est définie, sinon memory »."""
        env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items() if v not in (None, "")}
        for key, value in (cfg.get("env_defaults") or {}).items():
            if value not in (None, ""):
                env.setdefault(str(key), str(value))
        return env or None

    @staticmethod
    async def _connect(name: str, params: StdioServerParameters, stack: AsyncExitStack) -> _ServerHandle:
        """Cœur partagé entre un serveur statique (`_start_server`, sur
        `self._stack`) et un serveur scopé (`open_scoped_server`, sur son
        propre `AsyncExitStack`) — même séquence de connexion MCP, seule
        la pile qui possède le cycle de vie diffère."""

        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        return _ServerHandle(name=name, session=session, tools=list(listed.tools))

    @staticmethod
    def scoped_server_name(category: str, key: str) -> str:
        return f"{category}:{key}"

    async def open_scoped_server(self, category: str, key: str, root: str) -> str:
        """Démarre (ou réutilise s'il tourne déjà) un serveur MCP scopé à
        `root` — pour un serveur comme `filesystem-mcp` dont la racine
        doit être exactement le dossier d'une mission, pas un répertoire
        partagé entre tous les clients (voir `tools/dossiers.py`).
        `category` doit avoir une entrée dans `dynamic_servers`
        (integration.yaml) avec un `{root}` dans `args`.

        Idempotent : une mission reprise (`Command(resume=...)`, qui
        re-exécute le nœud appelant depuis le début — voir
        `graph/sectiona/worker_piece.py`) rappelant ceci avec la même
        `(category, key)` ne relance pas un second sous-processus, elle
        récupère le même déjà ouvert.

        Renvoie le nom sous lequel ce serveur est enregistré
        (`scoped_server_name`) — à passer à `list_tools`/`call_tool`,
        `list_tools_schema_for_agent` ou `call_tool_named` comme n'importe
        quel serveur statique."""

        name = self.scoped_server_name(category, key)
        if name in self._servers:
            self._scoped_last_used[name] = time.monotonic()
            return name
        if category not in self._dynamic_cfg:
            raise ValueError(
                f"ext.mcp_bridge : aucune entrée 'dynamic_servers.{category}' en config "
                f"(disponible(s) : {', '.join(self._dynamic_cfg) or 'aucune'})."
            )
        cfg = self._dynamic_cfg[category]
        if "command" not in cfg:
            raise ValueError(f"ext.mcp_bridge : dynamic_servers['{category}'] sans 'command'.")
        args = [str(arg).format(root=root) for arg in cfg.get("args", [])]
        cwd = cfg.get("cwd")
        if cwd is not None:
            cwd = cwd.format(root=root)
        params = StdioServerParameters(command=cfg["command"], args=args, env=self.server_env(cfg), cwd=cwd)

        stack = AsyncExitStack()
        try:
            handle = await self._connect(name, params, stack)
        except Exception:
            await stack.aclose()
            raise
        self._servers[name] = handle
        self._scoped_stacks[name] = stack
        self._scoped_last_used[name] = time.monotonic()
        logger.info(
            "ext.mcp_bridge : serveur scopé '%s' prêt (root=%s) — %d tool(s).", name, root, len(handle.tools)
        )
        return name

    async def close_scoped_server(self, category: str, key: str) -> None:
        """Arrête et désenregistre le serveur ouvert par
        `open_scoped_server` pour cette `(category, key)` — sans toucher
        aux autres (contrairement à `shutdown()`, qui ferme tout).
        Silencieux si déjà fermé ou jamais ouvert."""

        name = self.scoped_server_name(category, key)
        self._servers.pop(name, None)
        self._scoped_last_used.pop(name, None)
        stack = self._scoped_stacks.pop(name, None)
        if stack is not None:
            await stack.aclose()

    async def close_all_scoped_servers(self) -> None:
        """Filet de sécurité — ferme tout serveur scopé encore ouvert
        (ex. une mission jamais arrivée à un nœud terminal). Appelé par
        `shutdown()` ; peut aussi être appelé séparément par un futur
        "reaper" périodique — non construit ici, voir le docstring du
        module pour cette limite connue (les serveurs scopés
        s'accumulent jusqu'à l'arrêt de l'app tant qu'aucun appelant
        n'appelle `close_scoped_server` explicitement)."""

        for name in list(self._scoped_stacks):
            self._servers.pop(name, None)
            stack = self._scoped_stacks.pop(name)
            await stack.aclose()

    async def shutdown(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
        await self.close_all_scoped_servers()
        for name in list(self._server_stacks):
            try:
                await self._server_stacks.pop(name).aclose()
            except BaseException as exc:
                logger.debug("ext.mcp_bridge : fermeture de '%s' : %r", name, exc)
        await self._stack.aclose()
        self._servers.clear()
        self._status = ServiceStatus.STOPPED

    async def health_check(self) -> tuple[bool, str]:
        missing = [name for name in self._servers_cfg if name not in self._servers]
        if missing:
            return False, f"serveur(s) non connecté(s) : {', '.join(missing)}"
        return bool(self._servers), f"{len(self._servers)} serveur(s) MCP connecté(s)"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self._status.value,
            "servers": {name: [tool.name for tool in handle.tools] for name, handle in self._servers.items()},
        }

    def list_servers(self) -> list[str]:
        return list(self._servers)

    def list_tools(self, server: str, *, only: Iterable[str] | None = None) -> list[Tool]:
        """Les tools MCP de `server`, filtrés sur `only` si fourni (noms
        exacts, voir `tool.name`). `only=None` renvoie tout — à réserver
        aux nœuds qui n'ont volontairement pas de sélection restreinte
        dans `tools.catalog` (aucun aujourd'hui côté Section A)."""

        tools = self._server(server).tools
        if only is None:
            return tools
        wanted = set(only)
        return [t for t in tools if t.name in wanted]

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Appelle `tool_name` sur `server`. Lève si le serveur/tool est
        inconnu ou si le serveur MCP renvoie une erreur (`is_error`) —
        jamais de valeur inventée en silence."""

        self._in_flight[server] = self._in_flight.get(server, 0) + 1
        try:
            result = await self._call_once_with_restart(server, tool_name, arguments or {})
        finally:
            self._in_flight[server] -= 1
            if server in self._scoped_last_used:
                self._scoped_last_used[server] = time.monotonic()
        if result.is_error:
            raise RuntimeError(f"ext.mcp_bridge : '{server}.{tool_name}' a renvoyé une erreur : {result.content}")
        return result.content

    async def _call_once_with_restart(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Appel borné par le timeout. Si la session d'un serveur STATIQUE est
        rompue (processus mort, flux fermé), une relance puis un seul nouvel essai."""
        handle = self._server(server)
        try:
            return await asyncio.wait_for(handle.session.call_tool(tool_name, arguments), timeout=self._call_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"ext.mcp_bridge : '{server}.{tool_name}' sans réponse après {self._call_timeout:.0f}s."
            ) from exc
        except Exception as exc:
            if not _is_broken_session(exc) or not await self._restart_server(server):
                raise
        handle = self._server(server)
        try:
            return await asyncio.wait_for(handle.session.call_tool(tool_name, arguments), timeout=self._call_timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"ext.mcp_bridge : '{server}.{tool_name}' sans réponse après {self._call_timeout:.0f}s."
            ) from exc

    def list_tools_schema(self) -> list[dict[str, Any]]:
        """Schémas OpenAI-format des tools des serveurs STATIQUES
        connectés — utilisé par le plugin chat (_effective_tools) pour
        injecter les tools MCP dans le prompt du LLM. Pas de filtrage
        par agent ici : le plugin chat est l'agent principal et reçoit
        tout.

        Les serveurs scopés (`open_scoped_server`, ex. filesystem) en
        sont exclus : ils portent un nom type `filesystem:<key>` (pas un
        bon préfixe dans le prompt) et ne se bindent qu'à la demande,
        par une mission précise — via `open_scoped_server()` +
        `list_tools_schema_for_scoped()` (voir la délégation de sous-agent).

        Chaque tool est préfixé `mcp_<serveur>_` pour garantir l'unicité
        et permettre à `call_tool_named()` de retrouver le serveur d'origine."""

        schemas: list[dict[str, Any]] = []
        for name, handle in self._servers.items():
            if self._is_scoped(name):
                continue
            for tool in handle.tools:
                schemas.append(self._to_openai_schema(tool, prefix=name))
        return schemas

    def list_tools_schema_for_agent(self, agent: str) -> list[dict[str, Any]]:
        """Schémas OpenAI-format des tools accessibles à UN sous-agent
        (`catalog.mcp_tools_for_agent`) — le sous-ensemble exact qu'il
        peut invoquer, nommé `mcp_<serveur>_<tool>` et prêt à binder à
        sa propre boucle LLM. Renvoie `[]` si l'agent est inconnu ou
        n'a aucune sélection sur un serveur statique (ex. donna, qui
        n'a que du scopé)."""

        schemas: list[dict[str, Any]] = []
        for selection in mcp_tools_for_agent(agent):
            if selection.scoped:
                continue
            if selection.server not in self._servers:
                logger.warning(
                    "ext.mcp_bridge : sélection pour '%s' référence le serveur '%s', non connecté.",
                    agent, selection.server,
                )
                continue
            for tool in self._servers[selection.server].tools:
                if selection.tool_names is not None and tool.name not in selection.tool_names:
                    continue
                schemas.append(self._to_openai_schema(tool, prefix=selection.server))
        return schemas

    def list_tools_schema_for_scoped(self, agent: str, category: str, key: str) -> list[dict[str, Any]]:
        """Schémas OpenAI-format des tools d'un serveur scopé DÉJÀ OUVERT
        (`open_scoped_server(category, key, ...)`, ex. filesystem) pour
        `agent` — la même sélection catalog que
        `list_tools_schema_for_agent`, mais résolue contre ce serveur
        scopé au lieu des serveurs statiques. Renvoie `[]` si le serveur
        n'est pas ouvert (rien à binder) ou si l'agent n'a pas de
        sélection scopée sur `category`.

        C'est le chemin consommé par la délégation de sous-agent (plugin
        chat) : ouvrir le filesystem scopé sur la racine d'une mission,
        binder ses tools au sous-agent qui effectue le travail — donnant
        au sous-agent accès aux fichiers de mission sans jamais exposer
        de serveur partagé ni gonfler le contexte de donna."""

        name = self.scoped_server_name(category, key)
        handle = self._servers.get(name)
        if handle is None:
            return []
        selection = next(
            (s for s in mcp_tools_for_agent(agent) if s.scoped and s.server == category),
            None,
        )
        if selection is None:
            return []
        schemas: list[dict[str, Any]] = []
        for tool in handle.tools:
            if selection.tool_names is not None and tool.name not in selection.tool_names:
                continue
            schemas.append(self._to_openai_schema(tool, prefix=name))
        return schemas

    def list_sub_agents(self, agent: str) -> list[dict[str, str]]:
        """Les sous-agents pilotés par `agent`, avec leur description —
        consommé par le plugin chat pour documenter le tool
        `delegate_to_subagent` (les noms et rôles vont en toute lettre
        dans le prompt)."""

        declared = declared_agents()
        return [
            {"name": name, "description": declared[name].description}
            for name in sub_agents_for(agent)
            if name in declared
        ]

    def sub_agent_routing(self, agent: str) -> dict[str, dict[str, str]]:
        """Le bloc `routing:` déclaré par chaque sous-agent piloté par `agent` (agents.yaml).

        Transporté tel quel : ce que signifie un « niveau » appartient au plugin chat
        (app/chat/src/routing.py), pas à ce pont. Consommé par agent_registry.py.
        """
        declared = declared_agents()
        return {
            name: dict(declared[name].routing)
            for name in sub_agents_for(agent)
            if name in declared and declared[name].routing
        }

    def _public_tool_name(self, server: str, tool_name: str) -> str:
        """Nom `mcp_<serveur>_<tool>` conforme aux contraintes providers
        (^[a-zA-Z0-9_-]{1,64}$). Un serveur scopé `category:key` devient
        `category_<hash(key)>` — la clé (souvent un uuid de tenant) ne fuit
        pas dans le prompt et le `:` disparaît. Au-delà de 64 caractères,
        le nom du tool est remplacé par un hash stable."""
        if self._is_scoped(server):
            category, _, key = server.partition(":")
            prefix = f"{category}_{_short_hash(key)}"
        else:
            prefix = server
        prefix = _TOOL_NAME_INVALID.sub("_", prefix)
        name = f"mcp_{prefix}_{_TOOL_NAME_INVALID.sub('_', tool_name)}"
        if len(name) > _TOOL_NAME_MAX:
            name = f"mcp_{prefix}_{_short_hash(tool_name, 12)}"[:_TOOL_NAME_MAX]
        return name

    def _to_openai_schema(self, tool: Tool, prefix: str | None = None) -> dict[str, Any]:
        """Convertit un tool MCP (SDK mcp) en schéma OpenAI function-calling
        (le format attendu par les providers LLM dans tools=[...]).

        `prefix=None` garde le nom brut du tool (usage LangChain) ;
        sinon `prefix` est le nom du serveur et le tool reçoit un nom public
        (`_public_tool_name`), indexé pour que `call_tool_named()` retrouve
        le serveur et le tool natif d'origine."""
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        if prefix:
            name = self._public_tool_name(prefix, tool.name)
            self._tool_index[name] = (prefix, tool.name)
        else:
            name = tool.name
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool.description or "",
                "parameters": schema,
            },
        }

    @staticmethod
    def _is_scoped(server_name: str) -> bool:
        """Un serveur scopé porte le nom `scoped_server_name(category,
        key)` = `f"{category}:{key}"` — le caractère `:` n'apparaît dans
        aucun nom de serveur statique (clefs integration.yaml : pdf,
        excel, word...)."""
        return ":" in server_name

    async def call_tool_named(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> Any:
        """Appelle un tool MCP à partir de son nom préfixé
        `mcp_<serveur>_<tool>` — le format renvoyé par
        `list_tools_schema()` / `list_tools_schema_for_agent()` /
        `list_tools_schema_for_scoped()`, et attendu par le plugin chat /
        les boucles de sous-agents. Lève KeyError si le nom est inconnu
        (voir `resolve_tool_name`), PermissionError si un serveur scopé
        est appelé depuis un autre tenant que celui de sa clé."""

        server, tool_name, scoped = self.resolve_tool_name(name)
        if scoped and tenant_id is not None:
            # Un serveur scopé porte la clé de son propriétaire
            # (`filesystem:<tenant_id>`) : jamais appelable depuis un autre
            # tenant, même si le modèle devine ou recopie son nom.
            _, _, key = server.partition(":")
            if key != tenant_id:
                raise PermissionError(
                    f"ext.mcp_bridge : serveur scopé '{server}' refusé pour ce tenant."
                )
        return await self.call_tool(server, tool_name, arguments)

    def resolve_tool_name(self, name: str) -> tuple[str, str, bool]:
        """(serveur, tool natif, scopé ?) d'un nom public `mcp_*`. Lève
        KeyError si le nom est inconnu ou si son serveur n'est plus connecté.

        Index d'abord (noms générés par les schémas) ; sinon repli par plus
        long préfixe, limité aux serveurs STATIQUES — un serveur scopé n'est
        joignable que via un nom réellement présenté au modèle."""
        if not name.startswith("mcp_"):
            raise KeyError(f"ext.mcp_bridge : nom d'outil attendu sous 'mcp_<serveur>_<tool>', reçu '{name}'.")
        indexed = self._tool_index.get(name)
        if indexed is not None and indexed[0] in self._servers:
            server, tool_name = indexed
            return server, tool_name, self._is_scoped(server)

        rest = name[len("mcp_"):]
        static = [s for s in self._servers if not self._is_scoped(s) and rest.startswith(s + "_")]
        if static:
            server = max(static, key=len)
            return server, rest[len(server) + 1:], False
        raise KeyError(
            f"ext.mcp_bridge : serveur inconnu pour '{name}' "
            f"(connectés : {', '.join(self._servers) or 'aucun'})."
        )

    def _server(self, name: str) -> _ServerHandle:
        try:
            return self._servers[name]
        except KeyError as exc:
            raise KeyError(
                f"ext.mcp_bridge : serveur '{name}' inconnu ou non connecté "
                f"(disponibles : {', '.join(self._servers) or 'aucun'})."
            ) from exc
