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
import logging
from collections.abc import Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from xcore.services import BaseService, ServiceStatus
from .catalog import SUB_AGENT_ROLES, mcp_tools_for_agent, sub_agents_for

logger = logging.getLogger("ext.mcp_bridge")


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

    async def init(self) -> None:
        self._status = ServiceStatus.INITIALIZING
        for name, cfg in self._servers_cfg.items():
            await self._start_server_with_retry(name, cfg)
        self._status = ServiceStatus.READY if self._servers else ServiceStatus.DEGRADED
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
            env=cfg.get("env"),
            cwd=cfg.get("cwd"),
        )
        handle = await self._connect(name, params, self._stack)
        self._servers[name] = handle
        logger.info("ext.mcp_bridge : '%s' prêt — %d tool(s).", name, len(handle.tools))

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
        params = StdioServerParameters(command=cfg["command"], args=args, env=cfg.get("env"), cwd=cwd)

        stack = AsyncExitStack()
        try:
            handle = await self._connect(name, params, stack)
        except Exception:
            await stack.aclose()
            raise
        self._servers[name] = handle
        self._scoped_stacks[name] = stack
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
        await self.close_all_scoped_servers()
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

        handle = self._server(server)
        result = await handle.session.call_tool(tool_name, arguments or {})
        if result.is_error:
            raise RuntimeError(f"ext.mcp_bridge : '{server}.{tool_name}' a renvoyé une erreur : {result.content}")
        return result.content

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

        return [
            {"name": role.value, "description": SUB_AGENT_ROLES[role]}
            for role in sub_agents_for(agent)
        ]

    @staticmethod
    def _to_openai_schema(tool: Tool, prefix: str | None = None) -> dict[str, Any]:
        """Convertit un tool MCP (SDK mcp) en schéma OpenAI function-calling
        (le format attendu par les providers LLM dans tools=[...]).

        `prefix=None` garde le nom brut du tool (usage LangChain) ;
        sinon chaque tool est nommé `mcp_<prefix>_<nom>` — le format
        attendu par le plugin chat (_execute_tool_call_raw), qui permet
        à `call_tool_named()` de retrouver le serveur d'origine."""
        schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
        name = f"mcp_{prefix}_{tool.name}" if prefix else tool.name
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
        les boucles de sous-agents. Lève si le préfixe ou le tool est
        inconnu.

        Résolution par plus long préfixe de serveur connu : un serveur
        scopé porte un nom `category:<key>` dont la clé peut contenir
        des `_` (`filesystem:client-x:mission_2026`) — un découpage
        naïf sur `_` casserait le nom du tool juste avant le tool lui-même."""

        if not name.startswith("mcp_"):
            raise ValueError(f"ext.mcp_bridge : nom d'outil attendu sous 'mcp_<serveur>_<tool>', reçu '{name}'.")
        rest = name[len("mcp_"):]
        server = max(self._servers, key=lambda s: len(s) if rest.startswith(s + "_") else -1)
        if rest.startswith(f"{server}_"):
            tool_name = rest[len(server) + 1:]
            return await self.call_tool(server, tool_name, arguments)
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
