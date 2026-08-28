"""
Extension Xcore — point d'entrée `ext.mcp_bridge` : pont vers des serveurs
MCP (Model Context Protocol) externes lancés en sous-processus (stdio),
dont les outils sont exposés au format function-calling (OpenAI-style) pour
être injectés directement dans TOOLS_SCHEMA du plugin chat.

Chaque outil distant est préfixé `mcp_<serveur>_<nom>` pour garantir
l'unicité (les serveurs ne se namespacent pas tous eux-mêmes) et permettre
de retrouver la session d'origine à l'exécution. Un serveur indisponible au
démarrage (dépendance manquante, binaire absent...) est dégradé
silencieusement — loggé, mais n'empêche jamais le plugin chat de démarrer.

Configuration dans integration.yaml :
    extensions:
      mcp_bridge:
        module: services.mcp_bridge.service:McpBridgeService
        config:
          servers:
            word:
              command: /path/vers/.venv/bin/python
              args: ["/path/vers/word_mcp_server.py"]
              cwd: data/mcp_documents/word
              env: {PYTHONPATH: /path/vers/word}
              include: [create_document, add_heading, ...]   # optionnel, sinon tout exposé
              path_params: [filename]   # paramètres réécrits par tenant (isolation)
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from xcore.sdk import get_logger
from xcore.services import BaseService, ServiceStatus

logger = get_logger("ext.mcp_bridge")


class McpBridgeService(BaseService):
    name = "mcp_bridge"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._servers_cfg: dict[str, dict] = config.get("servers", {})
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tool_owner: dict[str, str] = {}  # nom préfixé -> serveur
        self._tools_schema: list[dict] = []
        # Isolation multi-tenant : les serveurs MCP tournent en sous-processus
        # PARTAGés (un seul par type, pas un par tenant — coût ressources) ;
        # l'isolation se fait à l'appel, en réécrivant les paramètres qui
        # ressemblent à un chemin de fichier vers un sous-dossier propre à
        # chaque tenant (voir call_tool) — path_params/cwd déclarés par
        # serveur dans integration.yaml.
        self._path_params: dict[str, list[str]] = {}
        self._cwd_by_server: dict[str, Path] = {}
        # Un verrou par serveur : si deux appels d'outils échouent en même
        # temps sur le même serveur mort, un seul déclenche vraiment la
        # reconnexion — l'autre attend son tour plutôt que de lancer un
        # second sous-processus en parallèle.
        self._reconnect_locks: dict[str, asyncio.Lock] = {}
        self._status = ServiceStatus.INITIALIZING

    async def init(self) -> None:
        for server_name, cfg in self._servers_cfg.items():
            if not cfg.get("enabled", True):
                continue
            try:
                await self._connect_server(server_name, cfg)
            except Exception as exc:
                logger.warning(
                    "MCP '%s' indisponible au démarrage (%s) — ses outils sont désactivés.",
                    server_name,
                    exc,
                )
        self._status = ServiceStatus.READY if self._sessions else ServiceStatus.DEGRADED

    async def _connect_server(self, server_name: str, cfg: dict) -> None:
        self._path_params[server_name] = cfg.get("path_params", [])
        if cfg.get("cwd"):
            self._cwd_by_server[server_name] = Path(cfg["cwd"])

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            cwd=cfg.get("cwd"),
            env=cfg.get("env"),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server_name] = session

        # Idempotent : une reconnexion appelle cette méthode une seconde
        # fois pour le même serveur — purge d'abord ses entrées précédentes,
        # sinon _tools_schema accumule des doublons à chaque reconnexion.
        self._tools_schema = [
            t for t in self._tools_schema if self._tool_owner.get(t["function"]["name"]) != server_name
        ]
        self._tool_owner = {k: v for k, v in self._tool_owner.items() if v != server_name}

        include = cfg.get("include")
        listing = await session.list_tools()
        exposed = 0
        for t in listing.tools:
            if include is not None and t.name not in include:
                continue
            prefixed = f"mcp_{server_name}_{t.name}"
            self._tool_owner[prefixed] = server_name
            self._tools_schema.append(
                {
                    "type": "function",
                    "function": {
                        "name": prefixed,
                        # Les docstrings MCP sont souvent verbeuses (style
                        # Args:/Returns:) — coûteux en tokens à chaque tour
                        # d'appel d'outils sur un provider à quota serré
                        # (constaté en pratique : 8000 tokens/min sur ce
                        # compte Groq, dépassé en quelques tours avec les
                        # descriptions complètes). Seule la première phrase
                        # utile suffit au modèle pour choisir l'outil.
                        "description": (t.description or "").split("\n")[0].strip()[:200],
                        "parameters": t.input_schema or {"type": "object", "properties": {}},
                    },
                }
            )
            exposed += 1
        logger.info(
            "MCP '%s' connecté — %d/%d outils exposés", server_name, exposed, len(listing.tools)
        )

    async def _ensure_connected(self, server_name: str) -> bool:
        """Reconnecte un serveur dont la session a disparu (sous-processus
        mort — pipe cassé détecté par un appel raté dans call_tool). Un seul
        essai : si ça échoue, l'appelant renvoie une erreur claire plutôt que
        de boucler indéfiniment sur un serveur durablement indisponible."""
        if server_name in self._sessions:
            return True
        cfg = self._servers_cfg.get(server_name)
        if cfg is None:
            return False

        lock = self._reconnect_locks.setdefault(server_name, asyncio.Lock())
        async with lock:
            if server_name in self._sessions:  # reconnecté entre-temps par un autre appel
                return True
            try:
                await self._connect_server(server_name, cfg)
                logger.info("MCP '%s' reconnecté avec succès.", server_name)
                return True
            except Exception as exc:
                logger.warning("MCP '%s' : reconnexion échouée (%s).", server_name, exc)
                return False

    async def shutdown(self) -> None:
        await self._stack.aclose()
        self._status = ServiceStatus.STOPPED

    def list_tools_schema(self) -> list[dict]:
        """Liste statique (figée au démarrage) — au format function-calling,
        directement concaténable à TOOLS_SCHEMA."""
        return self._tools_schema

    def _namespaced_arguments(self, server_name: str, tenant_id: str | None, arguments: dict) -> dict:
        """Réécrit les paramètres de type chemin vers un sous-dossier propre
        au tenant (`<tenant_id>/<valeur>`), pour qu'un « rapport.docx » créé
        par un tenant n'écrase ni ne soit visible pour un autre — y compris
        via un outil de listing (list_available_documents et équivalents),
        qui énumère un répertoire entier, pas juste des noms de fichiers."""
        if not tenant_id:
            return arguments
        param_names = self._path_params.get(server_name, [])
        if not param_names:
            return arguments

        base_cwd = self._cwd_by_server.get(server_name)
        if base_cwd is not None:
            (base_cwd / tenant_id).mkdir(parents=True, exist_ok=True)

        namespaced = dict(arguments)
        for param in param_names:
            raw = namespaced.get(param)
            if not isinstance(raw, str) or not raw:
                continue
            if raw.startswith(("/", "http://", "https://")) or ".." in raw:
                continue  # chemin absolu, URL, ou tentative de traversée : ne pas toucher
            if raw.startswith(f"{tenant_id}/"):
                continue  # déjà namespacé (ex: relu depuis un appel précédent)
            namespaced[param] = f"{tenant_id}/{raw}"
        return namespaced

    async def call_tool(self, prefixed_name: str, arguments: dict, tenant_id: str | None = None) -> str:
        # server_name reste valide même si le nom vient d'un outil dont le
        # serveur d'origine s'est reconnecté depuis (le nom préfixé et
        # _tool_owner ne bougent pas, seule la session change) — c'est ce qui
        # permet de reconnaître le bon serveur cible pour reconnecter.
        server_name = self._tool_owner.get(prefixed_name)
        if server_name is None:
            return f"Outil MCP inconnu ou non exposé : {prefixed_name}"

        original_name = prefixed_name[len(f"mcp_{server_name}_"):]
        call_args = self._namespaced_arguments(server_name, tenant_id, arguments)

        if server_name not in self._sessions:
            if not await self._ensure_connected(server_name):
                return f"Serveur MCP '{server_name}' indisponible."

        try:
            result = await self._sessions[server_name].call_tool(original_name, call_args)
        except Exception as exc:
            logger.warning(
                "appel MCP '%s' échoué (%s) — sous-processus probablement mort, tentative de reconnexion.",
                prefixed_name,
                exc,
            )
            self._sessions.pop(server_name, None)
            if not await self._ensure_connected(server_name):
                return f"Échec de l'appel à l'outil '{original_name}' ({server_name}) — serveur indisponible."
            try:
                result = await self._sessions[server_name].call_tool(original_name, call_args)
            except Exception as exc2:
                logger.warning("appel MCP '%s' échoué après reconnexion : %s", prefixed_name, exc2)
                return f"Échec de l'appel à l'outil '{original_name}' ({server_name})."

        parts = [block.text for block in result.content if hasattr(block, "text")]
        text = "\n".join(parts) if parts else "(pas de résultat texte)"
        if result.is_error:
            return f"Erreur de l'outil '{original_name}' : {text}"
        return text

    async def health_check(self) -> tuple[bool, str]:
        if not self._sessions:
            return False, "aucun serveur MCP connecté"
        return True, f"{len(self._sessions)} serveur(s) connecté(s)"

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self._status.value,
            "servers": list(self._sessions.keys()),
            "tools": len(self._tools_schema),
        }
