"""Catalogue des tools MCP par agent.

Donna est l'agent ORCHESTRATEUR : il pilote des sous-agents (cf.
SUB_AGENTS_OF) et a acces a tous les serveurs MCP connectes via
mcp_bridge pour repartir le travail. Chaque agent (donna ou sous-agent)
defini dans AGENT_MCP_TOOLS precise les serveurs/tools qu'il peut
invoquer.

Sous-agents actuels, pilotes par donna :
- "redacteur"  : Word (creation/edition) + PDF (lecture pieces jointes)
- "analyste"   : Excel (donnees) + PDF (OCR, tableaux, extraction)

Remplace l'ancien filtrage par section/nœud ( NODE_TOOL_PROFILES,
mcp_tool_selection_for(section, node) ) — le filtrage se fait
desormais par nom d'agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentRole(str, Enum):
    """Roles d'agent dans le systeme. Chaque agent a un sous-ensemble
    de tools MCP accessibles (pas tous les tools partout).

    Donna est l'ORCHESTRATEUR : il pilote des sous-agents (cf.
    SUB_AGENTS_OF) et garde un acces complet a tous les serveurs pour
    repartir le travail. Les sous-agents (REDACTEUR, ANALYSTE...) ne
    voient, eux, que les serveurs de leur specialite."""

    DONNA = "donna"
    REDACTEUR = "redacteur"
    ANALYSTE = "analyste"


# ---------------------------------------------------------------------------
# Orchestration : quel agent est pilote par qui.
# ---------------------------------------------------------------------------

SUB_AGENTS_OF: dict[AgentRole, tuple[AgentRole, ...]] = {
    # Donna pilote deux sous-agents : redaction (word + lecture pdf) et
    # analyse (excel + extraction pdf). Chacun a ses serveurs, donna
    # repartit et consolide.
    AgentRole.DONNA: (AgentRole.REDACTEUR, AgentRole.ANALYSTE),
}

# Description courte de chaque sous-agent — reutilisee par le plugin chat
# pour construire le tool `delegate_to_subagent` (en toute lettre dans le
# prompt) et le prompt system du sous-agent au moment de la delegation.
SUB_AGENT_ROLES: dict[AgentRole, str] = {
    AgentRole.REDACTEUR: (
        "crée et édite des documents Word (courriers, comptes rendus), "
        "s'appuie sur les pièces PDF fournies pour leur contenu."
    ),
    AgentRole.ANALYSTE: (
        "lit et analyse des classeurs Excel et des documents PDF (OCR, "
        "tableaux, formulaires) pour en extraire et en croiser les données."
    ),
}


def sub_agents_for(agent: str) -> tuple[AgentRole, ...]:
    """Les sous-agents pilotes par `agent` — tuple vide si agent
    inconnu ou sans equipe.

    >>> sub_agents_for("donna")
    (<AgentRole.REDACTEUR: 'redacteur'>, <AgentRole.ANALYSTE: 'analyste'>)
    >>> sub_agents_for("analyste")
    ()
    """
    try:
        role = AgentRole(agent)
    except ValueError:
        return ()
    return SUB_AGENTS_OF.get(role, ())


@dataclass(frozen=True)
class McpToolSelection:
    """Les tools MCP d'un agent sur un serveur donne.

    Si tool_names est None, l'agent a acces a TOUS les tools de ce
    serveur (filtre desactive).

    server : cle dans integration.yaml ->
        - services.extensions.mcp_bridge.config.servers (serveur statique,
          connecte une fois pour toutes a init())
        - ou services.extensions.mcp_bridge.config.dynamic_servers
          (serveur scopé) : dans ce cas il faut `scoped=True` — le serveur
          n'existe pas à init() et ne se bind qu'apres un
          `open_scoped_server(category, key, root)` (ex. filesystem, une
          racine par mission).
    tool_names : ensemble des noms de tools autorises, ou None = tout.
    """

    server: str
    tool_names: frozenset[str] | None = None
    note: str = ""
    scoped: bool = False


# ---------------------------------------------------------------------------
# Mapping agent -> tools MCP autorises.
# ---------------------------------------------------------------------------

AGENT_MCP_TOOLS: dict[AgentRole, tuple[McpToolSelection, ...]] = {
    # Donna = ORCHESTRATEUR : ne porte AUCUN serveur de travail. Si on
    # lui bindait les mêmes tools que ses sous-agents, les schémas
    # s'additionneraient dans son contexte et le pollueraient. Il
    # délègue le travail à un sous-agent (SUB_AGENTS_OF) qui, lui,
    # charge sa spécialité via list_tools_schema_for_agent().
    AgentRole.DONNA: (
        McpToolSelection(
            "filesystem",
            note="lecture/ecriture fichiers de mission — serveur scopé, ouvert "
                 "sur la racine de la mission via mcp_bridge.open_scoped_server().",
            scoped=True,
        ),
    ),
    # Sous-agent "redacteur" : cree/edite des documents Word, lit les
    # pieces jointes en pdf comme source, et accede aux fichiers de la
    # mission (filesystem scopé, une racine par tenant — exemple :
    # le dossier de travail du cabinet pour ce client).
    AgentRole.REDACTEUR: (
        McpToolSelection(
            "pdf",
            note="lecture des pieces jointes (texte, OCR).",
        ),
        McpToolSelection(
            "word",
            note="creation/edition de documents.",
        ),
        McpToolSelection(
            "filesystem",
            note="lecture/ecriture des fichiers de mission — serveur scopé, "
                 "ouvert sur la racine du dossier courant via "
                 "mcp_bridge.open_scoped_server().",
            scoped=True,
        ),
    ),
    # Sous-agent "analyste" : extrait, lit et analyse des donnees —
    # documents pdf (OCR, tableaux) et classeurs excel, au sein des
    # fichiers de la mission (filesystem scopé du dossier courant).
    AgentRole.ANALYSTE: (
        McpToolSelection(
            "pdf",
            note="OCR, tableaux, formulaires, annotations.",
        ),
        McpToolSelection(
            "excel",
            note="lecture/ecriture feuilles, graphiques, analyse.",
        ),
        McpToolSelection(
            "filesystem",
            note="lecture des fichiers de mission (liste, contenu, recherche) "
                 "— serveur scopé, ouvert sur la racine du dossier courant.",
            scoped=True,
        ),
    ),
}


def mcp_tools_for_agent(agent: str) -> tuple[McpToolSelection, ...]:
    """Selection MCP pour un agent — tuple vide si agent inconnu.

    Utilise par McpBridgeService.list_tools_schema_for_agent() pour
    filtrer les schema OpenAI-format retournes a la boucle LLM du
    sous-agent (et par `list_tools_schema` pour les serveurs statiques).

    >>> mcp_tools_for_agent("redacteur")
    (McpToolSelection(server='pdf', ...), McpToolSelection(server='word', ...))
    >>> mcp_tools_for_agent("inconnu")
    ()
    """
    try:
        role = AgentRole(agent)
    except ValueError:
        return ()
    return AGENT_MCP_TOOLS.get(role, ())
