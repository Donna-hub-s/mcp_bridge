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

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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


def sub_agents_for(agent: str) -> tuple[str, ...]:
    """Les noms des sous-agents pilotes par `agent` — tuple vide si agent
    inconnu ou sans equipe.

    Rend des CHAINES et non des AgentRole : depuis agents.yaml, un sous-agent
    peut porter un nom qui n'est pas un membre de l'enum (il n'aura alors
    aucun outil, voir mcp_tools_for_agent — declarable mais impuissant).

    >>> sub_agents_for("donna")
    ('redacteur', 'analyste')
    >>> sub_agents_for("analyste")
    ()
    """
    if agent != AgentRole.DONNA.value:
        return ()  # seul donna pilote une equipe (SUB_AGENTS_OF)
    return tuple(declared_agents())


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
        # Nom declare dans agents.yaml sans entree Python : l'agent existe (nom,
        # description) mais n'a aucun serveur. Declarable, impuissant.
        return ()
    plafond = AGENT_MCP_TOOLS.get(role, ())
    declared = declared_agents().get(agent)
    if declared is None or declared.tools is None:
        return plafond  # pas de declaration, ou pas de `tools:` : tout le plafond
    # `tools:` ne peut que RETIRER. Un serveur hors plafond a deja ete journalise
    # au chargement (load_agents) ; ici on se contente de l'intersection.
    return tuple(s for s in plafond if s.server in declared.tools)


# ---------------------------------------------------------------------------
# Identite declarative des sous-agents (agents.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredAgent:
    """L'identite d'un sous-agent MCP, lue depuis agents.yaml.

    `tools` est un SOUS-ENSEMBLE souhaite du plafond Python (AGENT_MCP_TOOLS),
    jamais un octroi : voir mcp_tools_for_agent. None = pas de `tools:` declare,
    donc tout le plafond.

    `routing` est transporte tel quel (chaines) — ce que signifie un « niveau »
    appartient au plugin chat (app/chat/src/routing.py), pas a ce pont.
    """

    name: str
    description: str = ""
    tools: frozenset[str] | None = None
    routing: dict[str, str] = field(default_factory=dict)


_ROUTING_KEYS = ("default_tier", "max_tier", "effort")
_AGENTS_FILE = Path(__file__).with_name("agents.yaml")


def _fallback_agents() -> dict[str, DeclaredAgent]:
    """Les sous-agents tels que ce module les connait en Python.

    Sert quand agents.yaml est absent ou illisible : une erreur de syntaxe dans un
    fichier de configuration ne doit pas priver l'utilisateur de ses sous-agents.
    """
    return {
        role.value: DeclaredAgent(name=role.value, description=description)
        for role, description in SUB_AGENT_ROLES.items()
    }


def load_agents(path: Path | None = None) -> dict[str, DeclaredAgent]:
    """Lit agents.yaml. Ne leve jamais : tout echec retombe sur les valeurs Python.

    Un nom revendiqué DEUX FOIS ecarte tous ses porteurs — rien ne dit lequel devrait
    l'emporter, et deleguer a un agent dont on ignore les outils est pire que de ne pas
    deleguer (meme regle que ipc_bridge/discovery.py).
    """
    source = path or _AGENTS_FILE
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return _fallback_agents()
    except Exception as exc:  # noqa: BLE001 — un fichier casse ne doit pas empecher le demarrage
        logger.warning("agents.yaml illisible (%s) : %s — repli sur le catalogue Python", source, exc)
        return _fallback_agents()

    entries = (raw or {}).get("agents") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        logger.warning("agents.yaml : `agents:` absent ou n'est pas une liste — repli sur le catalogue Python")
        return _fallback_agents()

    lus: list[DeclaredAgent] = []
    for entry in entries:
        if not isinstance(entry, dict):
            logger.warning("agents.yaml : entree ignoree (pas un bloc de cles)")
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            logger.warning("agents.yaml : entree sans `name:` — ignoree")
            continue
        lus.append(
            DeclaredAgent(
                name=name,
                description=str(entry.get("description") or "").strip(),
                tools=_tools_subset(name, entry.get("tools")),
                routing=_routing_subset(name, entry.get("routing")),
            )
        )

    porteurs: dict[str, int] = {}
    for agent in lus:
        porteurs[agent.name] = porteurs.get(agent.name, 0) + 1
    ambigus = {name for name, count in porteurs.items() if count > 1}
    for name in sorted(ambigus):
        logger.warning("agents.yaml : agent '%s' declare plusieurs fois — TOUS ecartes", name)

    retenus = {a.name: a for a in lus if a.name not in ambigus}
    if not retenus:
        logger.warning("agents.yaml : aucun agent exploitable — repli sur le catalogue Python")
        return _fallback_agents()
    return retenus


def _tools_subset(name: str, raw: Any) -> frozenset[str] | None:
    """Les serveurs demandes par un agent, bornes au plafond Python. None si `tools:` absent."""
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        logger.warning("agents.yaml : `tools:` de '%s' n'est pas une liste — plafond complet applique", name)
        return None
    demandes = {str(s).strip() for s in raw if str(s).strip()}
    try:
        plafond = {s.server for s in AGENT_MCP_TOOLS.get(AgentRole(name), ())}
    except ValueError:
        plafond = set()  # agent sans entree Python : aucun serveur, quoi qu'il demande
    hors_plafond = demandes - plafond
    if hors_plafond:
        logger.warning(
            "agents.yaml : '%s' demande %s, hors de son plafond Python (AGENT_MCP_TOOLS) — non accorde",
            name, ", ".join(sorted(hors_plafond)),
        )
    return frozenset(demandes & plafond)


def _routing_subset(name: str, raw: Any) -> dict[str, str]:
    """Le bloc `routing:` reduit a ses cles connues. Forme seulement — le sens appartient au chat."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        logger.warning("agents.yaml : `routing:` de '%s' n'est pas un bloc de cles — ignore", name)
        return {}
    routing: dict[str, str] = {}
    for key, value in raw.items():
        cle = str(key).strip()
        if cle not in _ROUTING_KEYS:
            logger.warning("agents.yaml : `routing.%s` de '%s' inconnu — ignore", cle, name)
            continue
        texte = str(value).strip() if value is not None else ""
        if texte:
            routing[cle] = texte
    return routing


_declared_cache: dict[str, DeclaredAgent] | None = None


def declared_agents() -> dict[str, DeclaredAgent]:
    """Les sous-agents declares, lus une fois puis memorises."""
    global _declared_cache
    if _declared_cache is None:
        _declared_cache = load_agents()
    return _declared_cache


def reset_declared_agents_for_tests() -> None:
    """Vide le cache — les tests chargent des agents.yaml differents."""
    global _declared_cache
    _declared_cache = None
