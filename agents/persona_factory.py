"""Persona Factory — dynamically decides which agents to spin up.

Reads the business context and data source documentation, then uses an LLM to
determine which expert perspectives are needed for this specific project.
Always includes the default roster and may add domain-specific specialisations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.personas.base import BasePersona, PersonaConfig
from agents.personas import PERSONA_REGISTRY

logger = logging.getLogger(__name__)

# The prompt used by the factory to analyse context and recommend personas
_FACTORY_SYSTEM_PROMPT = """\
You are an AI project staffing advisor.  Given a business context document and
data source documentation for a data science project, you decide which expert
perspectives are needed on the team.

You ALWAYS include these three core roles:
- data_scientist
- sales_director
- qa_engineer

You MAY recommend adding a **domain_focus** string to any role so they bring
sector-specific expertise.  For example, if the project is about FMCG, you
might set the data_scientist's domain_focus to "FMCG / Consumer Packaged Goods".

You MAY also recommend additional specialist personas beyond the core three if
the project warrants it (e.g., a "regulatory_compliance" persona for healthcare
data, or a "pricing_strategist" for revenue optimisation projects).  Any extra
persona will be instantiated as a data_scientist with the appropriate domain_focus.

Respond with a JSON array of objects, each with:
- "persona_type": one of the registry keys (data_scientist, sales_director, qa_engineer)
- "domain_focus": string (can be empty)
- "rationale": one-sentence explanation of why this perspective is needed

Example response:
[
  {"persona_type": "data_scientist", "domain_focus": "FMCG / CPG analytics", "rationale": "Core analytical lead with sector knowledge."},
  {"persona_type": "sales_director", "domain_focus": "SEA FMCG distribution", "rationale": "Understands multi-market channel dynamics."},
  {"persona_type": "qa_engineer", "domain_focus": "", "rationale": "Ensures pipeline correctness and data integrity."}
]

Return ONLY the JSON array, no markdown fences or extra text.
"""


class PersonaFactory:
    """Analyses project context and produces a list of configured personas.

    Parameters
    ----------
    factory_config : dict[str, Any]
        Settings from ``config.yaml`` under ``persona_factory``.
    model_config : dict[str, dict[str, Any]]
        Per-role model settings from the active quality preset.
    """

    def __init__(
        self,
        factory_config: dict[str, Any],
        model_config: dict[str, dict[str, Any]],
    ) -> None:
        self.factory_config = factory_config
        self.model_config = model_config
        self._llm = ChatOpenAI(
            model=factory_config.get("model", "gpt-5.4-mini"),
            temperature=factory_config.get("temperature", 0.2),
            max_tokens=factory_config.get("max_tokens", 2048),
        )

    def decide_personas(
        self,
        context_text: str,
        data_sources_text: str,
    ) -> list[BasePersona]:
        """Use an LLM to decide which personas to instantiate.

        Parameters
        ----------
        context_text : str
            Contents of the business context markdown file.
        data_sources_text : str
            Contents of the data sources markdown file.

        Returns
        -------
        list[BasePersona]
            Instantiated persona objects ready for the swarm.
        """
        user_msg = (
            f"## Business Context\n{context_text}\n\n"
            f"## Data Sources\n{data_sources_text}"
        )

        logger.info("PersonaFactory: analysing context to decide agent roster…")
        response = self._llm.invoke(
            [
                SystemMessage(content=_FACTORY_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )

        # Parse the JSON response
        try:
            recommendations = json.loads(response.content)  # type: ignore[arg-type]
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "PersonaFactory: failed to parse LLM response, falling back to defaults."
            )
            recommendations = self._default_recommendations()

        return self._build_personas(recommendations)

    def _default_recommendations(self) -> list[dict[str, str]]:
        """Return the default persona roster when LLM parsing fails."""
        return [
            {
                "persona_type": "data_scientist",
                "domain_focus": "",
                "rationale": "Core analytical lead.",
            },
            {
                "persona_type": "sales_director",
                "domain_focus": "",
                "rationale": "Business perspective and stakeholder lens.",
            },
            {
                "persona_type": "qa_engineer",
                "domain_focus": "",
                "rationale": "Quality assurance and validation.",
            },
        ]

    def _build_personas(
        self,
        recommendations: list[dict[str, str]],
    ) -> list[BasePersona]:
        """Instantiate personas from the recommendation list.

        Parameters
        ----------
        recommendations : list[dict[str, str]]
            Each dict has ``persona_type``, ``domain_focus``, and ``rationale``.

        Returns
        -------
        list[BasePersona]
            Ready-to-use persona instances.
        """
        personas: list[BasePersona] = []
        for rec in recommendations:
            ptype = rec.get("persona_type", "data_scientist")
            domain_focus = rec.get("domain_focus", "")

            # Look up model config for this role
            role_model_cfg = self.model_config.get(ptype, {})
            config = PersonaConfig(
                persona_type=ptype,
                domain_focus=domain_focus,
                model=role_model_cfg.get("model", "claude-sonnet-4-6"),
                temperature=role_model_cfg.get("temperature", 0.4),
                max_tokens=role_model_cfg.get("max_tokens", 8192),
            )

            # Resolve the persona class (fall back to data_scientist for custom roles)
            cls = PERSONA_REGISTRY.get(ptype, PERSONA_REGISTRY["data_scientist"])
            persona = cls(config=config)
            personas.append(persona)

            logger.info(
                "PersonaFactory: created %s (domain_focus=%r) — %s",
                persona.name,
                domain_focus,
                rec.get("rationale", ""),
            )

        return personas
