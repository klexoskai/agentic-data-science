"""Base persona class for all AI agents in the pipeline.

Every persona inherits from :class:`BasePersona` and overrides at minimum the
``system_prompt`` property to define its expert perspective.  The class provides
common methods for LLM invocation, self-reflection, and peer review.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PersonaConfig(BaseModel):
    """Configuration for a single persona instance."""

    persona_type: str = Field(description="Key in the persona registry")
    domain_focus: str = Field(
        default="",
        description="Optional domain specialisation injected into the system prompt",
    )
    model: str = Field(default="claude-sonnet-4-6")
    temperature: float = Field(default=0.4)
    max_tokens: int = Field(default=8192)


class BasePersona(ABC):
    """Abstract base class for all agent personas.

    Parameters
    ----------
    config : PersonaConfig
        Runtime configuration (model choice, temperature, etc.).
    tools : list[BaseTool] | None
        LangChain tools this persona is allowed to use.
    """

    def __init__(
        self,
        config: PersonaConfig,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self.config = config
        self.tools: list[BaseTool] = tools or []
        self._llm: Any | None = None  # lazy-initialised

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------
    def _get_llm(self) -> Any:
        """Return the configured LLM, creating it on first access."""
        if self._llm is not None:
            return self._llm

        model_name: str = self.config.model

        if "claude" in model_name.lower() or "anthropic" in model_name.lower():
            from langchain_anthropic import ChatAnthropic

            self._llm = ChatAnthropic(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

        # Bind tools if any
        if self.tools:
            self._llm = self._llm.bind_tools(self.tools)

        return self._llm

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Human-readable persona name (e.g. 'Senior Data Scientist')."""
        return self.__class__.__name__.replace("Persona", "").replace("_", " ")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the full system prompt defining this persona's expertise."""
        ...

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def invoke(self, user_message: str) -> str:
        """Send a message to the LLM and return its text response.

        Parameters
        ----------
        user_message : str
            The user/task message to process.

        Returns
        -------
        str
            The LLM's response content.
        """
        llm = self._get_llm()
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message),
        ]
        logger.info("[%s] Invoking LLM with %d-char message", self.name, len(user_message))
        response = llm.invoke(messages)
        return response.content  # type: ignore[union-attr]

    def reflect(self, own_output: str, task_context: str) -> str:
        """Self-critique: the persona reviews its own output and suggests improvements.

        Parameters
        ----------
        own_output : str
            The output the persona just produced.
        task_context : str
            The original task context for reference.

        Returns
        -------
        str
            Reflection notes — may include suggested edits, caveats, or a
            confirmation that the output is satisfactory.
        """
        reflection_prompt = (
            f"You previously produced the following output for this task:\n\n"
            f"--- TASK CONTEXT ---\n{task_context}\n\n"
            f"--- YOUR OUTPUT ---\n{own_output}\n\n"
            "Now critically review your own output. Identify:\n"
            "1. Any logical errors, gaps, or unsupported assumptions.\n"
            "2. Areas that could be more precise or actionable.\n"
            "3. Anything you would change if you could redo it.\n\n"
            "If the output is satisfactory, say 'LGTM' and briefly explain why. "
            "Otherwise, provide specific improvement suggestions."
        )
        logger.info("[%s] Running self-reflection", self.name)
        return self.invoke(reflection_prompt)

    def review(self, other_output: str, other_name: str, task_context: str) -> str:
        """Peer review: this persona reviews another agent's output.

        Parameters
        ----------
        other_output : str
            The output from the other agent.
        other_name : str
            Name of the agent being reviewed.
        task_context : str
            The original task context.

        Returns
        -------
        str
            Review feedback — approval or specific objections.
        """
        review_prompt = (
            f"You are reviewing the output of **{other_name}**.\n\n"
            f"--- TASK CONTEXT ---\n{task_context}\n\n"
            f"--- {other_name.upper()}'S OUTPUT ---\n{other_output}\n\n"
            "From your expert perspective, evaluate this output:\n"
            "1. Is it correct and complete?\n"
            "2. Does it align with the business objectives?\n"
            "3. Are there risks, blind spots, or better alternatives?\n\n"
            "End with a clear verdict: APPROVE or REQUEST_CHANGES, "
            "followed by specific feedback."
        )
        logger.info("[%s] Reviewing output from %s", self.name, other_name)
        return self.invoke(review_prompt)

    def is_satisfied(self, review_text: str) -> bool:
        """Parse a review response to determine if the agent approved.

        Parameters
        ----------
        review_text : str
            The text returned by :meth:`review` or :meth:`reflect`.

        Returns
        -------
        bool
            ``True`` if the review contains an approval signal.
        """
        upper = review_text.upper()
        return "APPROVE" in upper or "LGTM" in upper
