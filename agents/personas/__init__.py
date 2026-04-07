"""Persona definitions for the agentic data science pipeline."""

from agents.personas.base import BasePersona
from agents.personas.data_scientist import DataScientistPersona
from agents.personas.sales_director import SalesDirectorPersona
from agents.personas.qa_engineer import QAEngineerPersona

PERSONA_REGISTRY: dict[str, type[BasePersona]] = {
    "data_scientist": DataScientistPersona,
    "sales_director": SalesDirectorPersona,
    "qa_engineer": QAEngineerPersona,
}

__all__ = [
    "BasePersona",
    "DataScientistPersona",
    "SalesDirectorPersona",
    "QAEngineerPersona",
    "PERSONA_REGISTRY",
]
