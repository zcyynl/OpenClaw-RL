"""
openclaw-training-free — Training-Free GRPO experience accumulation.

A lightweight reinforcement-learning-inspired system that improves LLM
responses over time *without* updating model parameters.  It intercepts
conversations, extracts improvement hints via a Judge LLM, stores them
in a keyword-indexed experience library, and injects relevant past
experiences into future prompts.

Main components:

- :class:`ExperienceStore`     — persist and retrieve hints (JSON file).
- :class:`ExperienceExtractor` — call a Judge LLM to extract hints.
- :class:`PromptInjector`      — inject retrieved hints into prompts.

See ``openclaw_tf_server.py`` for the FastAPI proxy that ties them
together.
"""

from __future__ import annotations

from experience_store import ExperienceStore
from experience_extractor import ExperienceExtractor
from prompt_injector import PromptInjector

__all__ = [
    "ExperienceStore",
    "ExperienceExtractor",
    "PromptInjector",
]
