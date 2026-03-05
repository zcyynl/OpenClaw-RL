"""
openclaw-exploop — Training-Free Experience Loop for LLMs.

A lightweight system that improves LLM responses over time *without*
updating model parameters.  It intercepts conversations, extracts
improvement hints via a Judge LLM, stores them in a keyword-indexed
experience library, and injects relevant past experiences into future
prompts.

Zero GPU.  Zero parameter updates.  All improvement comes from
accumulated conversational wisdom injected at inference time.

Main components:

- :class:`ExperienceStore`     — persist and retrieve hints (JSON file).
- :class:`ExperienceExtractor` — call a Judge LLM to extract hints.
- :class:`PromptInjector`      — inject retrieved hints into prompts.

See ``exploop_server.py`` for the FastAPI proxy that ties them together.
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
