"""ExperienceOS: compile agent trajectories into reusable executable harnesses.

The framework formalises agent experience accumulation as *knowledge compilation*:

    Neural -> Execute -> Experience -> Compile -> Symbolic Artifact -> Deploy

Core components:
    - ``models``        : data models (Trajectory, Harness, Hoare Triple P/steps/I/Q/R)
    - ``services``      : unified ChatService + EmbeddingService (dependency-injected)
    - ``stores``        : three-layer Store facade (TraceStore / ExperienceStore / ArtifactStore)
    - ``compiler``      : 7-phase harness induction + sandbox validation
    - ``retriever``     : two-stage semantic + precondition matching (RuntimeRouter)
    - ``agent``         : LLM agent fallback + failure classification (F1-F4)
    - ``runtime``       : the main accumulate / deploy loop
    - ``environment``   : pluggable task environment interface
    - ``input_resolver``: schema-guided artifact input binding
    - ``experiments``   : unified runner protocol (compare / runner)

Deprecated (kept for backward compatibility, do not extend):
    - ``repository.py``          — domain facade over SQLite; use Store facade
    - ``storage.py``             — SQLite driver; use through Store facade only
    - ``experience_library.py``  — SQLite compatibility layer; use through Store facade only
    - ``llm.py``                 — deleted; use ``services.ChatService``
    - ``embedding.py``           — deleted; use ``services.EmbeddingService``
"""

from experience_os.config import Config
from experience_os.runtime import Runtime, SystemMode
from experience_os.services import Services, ProviderRegistry

__all__ = ["Config", "Runtime", "SystemMode", "Services", "ProviderRegistry", "__version__"]
__version__ = "0.1.0"
