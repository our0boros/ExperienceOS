"""ExperienceOS: compile agent trajectories into reusable executable harnesses.

The framework formalises agent experience accumulation as *knowledge compilation*:

    Neural -> Execute -> Experience -> Compile -> Symbolic Artifact -> Deploy

Core components:
    - ``models``      : data models (Trajectory, Harness, TaskTypeStats, ...)
    - ``llm``         : OpenAI-compatible LLM client (ollama / DeepInfra)
    - ``repository``  : four-layer experience repository + version DAG
    - ``retriever``   : semantic retrieval + precondition matching (Runtime Router)
    - ``compiler``    : six-phase harness induction + sandbox validation
    - ``agent``       : LLM agent fallback + failure classification (F1–F4)
    - ``runtime``     : the main accumulate / deploy loop
    - ``environment``: pluggable task environment interface
"""

from experience_os.config import Config
from experience_os.runtime import Runtime, SystemMode

__all__ = ["Config", "Runtime", "SystemMode", "__version__"]
__version__ = "0.1.0"
