"""compiler 子包 — Harness Induction 编译管线。

公共 API 向后兼容：``from experience_os.compiler import HarnessInductor``
仍然可用，同时重新导出 ``ValidationResult`` 与三个 prompt 模板。

子模块：
- :mod:`experience_os.compiler.prompts`     — LLM prompt 模板常量
- :mod:`experience_os.compiler.algorithms`   — 纯算法函数（LCS / Daikon / 分段等）
- :mod:`experience_os.compiler.inductor`     — ``HarnessInductor`` 类
"""

from experience_os.compiler.inductor import HarnessCandidate, HarnessInductor, ValidationResult
from experience_os.compiler.prompts import JUDGE_PROMPT, SEGMENT_PROMPT, SYNTHESIS_PROMPT

__all__ = [
    "HarnessInductor",
    "ValidationResult",
    "HarnessCandidate",
    "SEGMENT_PROMPT",
    "SYNTHESIS_PROMPT",
    "JUDGE_PROMPT",
]
