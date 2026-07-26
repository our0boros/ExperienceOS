# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **基准文档**：工程实施与论文撰写一律以 [`docs/ExperienceOS.md`](docs/ExperienceOS.md) 为准。本文件仅作为开发指引，当与基准文档冲突时以基准文档为准。
> 历史文档 `docs/Executable Experience Discuss.md` 保留为历史讨论，不作为实施依据。

## Project Overview

**ExperienceOS** — a research framework for *knowledge compilation*: converting agent trajectories into reusable executable harnesses. The core thesis:

> **Knowledge Compilation** — not RAG, not fine-tuning. We compile verified executable artifacts (harnesses) that are model-agnostic, environment-scoped, versioned, and self-repairing. A harness changes the *execution path* (bypassing LLM entirely on hit), not the *input distribution*.

The pipeline:
```
Neural → Execute → Experience → Compile → Symbolic Artifact → Deploy
```

Key insight: "intelligence is not whether an agent uses tools, but whether it **creates tools**."

## Key Commands

```bash
# Install the package (editable)
pip install -e .

# Install with dev dependencies (linting, testing)
pip install -e ".[dev]"

# Install with embedding support (sentence-transformers + torch)
pip install -e ".[embeddings]"

# Install with tau2 support (needed for τ-bench experiments)
pip install -e ".[tau2]"

# Lint with ruff
ruff check experience_os/

# Check LLM connectivity
experience-os ping

# Run mock demo (accumulation → induction → deployment)
experience-os demo

# Run τ-bench integration demo
experience-os tau2-demo --domain retail --warmup 3 --eval 3

# Run a baseline comparison experiment
experience-os compare --method react --model ollama/qwen2.5:7b --domain retail --warmup 3 --eval 5

# Plot accumulation curve from result JSONs
experience-os curve docs/exp_results/*.json --window 3

# List experiments in LTS store
experience-os lts

# Collect environment metadata
experience-os env-info

# Show repository status
experience-os status

# List compiled harnesses
experience-os harnesses
```

**No tests exist yet for the `experience_os` package** — only submodules (tau2-bench, SkillOpt, harbor-TerminalBench) have their own tests.

## Environment Configuration

All settings via environment variables (see `.env.example`):

- `EOS_LLM_BACKEND=ollama` (local testing, default) or `deepinfra` (production)
- `EOS_OLLAMA_MODEL=qwen2.5:7b` — local model
- `EOS_DEEPINFRA_MODEL=MiniMaxAI/MiniMax-M2.7` — remote model
- `DEEPINFRA_TOKEN=sk-...` — API key for DeepInfra
- `EOS_EMBEDDING_MODEL=qwen2.5:7b` — embedding model (via ollama)
- `EOS_MIN_SUPPORT=3` — trajectories needed before induction triggers
- `EOS_VALIDATION_THRESHOLD=0.8` — harness replay success rate gate

## Design Philosophy

### Knowledge Compilation, Not RAG

RAG retrieves text into the LLM's context window, changing the *input distribution* at the same marginal cost per token. ExperienceOS compiles experience into executable code, changing the *computation path* — harness execution bypasses the LLM entirely.

This distinction drives all design decisions:
- **RAG** = give the CPU a better cheat sheet
- **Knowledge Compilation** = extend the instruction set

### The Hoare Triple Formalism

Every compiled harness implements the extended Hoare Triple from the research proposal:

```
H = <P, steps, I, Q, R>

P = preconditions (env state required for execution)
steps = parameterized action sequence
I = invariants (predicates that remain true throughout)
Q = postcondition / terminal verifier (how to check success)
R = rollback strategy (how to undo on failure)
```

### Bayesian Induction Criterion

A harness is created only when the Bayesian utility threshold is met:

```
H* = argmax_H P(H | T_c) ∝ P(T_c | H) · P(H)

likelihood P(T_c | H) = replay success rate on source trajectories
prior    P(H) ∝ exp(-λ · MDL(H))
MDL(H) = α·|steps| + β·|params| + γ·|invariants|
```

Induction fires when `support_count >= MIN_SUPPORT` (default 3). Rationale: 1 trajectory is noise, 2 is coincidence, 3 is sufficient to infer common structure.

### Artifact Utility

```
U(a) = ΔSuccess(a) + ΔEfficiency(a) - Cost_creation
```

An artifact is only worth compiling if its expected utility over future tasks exceeds the cost of creating it. This prevents artifact explosion from one-off patterns.

## Architecture

### Core Loop (`runtime.py`)

Two modes controlled by `SystemMode`:
- **ACCUMULATION** — always use LLM agent, record trajectories, trigger induction when `support_count >= MIN_SUPPORT`. Clean trajectories prevent bootstrapping artifacts (harness quality affecting induction quality).
- **DEPLOYMENT** — prefer harness via `RuntimeRouter`, fall back to agent on miss/failure, continue recording for online learning.

After each task: trajectory logged → stats updated → induction trigger checked.

### System Flow

```
User Task → [Runtime Router] → (harness match? → execute harness | agent fallback)
                                                              ↓
                                   [Experience Accumulator] ←─┘
                                        ↓
                                   (trigger satisfied? → Inductor)
                                        ↓
                                   [Harness Inductor] (6 phases)
                                        ↓
                                   [Repository / LTS Store]
```

### Main Modules

| Module | Role |
|--------|------|
| `models.py` | Data models: Trajectory, Harness (Hoare Triple), ExperienceRecord, TaskTypeStats, ExecutionResult |
| `config.py` | Environment-var-based Config (LLMConfig, InductionConfig) |
| `llm.py` | OpenAI-compatible LLM client (ollama/DeepInfra), embedding, streaming |
| `embedding.py` | 3-level fallback: Qwen3-Embedding-8B → ollama API → hash pseudo-vector. SQLite-cached. |
| `environment.py` | `BaseEnvironment` ABC + `MockEnvironment` (for testing the full loop) |
| `agent.py` | ReAct-style LLM agent fallback + F1-F4 failure classifier |
| `compiler.py` | 6-phase Harness Induction: segment → intersect preconditions → mine invariants → abstract steps → synthesize → validate |
| `retriever.py` | RuntimeRouter: 2-stage retrieval (semantic cosine → precondition matching) |
| `repository.py` | 4-layer JSON-file repository + version DAG ancestry traversal |
| `storage.py` | SQLite storage layer (structured queries, vector BLOBs, env metadata) |
| `experience_library.py` | Hierarchical SQLite library: **LTS** (persistent, append-only) + **experiment** (temporary) |
| `tau2_adapter.py` | τ-bench adapter: environment wrapper, trajectory conversion, warmup/eval splitting |
| `baseline_eval.py` | Vanilla LLM baseline evaluation |
| `experiments/compare.py` | Unified runner: vanilla / react / experienceos / skillopt |
| `experiments/curve.py` | Accumulation curve plotting (SR + cost convergence) |
| `cli.py` | CLI entry point with all subcommands |

### Failure Classification (F1-F4)

| Code | Meaning | Trigger | Harness Action |
|------|---------|---------|----------------|
| F1 | Precondition gap | Env state doesn't match harness expectations | Keep old harness, create specialized variant |
| F2 | Implementation error | Harness code bug (NameError, etc.) | Mark DEPRECATED → patch generation → new version |
| F3 | Environment drift | Changed API/UI ("not found", KeyError) | Keep old, mark version scope, add new adapter |
| F4 | Out of scope | Task outside capability | No change, record OOD case |

### Induction Pipeline (6 Phases)

1. **Segment** — identify semantic sub-task boundaries (LLM for >3 steps). *Currently: results discarded after phase — gap.*
2. **Intersect preconditions** — common env attributes across all trajectories (cross-trajectory set intersection).
3. **Mine invariants** — heuristic: consistent first action + always-success outcome. *Gap: not Daikon-style dynamic invariant detection.*
4. **Abstract steps** — regex-based quotation replacement with `{param}` slots. *Gap: not true LCS + type-aware parameterization.*
5. **Synthesize** — LLM generates `run()` function using `call_tool()` API from few-shot examples.
6. **Validate** — sandbox replay; gate at `validation_threshold` (default 0.8). Result: APPROVED / NEEDS_REVISION / REJECTED. *Gap: NEEDS_REVISION doesn't trigger retry loop.*

## Storage Architecture

### Four-Layer Experience Repository

| Layer | Type | Table/Template | Append-only? | Purpose |
|-------|------|----------------|--------------|---------|
| 0 | Trajectory | `trajectories` | ✅ Yes | Raw observation-action-result sequences + full LLM conversation |
| 1 | ExperienceRecord | `records` | No (versioned) | Semantic summaries: preconditions, parameterized steps, invariants |
| 2 | Harness / Artifact | `harnesses` / `artifacts` | No (version DAG) | Compile-time: executable Python code or text skill document |
| 3 | Meta-Experience | `stats` | No (updated) | Per-task-type success rates, token savings, failure patterns |

### Dual Storage: SQLite + JSON

Two parallel storage paths exist for historical reasons:

**SQLite (primary, modern)**:
- Managed by `storage.py` (older) and `experience_library.py` (newer standard)
- Tables: `trajectories`, `records`, `harnesses`, `stats`, `embeddings`, `env_metadata`
- Embeddings stored as float32 BLOBs with `text_hash → vector` lookup
- Structured env metadata as independent columns (not JSON blob) for SQL queries

**JSON files (legacy, backward-compatible)**:
- Managed by `repository.py` — reads/writes `.json` files in `trajectories/`, `records/`, `harnesses/`, `stats/` subdirectories
- Gradually being replaced by ExperienceLibrary
- Migration path: `Storage.migrate_from_json()`

### ExperienceLibrary: The Modern Standard

Defined in `experience_library.py`. Two database modes:

**LTS (Long-Term Storage)** — `.experience_os_data/lts_library.db`
- Persistent across experiments
- Trajectories are **append-only, never deleted** (ground truth)
- Full conversation logging via `serialize_messages()` (every prompt/response/tool_call)

**Experiment (ephemeral)** — `.experience_os_data/exp_<id>.db`
- Per-experiment temporary database
- Raw data is always in LTS even if experiment DB is discarded

Tables:
```sql
trajectories -- seq, experiment_id, method, domain, task_id, task_type, 
             -- task_description, idx, phase(warmup|eval), success, reward,
             -- tokens, latency, path, task_json, messages_json, steps_json
records     -- seq, task_type, preconditions_json, param_steps_json, 
             -- invariants_json, terminal_verifier, superseded_by
artifacts   -- seq, task_type, artifact_type(harness|skill), procedure_code,
             -- skill_text, verification_status, validation_score,
             -- embedding_blob, parent_seq, edge_type(patch|specialization|composition)
stats       -- task_type, total_executions, harness/agent_executions, 
             -- successes, failure_counts_json, estimated_token_savings
embeddings  -- text_hash, embedding_blob, model, created_at
```

### Version DAG

Artifact versions form a directed acyclic graph, not a linear chain:

```
AssetVerifier-v1
    │
    ├── [F2: schema drift] → AssetVerifier-v2 (patch, parent=v1)
    │
    ├── [F3: Chrome/FF diff] → AssetVerifier-v1.chrome (specialization, parent=v1)
    │
    └── [composition] → DocValidator (composition, parents=[v2, FileReader-v1])
```

Edge types:
- `patch` — bug fix or implementation update
- `specialization` — environment-specific adaptation (browser, OS, app version)
- `composition` — combination of multiple artifacts into a higher-level one

## Artifact Lifecycle

```
Trajectory Accumulation
    ↓
Pattern Discovery (clustering + support_count tracking)
    ↓
Induction Trigger (support_count ≥ MIN_SUPPORT or F2 ≥ 2)
    ↓
6-Phase Compilation (segment → preconditions → invariants → params → synthesize → validate)
    ↓
Validation Decision
    ├─ APPROVED  → Registry (ACTIVE)
    ├─ NEEDS_REVISION → store as DRAFT → (gap: retry loop not implemented)
    └─ REJECTED  → discard
    ↓
Deployment via RuntimeRouter
    ↓
Execution
    ├─ Success → update confidence stats
    └─ Failure → F1-F4 classification → version update trigger
```

### Runtime Router: Two-Stage Retrieval

**Stage 1 — Semantic Retrieval (coarse):** Cosine similarity between task embedding and harness embedding. Harness embedding = `task_type + description + preconditions_summary + example_tasks`.

**Stage 2 — Precondition Matching (fine):** Check every hard precondition against the environment snapshot. Distinguish:
- **Hard conditions** (OS type, app existence, permissions) — must match
- **Soft conditions** (browser version, screen resolution, latency) — allow degraded execution

Retrieval result: `FULL_MATCH` → high confidence / `SOFT_MATCH` → medium confidence / `NO_MATCH` → agent fallback.

## Embedding Architecture

Three-level fallback with SQLite caching:

1. **Qwen3-Embedding-8B** (sentence-transformers, GPU if available) — best quality
2. **Ollama embeddings API** (OpenAI-compatible `/v1/embeddings`) — fallback
3. **Hash-based pseudo-vector** (SHA-256 → float32, no semantic meaning) — last resort

All embeddings cached in SQLite `embeddings` table by `text_hash → float32 BLOB`. Cache hit avoids recomputation.

## Experiment Design

### Four Baseline Methods (same backbone, same warmup data)

| Method | Description | Experience Form |
|--------|-------------|-----------------|
| **vanilla** | Pure LLM, no tools | None (lower bound) |
| **react** | τ-bench ReAct agent | None (current SotA) |
| **coe** (Compilation of Experience) | ExperienceOS DEPLOYMENT mode | Compiled Python code (ours) |
| **skillopt** | SkillOpt-optimized text skill | Text document (strongest baseline) |

### Data Split: Warmup vs Evaluation

Critical design to prevent data leakage:
- **Warm-up Pool**: first K=3 instances per task type → used for accumulation/induction
- **Evaluation Pool**: remaining instances → used for evaluation, NEVER seen during warmup
- All baselines use the **same warmup data** — comparison is about "how to use experience," not "how much experience"

### Experiment Variants

| Variant | Accumulation Pool | Evaluation Pool | What It Tests |
|---------|------------------|-----------------|---------------|
| `type_split` (default) | First K per type | Same type, remaining instances | Within-type generalization |
| `replay` | First K per type | Same instances re-run | Upper bound (memory vs generalization) |
| `cross_domain` | Domain X (e.g. airline) | Domain Y (e.g. retail) | Cross-domain transfer |

### Key Metrics

- Task Success Rate (SR)
- Average Tokens per Task
- Harness Hit Rate (fraction of eval tasks where harness was used)
- Accumulation curve: rolling SR vs task sequence number (the "crossover" proof)
- Token cost convergence curve

### The "Accumulation Curve" (Core Chart)

```
x-axis: task sequence number
y-axis: rolling average success rate

Lines:
  - Vanilla agent: flat (no learning from experience)
  - RAG agent: slight upward (few-shot benefit)
  - ExperienceOS (coe): flat during warmup, then significant rise at crossover point
  - Fixed harness (upper bound): highest from the start
```

This crossover is the central empirical claim of the paper.

## Submodules (git submodules)

- `tau2-bench/` — Sierra's τ-bench framework (retail/airline customer service). Primary experiment environment.
- `harbor-TerminalBench/` — Terminal-based task evaluation. Supplementary environment for CLI-task experiments.
- `SkillOpt/` — Microsoft's SkillOpt. Strongest baseline comparator for text-skill optimization.

## Knowledge Base Vision

The architecture naturally extends to a three-tier knowledge infrastructure:

```
Level 0: Personal Knowledge Base
  - Private, user-specific patterns
  - Environment: individual's OS/apps/settings

Level 1: Organization Knowledge Base
  - Enterprise-specific workflows and tooling
  - Company-approved tool versions and processes

Level 2: Public Knowledge Base
  - Universal software (Office, Chrome, Slack, etc.)
  - Community-verified, cross-platform
  - Like npm/pip for agent skills

Priority chain: Personal > Organization > Public
(Like Linux PATH resolution)
```

### Economic Implication

```
Traditional agent economics:
  Cost = tasks × per-task reasoning cost
  Scale → linear cost growth

Knowledge Infrastructure economics:
  Cost = initial accumulation + tasks × marginal execution cost
  Marginal execution cost ≈ 0 (deterministic code)
  Scale → per-user cost approaches zero
```

## Research Questions

| RQ | Question | Verification |
|----|----------|-------------|
| RQ1 | Can an agent reliably discover patterns in trajectories and compile them into verified artifacts? | Induction success rate + replay validation |
| RQ2 | Do compiled artifacts generalize across entities, task families, environments, and model backbones? | Transfer experiments (type_split, cross_domain) |
| RQ3 | Does performance monotonically improve as the artifact repository grows? | Accumulation curve |
| RQ4 | Can artifacts self-repair through failure feedback without regression on previously successful tasks? | Version DAG + patch success rate |

## Key Gaps (from ExperienceOS.md §10 gap analysis)

| Priority | Gap | Impact |
|----------|-----|--------|
| P1 | Phase 1 (segmentation) results discarded in compiler.py | Multi-step trajectories not properly segmented |
| P1 | Phase 3 (invariants) is heuristic, not Daikon-style | Weak invariant discovery |
| P1 | Phase 4 (parametrization) uses regex, not LCS | Poor parametrization on real trajectories |
| P1 | No baseline comparison framework fully set up | Can't measure progress vs alternatives |
| P1 | No accumulation curve visualization integrated | Core empirical claim unverifiable |
| P2 | NEEDS_REVISION has no retry/fix loop | Failed induction gives up after one attempt |
| P2 | No specialization trigger (new_variation_detected) | Environment drift not handled |
| P2 | StructuredCoT only fills `goal` field | Missing constraint/unknown/risk signals |

## Development Notes

- Python 3.11+ required
- Ruff configured for line-length 100, target-version py311
- Use `pip install -e ".[dev]"` for ruff + pytest
- For embeddings support: `pip install -e ".[embeddings]"` (sentence-transformers + torch)
- The `models/` directory is a symlink to a shared model cache on this machine
- All configuration via environment variables (see `.env.example`), no config files
