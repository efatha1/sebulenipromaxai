# Trae Work implementation kickoff plan

## Summary

The most efficient way to start implementation in Trae Work for this project is to use a two-level workflow:

1. Create one persistent system-level artifact with `/spec` using the approved SRS and master implementation plan as the authoritative sources.
2. Implement one bounded implementation unit at a time with `/plan`, following the dependency order already defined in the master plan.

For this project, the correct starting point is not model code. The correct starting point is the deterministic foundation:

- `U1` configuration and contracts
- `U2` OHLC ingestion, session handling, and resampling
- `U3` deterministic feature engineering
- `U4` label generation and horizon logic
- `U5` walk-forward split and evaluation scaffolding

Only after those are stable should model and serving work begin.

## Current State Analysis

### Approved project knowledge already exists

The project already has two authoritative design documents:

- `sebuleni_srs.html`
- `sebuleni_master_implementation_plan.html`

These documents already define:

- v1 scope
- system boundaries
- acceptance criteria
- implementation units `U1` to `U12`
- dependency order
- milestone structure
- testing obligations

This means Trae Work should not be used to rediscover architecture from scratch. It should be used to turn the approved design into a disciplined execution workflow.

### Official Trae Work workflow findings

Based on the official Trae Work documentation:

- `Code` mode is the correct mode for developer implementation work.
- `/spec` is the preferred workflow for larger system-level tasks and creates durable specification artifacts.
- `/plan` is the preferred workflow for medium and bounded implementation tasks and creates plan artifacts under `.trae/documents/`.
- Skills should be used on demand for repeatable specialist workflows, not as a replacement for project requirements.
- Rules should be used to enforce stable project constraints.
- Todos should be used to track bounded execution progress.
- Artifacts should be preserved as project knowledge, rather than re-derived repeatedly in later sessions.

### Project-specific execution risks

The main risk in this project is not starting too slowly. The main risk is starting the wrong layer too early. The highest-risk failure modes are:

- future leakage in preprocessing or labeling
- misaligned resampling across timeframes
- training and inference feature mismatch
- incorrect event semantics
- explanation contamination from future data
- contract drift between CLI, API, training, and inference

Because of those risks, the implementation workflow must prioritize contracts and data correctness before models.

## Proposed Changes

### 1. Create one system-level spec artifact in Trae Work

**Files involved**

- read: `sebuleni_srs.html`
- read: `sebuleni_master_implementation_plan.html`
- generate in Trae Work: `.trae/specs/<task-name>/spec.md`
- generate in Trae Work: `.trae/specs/<task-name>/tasks.md`
- generate in Trae Work: `.trae/specs/<task-name>/checklist.md`

**What**

Use `/spec` once for the entire system to convert the approved SRS and implementation plan into Trae-native planning artifacts.

**Why**

This reduces future prompt repetition, keeps implementation aligned with the approved architecture, and creates a single long-lived reference inside Trae Work.

**How**

In Trae Work `Code` mode, start with a prompt like:

```text
/spec Use sebuleni_srs.html and sebuleni_master_implementation_plan.html as the authoritative sources.
Create a project spec for Sebuleni Pro Max AI.
Preserve implementation units U1-U12 and their dependency order.
Highlight:
1. system contracts
2. config and validation requirements
3. leakage-prevention rules
4. train/inference parity requirements
5. required tests by phase
6. explicit out-of-scope items for v1
If conflicts exist, prefer the SRS and master implementation plan.
```

### 2. Establish project Rules before code generation

**Files involved**

- no repository files should be edited for this step
- Trae Work Rules should be configured in the project context

**What**

Define explicit Rules in Trae Work before implementation begins.

**Why**

This project has strict engineering constraints that Trae should not be allowed to violate opportunistically during code generation.

**How**

Add Rules that enforce:

- no future information
- deterministic preprocessing only
- isolated training, validation, test, and inference
- closed-bars-only live inference
- no hardcoded paths or hyperparameters
- config-driven behavior
- fail-fast validation
- no silent drops or silent repair
- CPU required, CUDA optional and graceful
- review-gated retraining only in v1

### 3. Use `/plan` per implementation unit, not for the entire project at once

**Files involved**

- generated plan artifacts in `.trae/documents/`
- the current approved docs remain read-only references:
  - `sebuleni_srs.html`
  - `sebuleni_master_implementation_plan.html`

**What**

Break implementation into one-unit planning cycles using `/plan`.

**Why**

This minimizes coupling, reduces context overload, improves testability, and matches the dependency graph already approved in the master plan.

**How**

Plan and execute units in this exact order:

1. `U1` Configuration and contracts
2. `U2` OHLC ingestion, calendar normalization, and resampling
3. `U3` Deterministic feature engineering
4. `U4` Label generation and horizon engine
5. `U5` Walk-forward split and evaluation scaffolding
6. `U6` Shared multi-timeframe backbone
7. `U7` Prediction heads and regime logic
8. `U8` Explanation retrieval and grounded narration
9. `U9` Training orchestration and evaluation runner
10. `U10` Inference service and prediction pipeline
11. `U11` REST API, CLI, and batch reporting
12. `U12` Monitoring, scheduled jobs, and controlled retraining review

### 4. Keep one active Todo group at a time

**Files involved**

- no repository files are required for this planning pattern
- use Trae Work Todos only

**What**

Use Todos only for the currently active implementation unit.

**Why**

This keeps the working context tight and prevents Trae from prematurely implementing downstream units.

**How**

For `U1`, the Todo structure should be similar to:

- define config schema fields from SRS
- define immutable runtime config objects
- define shared training and inference contracts
- define API request and response schemas
- add validation failure handling
- add unit tests for config validation
- add integration tests for config loading into entry points

Only when `U1` is fully complete should `U2` become active.

### 5. Use Skills only after planning has narrowed scope

**Files involved**

- no repository edits required for the decision
- project or global skills may be loaded as needed

**What**

Use Skills for bounded specialist tasks after `/plan` has already narrowed the implementation scope.

**Why**

Using Skills too early can push Trae into implementation patterns before the project constraints for the active unit are clear.

**How**

Use Skills for:

- contract-first Python modules
- FastAPI schema patterns
- Typer CLI patterns
- pytest test generation
- structured logging patterns
- deterministic time-series processing patterns

Do not use Skills to redefine the system architecture.

### 6. Use MCP and docs lookup only for library behavior, not project requirements

**Files involved**

- no repository edits required

**What**

Use MCP or official documentation lookup for exact library behavior and integration details.

**Why**

This is useful for correctness, but the project requirements themselves are already settled by the approved SRS and master plan.

**How**

Use external lookup for:

- PyTorch behavior
- FAISS integration details
- FastAPI request patterns
- MLflow usage
- timezone and calendar library behavior

Do not use it to reopen already-approved project decisions.

### 7. Preserve artifact memory instead of re-planning later

**Files involved**

- `.trae/specs/...`
- `.trae/documents/...`

**What**

Keep a durable artifact set inside Trae Work for the life of the project.

**Why**

This reduces repeated analysis and keeps later implementation sessions aligned.

**How**

Recommended artifact set:

- one system-level spec artifact
- one plan artifact per implementation unit
- one acceptance test matrix artifact
- one v1 out-of-scope artifact if needed

### 8. Do not start from API or model code

**Files involved**

- future implementation files are governed by the master plan
- no changes should begin in:
  - `api/`
  - `models/`
  until `U1` to `U5` are approved and stable

**What**

Avoid starting with serving surfaces or model architecture.

**Why**

For this project, the data and semantics foundation is the actual dependency root. Model and serving layers should consume stable contracts, not define them.

**How**

Do not implement:

- FastAPI endpoints
- CLI workflows
- model backbone
- explanation engine

before:

- config schema is stable
- resampling is deterministic
- feature generation is causal
- labels are approved in code form
- walk-forward evaluation is proven leak-free

## Assumptions & Decisions

### Assumptions

- The approved `sebuleni_srs.html` and `sebuleni_master_implementation_plan.html` remain the authoritative design sources.
- Trae Work will be used in `Code` mode for implementation.
- The user wants the most efficient path with the least rework, not the fastest uncontrolled code generation.

### Decisions

- Use `/spec` once at the whole-system level.
- Use `/plan` repeatedly at the implementation-unit level.
- Start implementation with `U1`, not with API or models.
- Treat `1m`, `5m`, `15m`, `1h`, `4h`, and `1d` as the active modeled timeframe stack.
- Keep retraining review-gated in v1.
- Keep the approved docs and generated Trae artifacts as persistent project memory.

## Verification steps

Before implementation starts, verify the following inside Trae Work:

1. The project is opened in `Code` mode.
2. The approved source documents are available and referenced:
   - `sebuleni_srs.html`
   - `sebuleni_master_implementation_plan.html`
3. Project Rules are added for determinism, leakage prevention, fail-fast validation, config-only behavior, and closed-bar inference.
4. A system-level `/spec` artifact has been generated and reviewed.
5. The generated spec preserves `U1` to `U12` and does not introduce unauthorized scope.
6. The first active Todo list contains only `U1`.
7. The first `/plan` artifact is specifically for `U1`.
8. No implementation begins for `U6` to `U12` before `U1` to `U5` are completed and verified.
9. Any conflict between older notes and current project docs is resolved in favor of:
   - `sebuleni_srs.html`
   - `sebuleni_master_implementation_plan.html`

If all nine checks pass, Trae Work is being used in the most efficient and lowest-rework way for this project.
