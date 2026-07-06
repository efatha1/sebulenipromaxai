# Sebuleni Pro Max AI (ADSIP) — v1 Operator Guide

This repository implements the v1 system described by the authoritative documents:

- [sebuleni_srs.html](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/sebuleni_srs.html)
- [sebuleni_master_implementation_plan.html](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/sebuleni_master_implementation_plan.html)

If there is any conflict between this README and any implementation detail, the two documents above are the source of truth.

## Purpose and Scope (v1)

Sebuleni Pro Max AI is an offline-training + live-inference forecasting and explanation system operating on one instrument at a time over a fixed multi-timeframe stack: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`.

In-scope and out-of-scope constraints are defined in the SRS and summarized in [spec.md](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/spec.md#L26-L50).

Hard v1 constraints include:

- No trade execution, broker integration, order routing, or automated position management.
- Inference uses fully closed `1m` bars only.
- Deterministic preprocessing and reproducibility are enforced by design.
- Retraining remains review-gated; auto-promotion is forbidden.

## Repository Layout

- [config/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/config)
  - [base.yaml](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/config/base.yaml): baseline config template.
  - [instrument.example.yaml](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/config/instrument.example.yaml): example instrument override.
- [training/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training): U1–U5, U9, U12 implementation modules.
- [models/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/models): U6–U8 model backbone + heads + explanation contracts.
- [inference/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/inference): U10 runtime feature parity + inference pipeline + reporting.
- [api/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/api): U11 FastAPI surface over shared services.
- [tests/unit/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/tests/unit) and [tests/integration/](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/tests/integration): validation gates for each unit.

## Setup

### Python

- Python 3.10+

### Install dependencies

```bash
py -3 -m pip install -r requirements.txt
py -3 -m pip install -r requirements-dev.txt
```

## Configuration

Configuration is YAML-driven and validated into an immutable runtime config contract (U1):

- Loader: [load_config](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/config_loader.py#L73-L103)
- Schema: [RuntimeConfig](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/config_schema.py)

### Environment overrides

Environment overrides use the prefix `SEBULENI__` (double underscore separators) and are applied deterministically:

- Implementation: [training/config_loader.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/config_loader.py)

## CLI (U11)

The CLI provides operator workflows required by the SRS (train/evaluate/predict/report/retraining/inspection):

- Entry point: [training/cli.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/cli.py)

### Commands

#### `train`

Runs training/evaluation from a serialized U9 training bundle and writes an evaluation summary artifact. This command never promotes an active model.

Inputs:
- `--config-path`: YAML config path
- `--bundle-path`: path to a serialized training bundle
- `--evaluation-output-path`: output JSON path for evaluation summary

#### `evaluate`

Same execution surface as `train`, intended for explicit evaluation runs (also never promotes).

#### `predict`

Runs inference using an active-model manifest:

- `--active-model-manifest-path` must exist and is enforced to match training metadata (lookbacks and config hash).

#### `generate-report`

Generates a batch report from approved prediction outputs only (does not re-run inference):

- `--responses-path`: JSON file containing a single prediction response object or a list of them.

#### `request-retraining`

Writes a retraining request artifact into the configured output directory (review-gated).

#### `inspect-model`

Reads active-model metadata via the shared adapter.

## REST API (U11)

FastAPI surface required by the SRS:

- App factory: [create_app](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/api/main.py)
- Routes: [api/routes.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/api/routes.py)

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /models/current`
- `POST /reports/generate`
- `POST /retraining/request`

Reporting requires passing approved prediction outputs (responses) rather than raw requests:

- Request schema: [GenerateReportRequest](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/api/routes.py#L17-L22)

## Model Lifecycle and Review Gate (U12)

### Active-model manifest

The prediction surfaces (API + CLI) load the active model from a manifest persisted on disk:

- Manifest helpers: [api/dependencies.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/api/dependencies.py#L201-L225)
- Loader enforces config hash parity and lookback metadata parity: [inference/model_store.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/inference/model_store.py#L39-L75)

### Review-gated approval

Candidate models are never auto-promoted.

- Explicit approval function: [approve_candidate](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/review.py#L35-L116)
- Scheduled reviews only write recommendations and explicitly record `auto_promotion_performed: False`:
  - [training/scheduler.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/scheduler.py#L41-L125)

## Determinism and Leakage Prevention

The implementation follows the master plan’s constraints on deterministic preprocessing, strict temporal isolation, and no future leakage:

- Deterministic resampling: [training/resample.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/resample.py)
- Causal feature generation and registry parity: [training/features.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/features.py), [training/feature_registry.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/feature_registry.py)
- Label semantics: [training/labeling.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/labeling.py)
- Walk-forward isolation with purge gaps to prevent horizon leakage: [training/folds.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/folds.py#L46-L84)
- Closed-bar-only inference: [training/data_quality.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/training/data_quality.py) and [inference/runtime_features.py](file:///c:/Users/Spirt%20Embassy/Documents/sebuleni/inference/runtime_features.py#L114-L130)

## Quality Gates

### Static analysis (ruff)

```bash
py -3 -m ruff check .
```

### Type checking (mypy)

```bash
py -3 -m mypy
```

### Unit tests

```bash
py -3 -m pytest tests/unit
```

### Integration tests

```bash
py -3 -m pytest tests/integration
```

## Technical Notes / Known Operator Considerations

- The CLI `train`/`evaluate` workflows currently operate on a serialized “training bundle” artifact (dataset, folds, retrieval memory). This artifact is treated as an internal orchestration input and is exercised through integration tests. If you want a first-class “build bundle from CSV” CLI workflow, add it only if/when the SRS is extended to specify it explicitly.

