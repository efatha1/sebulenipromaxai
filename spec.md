# Sebuleni Pro Max AI Project Specification

## Authoritative Sources

- Primary source of truth: `sebuleni_srs.html`
- Primary source of truth: `sebuleni_master_implementation_plan.html`
- Conflict rule: if any conflict exists between this specification and any other project file, `sebuleni_srs.html` and `sebuleni_master_implementation_plan.html` take precedence.

## Specification Purpose

- Define the complete v1 project specification for Sebuleni Pro Max AI from the approved SRS and master implementation plan only.
- Preserve implementation units `U1` through `U12` exactly in dependency order.
- Preserve v1 scope boundaries, semantics, validation rules, leakage-prevention rules, testing requirements, and operational constraints without adding unapproved workflows.

## v1 Product Definition

- Product name: `Sebuleni Pro Max AI`
- v1 operating model: offline training, live inference, and controlled retraining with manual plus scheduled review.
- Deployment target: single-machine deployment hosting all v1 services.
- User model: single physical operator.
- Data scope: one selected instrument per training, validation, test, and live deployment cycle.
- Input source: user-provided `1m` OHLC data only.
- Active timeframe stack: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- Higher timeframes: deterministically resampled from the `1m` source.

## v1 In Scope

- `1m` OHLC ingestion and deterministic preprocessing.
- Calendar-aware and session-aware normalization with configurable timezone, weekend policy, holiday policy, and source-data timezone.
- Deterministic resampling from `1m` into `5m`, `15m`, `1h`, `4h`, and `1d`.
- Deterministic OHLC-derived and calendar-derived feature engineering.
- Self-supervised labeling using approved event semantics.
- Rolling walk-forward training and validation.
- Multi-timeframe predictive modeling over the active timeframe stack.
- Prediction outputs for event probability, confidence, lowest reachable price, highest reachable price, event start, maturity, duration, and grounded explanation.
- Historical analog retrieval and deterministic grounded explanation generation.
- REST API, CLI workflows, and batch report generation.
- Manual and scheduled retraining workflows with review-gated promotion.
- Structured logging, monitoring, scheduled reporting, and rollback-safe review handling.

## Explicit v1 Out Of Scope

- Trade execution, broker integration, order routing, or risk management automation.
- Cross-instrument joint training in v1.
- Distributed training or cloud-native production infrastructure.
- Real-time inference on incomplete bars.
- Unbounded online self-adaptation without review.
- Multi-user authentication or role-based access control beyond single-user local operation.
- Dashboard UI in v1.

## Core Domain Semantics

- Event type: absolute move magnitude regardless of direction.
- Reference price: latest close of the latest fully closed bar.
- Event trigger: an event occurs if future highest high or future lowest low produces an absolute excursion from the latest closed-bar close that crosses the configured price-difference threshold within the selected horizon.
- Event start: the first future bar where the threshold-crossing excursion occurs.
- Maturity: the point of maximum excursion within the prediction horizon.
- Boundary meaning: lowest and highest reachable prices within the horizon.
- Horizon modes: single-horizon per run and configuration-driven multi-horizon mode are both required.
- Inference bar eligibility: only fully closed `1m` bars may be used in live inference.
- Low-support behavior: processable but weak-evidence cases may return low-confidence advisory output rather than hard refusal.

## System Invariants

- Training, validation, test, and inference remain isolated.
- No future information may be used in preprocessing, labeling, feature generation, evaluation, retrieval memory, or inference.
- All preprocessing is deterministic and reproducible.
- Random seed is configurable.
- CPU execution is required; CUDA detection must be graceful when available.
- Non-deterministic algorithms are disallowed unless explicitly enabled by approved requirements.
- All mutable behavior is configuration-driven.
- No hardcoded file paths, hyperparameters, dataset names, URLs, output directories, or credentials.
- No silent drops, silent data repair, silent timestamp correction, or silent fallback behavior.
- All public inputs are validated and fail early with actionable diagnostics.

## Configuration And Validation Contracts

### Required Configuration Coverage

All mutable system behavior must be file-driven and configurable. At minimum, configuration must cover:

- instrument identifier and data source paths
- source-data timezone and runtime timezone
- session calendar definitions and daily close time
- weekend and holiday policies
- resampling targets
- feature toggles and deterministic derived feature set
- event thresholds and horizon configuration list
- walk-forward fold settings
- training hyperparameters and random seed
- top-K analog count
- API host/port and report output settings
- scheduled retraining and reporting schedules

### Validation Contract Rules

- Validate schema before any training or inference entry point proceeds.
- Validate timestamp order, duplicates, missing bars, timezone configuration, and instrument/session configuration.
- Reject malformed files, malformed bars, invalid dtypes, invalid dimensions, unresolved timezone values, invalid horizons, and invalid thresholds.
- Reject incomplete recent bars for live inference.
- Expose actionable diagnostics through API and CLI exceptions.
- Use one validated runtime configuration contract across training, inference, API, CLI, reporting, and retraining review flows.

## Deterministic Preprocessing Rules

- Ingest `1m` OHLC bars for one selected instrument at a time.
- Normalize source timezone and runtime timezone according to configuration.
- Apply configured session definitions, weekend policy, holiday policy, and DST-aware daily close handling.
- Deterministically resample the validated `1m` stream into `5m`, `15m`, `1h`, `4h`, and `1d` using OHLC-consistent aggregation.
- Preserve timestamp correctness and bar alignment across all modeled timeframes.
- Generate deterministic OHLC-derived and calendar/time-context features only from information available at or before the reference timestamp.
- Build aligned feature matrices and windows keyed by timeframe and reference timestamp.
- Ensure identical valid inputs produce identical preprocessing outputs.

## Leakage-Prevention Rules

- No future information in preprocessing, features, labels, evaluation, retrieval, or inference.
- Feature generation must be causal.
- Label generation must use approved future-window semantics only for supervised targets, never for inference-time features.
- Walk-forward folds must be chronological and temporally isolated.
- No overlap-induced leakage across fold boundaries.
- No reused future-aware state across train, validation, and test partitions.
- Retrieval memory for explanations must be built from training-only historical memory, never from future or evaluation leakage.
- Inference must use only recent fully closed bars and the active approved model artifact.

## Train Versus Inference Feature Parity Requirements

- `U2` and `U3` define the canonical data and feature contracts.
- Training and inference must consume the same deterministic preprocessing semantics, timeframe definitions, calendar handling, and feature definitions.
- Runtime feature assembly in inference must preserve the same feature meaning, ordering, alignment, and window semantics used during training.
- Any configured feature toggle available in training must be interpreted identically in inference.
- Feature parity includes the active timeframe stack: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`.
- Feature parity verification is required before model promotion or live usage.

## External Interfaces

### REST API

Required endpoints:

- `GET /health`
- `POST /predict`
- `GET /models/current`
- `POST /reports/generate`
- `POST /retraining/request`

### Prediction Request Contract

- `instrument_id`
- `bars_1m`: fully closed `1m` OHLC bars only
- `horizon_mode`: `single` or `multi`
- `horizon_bars` when single-horizon mode is used
- `threshold`
- `top_k_analogs`

### Prediction Response Contract

- event probability
- confidence score
- lowest reachable price
- highest reachable price
- event start estimate
- maturity estimate
- duration estimate
- low-confidence advisory flag
- top-K analogs
- summary statistics
- grounded natural-language explanation

### CLI Workflows

- `train`
- `evaluate`
- `predict`
- `generate-report`
- `request-retraining`
- `inspect-model`

## Deployment And Operations

- v1 deployment target is one machine hosting inference services, scheduled training and retraining review pipelines, batch reporting, local artifact storage, logging, and any local database services where applicable.
- Services should default to local binding unless explicitly configured otherwise.
- Structured logs are required for preprocessing, resampling, feature generation, training, evaluation, inference, reporting, and retraining requests.
- Every model run must log configuration hash, instrument, date range, walk-forward fold identifier, metrics, and artifact locations.
- Every inference request must log request ID, reference timestamp, horizon mode, threshold, latency, degraded-mode flag, and model version.
- Retraining is review-gated and must recommend review rather than auto-promote new models.

## Implementation Units

Implementation order must remain exactly `U1` through `U12` in the approved dependency chain. No later unit may be implemented before its dependencies are complete and verified.

### U1 Configuration and contracts

- Objective: define all configuration schemas, data contracts, and validation models for deterministic operation.
- Scope: YAML schema loading, typed config objects, environment overrides, request/response contracts, and shared validation logic.
- Public interfaces: `load_config()`, `validate_config()`, typed config classes, API request/response models.
- Inputs: YAML files, environment variables, API payloads, CLI arguments.
- Outputs: validated immutable runtime configuration and shared contracts.
- Dependencies: none.
- Acceptance: invalid config fails early; all runtime modules consume one validated config object.

### U2 OHLC ingestion, calendar normalization, and resampling

- Objective: create a deterministic ingestion and resampling pipeline for the active multi-timeframe stack built from `1m` OHLC input.
- Scope: input validation, timezone normalization, session handling, weekend and holiday rules, DST-safe daily close handling, and OHLC aggregation from `1m` into `5m`, `15m`, `1h`, `4h`, and `1d`.
- Public interfaces: `load_ohlc_frame()`, `normalize_calendar()`, `resample_timeframes()`, `validate_bar_sequence()`
- Inputs: validated config and raw OHLC files.
- Outputs: validated `1m` bars and derived `5m`, `15m`, `1h`, `4h`, and `1d` bars aligned for multi-timeframe modeling.
- Dependencies: `U1`
- Acceptance: no timestamp misalignment, no silent corrections, deterministic identical output for identical input.

### U3 Deterministic feature engineering

- Objective: generate reproducible OHLC-derived and calendar-derived features for each modeled timeframe.
- Scope: returns, ranges, wick/body ratios, rolling statistics, session flags, and multi-timeframe alignment-ready feature windows.
- Public interfaces: `build_features()`, `build_windows()`, `list_enabled_features()`
- Inputs: resampled OHLC frames from `U2` and config feature toggles.
- Outputs: aligned feature matrices and window tensors keyed by timeframe and reference timestamp.
- Dependencies: `U1`, `U2`
- Acceptance: all feature generation is causal, deterministic, and explainable.

### U4 Label generation and horizon engine

- Objective: generate event, boundary, start, maturity, and duration labels for single-horizon and multi-horizon modes.
- Scope: threshold-cross event logic, boundary extraction, event start definition, maturity detection, ambiguous-sample handling, and config-driven horizon list support.
- Public interfaces: `generate_labels()`, `resolve_horizons()`, `classify_ambiguity()`
- Inputs: reference bars, future windows, threshold config, horizon config.
- Outputs: labeled datasets for each horizon and task head.
- Dependencies: `U1`, `U2`, `U3`
- Acceptance: labels match approved semantics exactly and stay within ambiguity thresholds.

### U5 Walk-forward split and evaluation scaffolding

- Objective: implement the official rolling walk-forward validation protocol with strict temporal isolation.
- Scope: fold generation, data isolation, fold metadata tracking, and reusable evaluation contract.
- Public interfaces: `build_walk_forward_folds()`, `fold_iterator()`, `validate_temporal_isolation()`
- Inputs: labeled dataset and split configuration.
- Outputs: chronological train/validation/test fold definitions.
- Dependencies: `U1`, `U2`, `U3`, `U4`
- Acceptance: no data leakage across any fold boundary.

### U6 Shared multi-timeframe backbone

- Objective: implement the shared backbone that learns multi-timeframe OHLC dynamics across `1m`, `5m`, `15m`, `1h`, `4h`, and `1d`.
- Scope: per-timeframe encoder modules, cross-timeframe fusion, latent projection, seed handling, CPU/GPU support, and active participation of both `1m` and `5m` in the learned hierarchy.
- Public interfaces: `Backbone.forward()`, `encode_timeframe()`, `fuse_latents()`
- Inputs: aligned feature windows and training config.
- Outputs: shared latent state for downstream heads.
- Dependencies: `U1`, `U3`, `U5`
- Acceptance: forward pass is deterministic under fixed seed and works on CPU and CUDA.

### U7 Prediction heads and regime logic

- Objective: implement event, boundary, timing, confidence, and regime heads on top of the shared latent state.
- Scope: task-specific heads, multi-task loss orchestration, confidence output, regime scoring.
- Public interfaces: `predict_event()`, `predict_boundaries()`, `predict_timing()`, `predict_confidence()`
- Inputs: shared latent state and labels from `U4`.
- Outputs: supervised prediction outputs needed for inference.
- Dependencies: `U4`, `U6`
- Acceptance: heads train together and expose stable typed outputs for inference.

### U8 Explanation retrieval and grounded narration

- Objective: provide historical analog retrieval and deterministic explanation generation.
- Scope: latent export, FAISS indexing, top-K retrieval, summary statistics, deterministic language rendering.
- Public interfaces: `build_retrieval_index()`, `retrieve_analogs()`, `render_explanation()`
- Inputs: latent states, historical outcome metadata, prediction outputs, `top_k_analogs` setting.
- Outputs: top-K analog list, summary statistics, deterministic grounded explanation.
- Dependencies: `U6`, `U7`
- Acceptance: no unexplained high-confidence predictions and deterministic explanation for identical inputs.

### U9 Training orchestration and evaluation runner

- Objective: provide end-to-end training, fold evaluation, checkpointing, and metrics reporting.
- Scope: Lightning trainer wiring, checkpoint save/load, fold loop, acceptance metric calculation, MLflow logging.
- Public interfaces: `run_training()`, `run_evaluation()`, `select_candidate_model()`
- Inputs: config, folds, features, labels, models.
- Outputs: trained artifacts, evaluation summaries, candidate model metadata.
- Dependencies: `U5`, `U6`, `U7`, `U8`
- Acceptance: walk-forward training and evaluation produce reproducible metrics and artifacts.

### U10 Inference service and prediction pipeline

- Objective: serve inference from fully closed bars through a stable local prediction pipeline.
- Scope: recent-bar validation, runtime feature assembly, model loading, low-confidence advisory logic, response formatting.
- Public interfaces: `predict()`, `load_active_model()`, `build_runtime_window()`
- Inputs: recent `1m` OHLC bars, config, active model artifacts.
- Outputs: prediction payload ready for API, CLI, or report use.
- Dependencies: `U2`, `U3`, `U6`, `U7`, `U8`, `U9`
- Acceptance: inference uses only closed bars and meets latency thresholds on target hardware.

### U11 REST API, CLI, and batch reporting

- Objective: expose all operator workflows through stable external interfaces.
- Scope: FastAPI endpoints, CLI commands, report generation, structured output files, and operator-friendly error surfaces.
- Public interfaces: REST endpoints and CLI commands defined in the SRS.
- Inputs: prediction requests, report parameters, retraining requests, command-line arguments.
- Outputs: JSON responses, CLI output, report files.
- Dependencies: `U1`, `U9`, `U10`
- Acceptance: all required surfaces operate against the same inference and training contracts.

### U12 Monitoring, scheduled jobs, and controlled retraining review

- Objective: implement operational monitoring, scheduled reporting, scheduled retraining review jobs, and rollback-safe review flows.
- Scope: latency monitors, drift monitors, schedule runner, retraining request queue, promotion review logic, local artifact retention.
- Public interfaces: `evaluate_live_health()`, `request_retraining()`, `run_scheduled_review()`, `approve_candidate()`
- Inputs: logged metrics, candidate artifacts, schedule config, retraining request parameters.
- Outputs: operational alerts, review artifacts, approved or rejected candidate decisions.
- Dependencies: `U9`, `U10`, `U11`
- Acceptance: retraining remains controlled, review-gated, auditable, and non-destructive.

## Dependency Order

The approved dependency chain is:

- `U1`
- `U2`
- `U3`
- `U4`
- `U5`
- `U6`
- `U7`
- `U8`
- `U9`
- `U10`
- `U11`
- `U12`

No phase or unit reordering is permitted.

## Phase Plan

- Phase 1: `U1`, `U2`, `U3`
- Phase 2: `U4`, `U5`
- Phase 3: `U6`, `U7`
- Phase 4: `U8`, `U9`
- Phase 5: `U10`, `U11`
- Phase 6: `U12`

## Required Tests By Phase

### Phase 1

- Unit tests: config, schema, timezone, DST, resampling, feature determinism
- Integration tests: raw OHLC to aligned multi-timeframe feature windows
- Regression focus: timestamp correctness and no silent drops

### Phase 2

- Unit tests: threshold-cross semantics, maturity detection, fold generation
- Integration tests: features plus labels plus walk-forward partitions
- Regression focus: no label leakage and no fold overlap

### Phase 3

- Unit tests: model shape, loss, device, determinism, calibration hooks
- Integration tests: one fold training plus evaluation pipeline
- Regression focus: training stability and task-output consistency

### Phase 4

- Unit tests: retrieval determinism, explanation grounding, contradiction guards
- Integration tests: offline predictions plus analog evidence generation
- Regression focus: no future contamination in retrieval index

### Phase 5

- Unit tests: API validation, CLI parsing, report serialization
- Integration tests: train, load, predict, and generate reports end to end
- Regression focus: prediction payload compatibility and latency bounds

### Phase 6

- Unit tests: threshold alerts, schedule policies, approval guards
- Integration tests: scheduled review, retraining request, candidate approval or rejection
- Regression focus: no unsafe promotion and no broken review chain

## Cross-Cutting Test Requirements

- Unit tests for resampling, timezone handling, session segmentation, feature engineering, labeling, model interfaces, explanation summarization, and API request validation.
- Walk-forward evaluation tests verifying strict temporal isolation.
- Integration tests from OHLC ingestion to prediction output.
- Regression tests for deterministic identical-input identical-output explanation behavior.
- Performance tests for API latency, batch throughput, and feature computation cost.
- Failure-mode tests for malformed input, missing bars, weak analog support, and invalid configuration.

## Approval Gate

- This specification is complete only if it remains consistent with `sebuleni_srs.html` and `sebuleni_master_implementation_plan.html`.
- Any requirement not present in those documents is excluded from implementation scope until the authoritative documents are amended.
- Implementation must pause for review of this specification before coding begins.
