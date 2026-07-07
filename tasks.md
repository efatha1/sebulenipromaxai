# Sebuleni Pro Max AI Specification Tasks

## Execution Rule

- Preserve implementation units `U1` through `U12` exactly in dependency order.
- Do not skip ahead to later units.
- Do not change implementation order.
- Use `sebuleni_srs.html` and `sebuleni_master_implementation_plan.html` as the only authoritative implementation sources.

## Phase 1 - Contracts and deterministic data foundation

### U1 Configuration and contracts

- Objective: define all configuration schemas, data contracts, and validation models for deterministic operation.
- Dependencies: none
- Files: `training/config_loader.py`, `training/config_schema.py`, `api/schemas.py`, `training/contracts.py`, `config/base.yaml`, `config/instrument.example.yaml`
- Verification: schema validation, missing field failure, invalid type failure, environment override tests, end-to-end config load into training and inference entry points

### U2 OHLC ingestion, calendar normalization, and resampling

- Objective: create a deterministic ingestion and resampling pipeline for the active multi-timeframe stack built from `1m` OHLC input.
- Dependencies: `U1`
- Files: `training/data_loader.py`, `training/calendar.py`, `training/resample.py`, `training/data_quality.py`
- Verification: timezone, DST, duplicate timestamp, missing bar, daily close, OHLC aggregation tests, aligned multi-timeframe output integration test

### U3 Deterministic feature engineering

- Objective: generate reproducible OHLC-derived and calendar-derived features for each modeled timeframe.
- Dependencies: `U1`, `U2`
- Files: `training/features.py`, `training/windowing.py`, `training/feature_registry.py`
- Verification: determinism tests, leakage tests, feature correctness tests for small bar sequences, expected window-shape integration test

## Phase 2 - Labeling and evaluation foundation

### U4 Label generation and horizon engine

- Objective: generate event, boundary, start, maturity, and duration labels for single-horizon and multi-horizon modes.
- Dependencies: `U1`, `U2`, `U3`
- Files: `training/labeling.py`, `training/horizons.py`, `training/ambiguity.py`
- Verification: threshold-cross tests, boundary extraction tests, maturity tests, ambiguity tests, stable label integration test across horizons

### U5 Walk-forward split and evaluation scaffolding

- Objective: implement the official rolling walk-forward validation protocol with strict temporal isolation.
- Dependencies: `U1`, `U2`, `U3`, `U4`
- Files: `training/splits.py`, `training/folds.py`, `training/eval_contract.py`
- Verification: fold chronology tests, overlap tests, edge-case final-window tests, reproducible walk-forward partition integration test

## Phase 3 - Core predictive model

### U6 Shared multi-timeframe backbone

- Objective: implement the shared backbone that learns multi-timeframe OHLC dynamics across `1m`, `5m`, `15m`, `1h`, `4h`, and `1d`.
- Dependencies: `U1`, `U3`, `U5`
- Files: `models/backbone.py`, `models/timeframe_encoder.py`, `models/fusion.py`, `models/common.py`
- Verification: shape tests, determinism tests, device fallback tests, backbone pass integration test within expected memory envelope

### U7 Prediction heads and regime logic

- Objective: implement event, boundary, timing, confidence, and regime heads on top of the shared latent state.
- Dependencies: `U4`, `U6`
- Files: `models/event_head.py`, `models/boundary_head.py`, `models/timing_head.py`, `models/confidence_head.py`, `models/regime_head.py`, `models/losses.py`
- Verification: loss tests, output-range tests, head-shape tests, confidence monotonicity sanity tests, one-fold train integration test

## Phase 4 - Explanation and evaluation runner

### U8 Explanation retrieval and grounded narration

- Objective: provide historical analog retrieval and deterministic explanation generation.
- Dependencies: `U6`, `U7`
- Files: `inference/retrieval.py`, `models/explanation.py`, `training/latent_export.py`, `inference/explanation_templates.py`
- Verification: retrieval determinism tests, template grounding tests, contradiction tests, analog evidence integration test using training-only retrieval memory

### U9 Training orchestration and evaluation runner

- Objective: provide end-to-end training, fold evaluation, checkpointing, and metrics reporting.
- Dependencies: `U5`, `U6`, `U7`, `U8`
- Files: `training/trainer.py`, `training/evaluate.py`, `training/metrics.py`, `training/checkpointing.py`
- Verification: metric correctness tests, checkpoint path tests, candidate selection tests, full walk-forward run integration test with artifact emission

## Phase 5 - Operator surfaces

### U10 Inference service and prediction pipeline

- Objective: serve inference from fully closed bars through a stable local prediction pipeline.
- Dependencies: `U2`, `U3`, `U6`, `U7`, `U8`, `U9`
- Files: `inference/predictor.py`, `inference/runtime_features.py`, `inference/model_store.py`, `inference/advisory.py`
- Verification: closed-bar validation tests, payload formatting tests, degraded-mode tests, training-artifact load-and-query integration test for single and multi-horizon inference

### U11 REST API, CLI, and batch reporting

- Objective: expose all operator workflows through stable external interfaces.
- Dependencies: `U1`, `U9`, `U10`
- Files: `api/main.py`, `api/dependencies.py`, `api/routes.py`, `inference/reporting.py`, `training/cli.py`
- Verification: request validation tests, CLI command parsing tests, report serialization tests, end-to-end API and CLI workflow integration tests

## Phase 6 - Operations and controlled retraining

### U12 Monitoring, scheduled jobs, and controlled retraining review

- Objective: implement operational monitoring, scheduled reporting, scheduled retraining review jobs, and rollback-safe review flows.
- Dependencies: `U9`, `U10`, `U11`
- Files: `training/monitoring.py`, `training/retraining.py`, `training/scheduler.py`, `training/review.py`
- Verification: threshold-rule tests, scheduling tests, approval-guard tests, rollback metadata tests, scheduled review pipeline integration test from candidate metrics to approval or rejection

## Stop Conditions

- Do not begin implementation of any unit before all prior dependent units are complete and verified.
- Do not auto-promote retrained models in v1.
- Do not permit incomplete-bar inference.
- Do not introduce trade execution behavior.
- Do not expand beyond single-instrument, single-machine v1 scope.
