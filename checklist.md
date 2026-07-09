# Sebuleni Pro Max AI Specification Review Checklist

## Source Control

- [ ] `sebuleni_srs.html` is treated as authoritative.
- [ ] `sebuleni_master_implementation_plan.html` is treated as authoritative.
- [ ] Conflict precedence is explicitly stated in favor of the SRS and master plan.
- [ ] No unapproved requirement, API, module, workflow, or implementation order has been added.

## Scope And Boundaries

- [ ] v1 in-scope items are explicitly documented.
- [ ] v1 out-of-scope items are explicitly documented.
- [ ] No trade execution behavior is included.
- [ ] Single-instrument training and inference is preserved.
- [ ] Single-machine deployment is preserved.
- [ ] Offline training, live inference, and controlled retraining review are preserved.

## Domain Semantics

- [ ] Active timeframe stack is exactly `1m`, `5m`, `15m`, `1h`, `4h`, `1d`.
- [ ] Event semantics are absolute move magnitude regardless of direction.
- [ ] Reference price is the latest closed-bar close.
- [ ] Event start is the threshold-cross bar.
- [ ] Maturity is maximum excursion within horizon.
- [ ] Lowest and highest reachable price boundaries are preserved.
- [ ] Inference uses fully closed `1m` bars only.

## Contracts And Validation

- [ ] Configuration and validation contracts are explicitly defined.
- [ ] Deterministic preprocessing rules are explicitly defined.
- [ ] Leakage-prevention rules are explicitly defined.
- [ ] Train-versus-inference feature parity requirements are explicitly defined.
- [ ] Fail-fast behavior is preserved for malformed data, invalid dtype, invalid dimensions, missing files, missing config, timestamp errors, and schema violations.
- [ ] No silent drops, no silent repair, and no silent fallback behavior are permitted.

## Interfaces

- [ ] REST API endpoints are listed.
- [ ] Prediction request contract is listed.
- [ ] Prediction response contract is listed.
- [ ] CLI workflows are listed.
- [ ] Batch report generation is preserved.

## Implementation Order

- [ ] Units `U1` through `U12` are preserved exactly.
- [ ] Dependencies are preserved exactly.
- [ ] No unit is moved ahead of its approved dependencies.
- [ ] Phase grouping matches the master implementation plan.

## Testing

- [ ] Required tests by phase are listed.
- [ ] Cross-cutting unit, integration, regression, performance, and failure-mode testing requirements are listed.
- [ ] Rolling walk-forward validation is preserved.
- [ ] Training, validation, test, and inference isolation is preserved.

## Review Gate

- [ ] Specification artifacts are ready for review.
- [ ] Implementation remains paused pending approval.
