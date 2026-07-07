# Trae Code prompt pack

This file contains exact prompts to paste into Trae Code from the beginning of implementation to project completion.

These prompts are designed to force Trae Code to:

- use `sebuleni_srs.html` as the main requirements source
- use `sebuleni_master_implementation_plan.html` as the execution source
- follow the approved implementation order
- avoid scope drift
- enforce deterministic preprocessing
- prevent future leakage
- implement and test one unit at a time

Use the prompts in the order shown below.

---

## Before you paste anything

In Trae Code:

1. open the project in `Code` mode
2. make sure these files are visible in the project:
   - `sebuleni_srs.html`
   - `sebuleni_master_implementation_plan.html`
3. make sure Trae can read them before implementation starts

If you use project Rules in Trae, set them first. If not, the prompts below repeat the critical guardrails.

---

## Prompt 0 - Session anchor prompt

Paste this first at the start of a new implementation session:

```text
You are implementing Sebuleni Pro Max AI strictly from the approved project documents.

Authoritative sources:
1. sebuleni_srs.html
2. sebuleni_master_implementation_plan.html

Rules you must follow in every response and code change:
- Use the SRS and master implementation plan as the source of truth.
- Do not invent requirements, APIs, modules, or workflows not present in those documents.
- Do not change implementation order.
- Do not skip ahead to later units.
- No future information in preprocessing, features, labels, evaluation, or inference.
- Training, validation, test, and inference must remain isolated.
- All preprocessing must be deterministic.
- Random seed must be configurable.
- Support CPU first and detect CUDA gracefully.
- Do not hardcode file paths, hyperparameters, dataset names, URLs, or output directories.
- Configuration must be file-driven.
- Fail early on malformed data, invalid dtype, invalid dimensions, missing files, missing config, timestamp errors, and schema violations.
- No silent drops, no silent repair, no silent fallback behavior.
- Inference must use only fully closed bars.
- Retraining in v1 is review-gated and must never auto-promote models.
- Keep functions readable, testable, and production quality.
- Before writing code, always restate the current unit objective, files to create or modify, interfaces, dependencies, risks, and verification steps based on the master implementation plan.

From now on, every implementation step must reference and follow:
- sebuleni_srs.html
- sebuleni_master_implementation_plan.html
```

---

## Prompt 1 - Create the system-level spec

Paste this next:

```text
/spec Use sebuleni_srs.html and sebuleni_master_implementation_plan.html as the authoritative sources.
Create a complete project spec for Sebuleni Pro Max AI.
Preserve implementation units U1 through U12 exactly in dependency order.

The spec must explicitly preserve:
- v1 scope and out-of-scope boundaries
- the active timeframe stack: 1m, 5m, 15m, 1h, 4h, 1d
- single-instrument training and inference
- absolute move magnitude event semantics
- latest closed-bar close as reference price
- event start as threshold-cross bar
- maturity as maximum excursion within horizon
- lowest and highest reachable price boundaries
- rolling walk-forward validation
- offline training + live inference + controlled retraining review
- API + CLI + batch report generation
- single-machine deployment
- no trade execution behavior

Highlight in the spec:
1. configuration and validation contracts
2. deterministic preprocessing rules
3. leakage-prevention rules
4. train versus inference feature parity requirements
5. required tests by phase
6. explicit v1 out-of-scope items

If any conflict exists, prefer sebuleni_srs.html and sebuleni_master_implementation_plan.html over any other file.
Pause for review when the spec artifacts are ready.
```

---

## Prompt 2 - Validate the generated spec

Paste this after the `/spec` artifacts are generated:

```text
Review the generated spec against sebuleni_srs.html and sebuleni_master_implementation_plan.html.

Check specifically:
- U1 to U12 are preserved exactly
- dependency order is unchanged
- 1m and 5m are included in the active modeled timeframe stack
- no unauthorized features were added
- no v1 out-of-scope items were pulled into scope
- no trading or execution behavior was introduced
- deterministic preprocessing and leakage-prevention rules are explicit

List any deviation as:
1. file
2. section
3. deviation
4. required correction

If the spec is correct, say it is ready and recommend starting U1 only.
```

---

## Prompt 3 - Create the first execution Todo list

Paste this after the spec is approved:

```text
Create a Todo list for implementation, but activate only U1.

The Todo list must:
- include all units U1 to U12
- mark only U1 as active or in progress
- leave U2 to U12 pending
- reflect the dependency order from sebuleni_master_implementation_plan.html

For U1, break the work into concrete sub-tasks for:
- config schema
- config loader
- immutable runtime config objects
- shared contracts
- API request/response schemas
- validation failure handling
- unit tests
- integration tests
```

---

## Prompt 4 - Plan U1

```text
/plan Implement U1 Configuration and contracts first, using sebuleni_srs.html and sebuleni_master_implementation_plan.html as the authoritative sources.

The plan must include:
- objective
- scope
- files to create or modify
- public interfaces
- inputs
- outputs
- dependencies
- internal components
- acceptance criteria
- unit testing requirements
- integration testing requirements
- potential risks

Do not include implementation for U2 or later units.
Do not include model code.
Pause after generating the plan.
```

---

## Prompt 5 - Execute U1

Paste this after you approve the U1 plan:

```text
Implement U1 exactly as defined in the approved plan and the authoritative documents:
- sebuleni_srs.html
- sebuleni_master_implementation_plan.html

Constraints:
- do not modify unrelated files
- do not start U2
- use typed configuration objects
- validate all public inputs
- fail early with meaningful exceptions
- use structured logging
- keep code deterministic
- keep code testable
- add pytest unit tests
- add integration tests for config loading into training and inference entry points

Before changing files:
1. list the files you will create or modify
2. list the public interfaces you will implement
3. list the validations you will enforce
4. list the tests you will add

Then implement U1 completely.
After implementation, run:
- syntax checks
- import validation
- tests relevant to U1

Then report:
- changed files
- implemented interfaces
- test results
- remaining risks

Do not continue to U2.
```

---

## Prompt 6 - U1 verification gate

```text
Review U1 against the implementation plan and SRS.

Verify:
- all required interfaces exist
- config is file-driven
- no hardcoded paths or hyperparameters were introduced
- invalid input fails early
- unit tests pass
- integration tests pass
- documentation was updated if needed

If U1 is not complete, list only the missing items and fix them.
If U1 is complete, mark U1 completed and move only U2 to active.
```

---

## Prompt 7 - Plan U2

```text
/plan Implement U2 OHLC ingestion, calendar normalization, and resampling using the approved U1 contracts and the authoritative documents.

Use these documents as the source of truth:
- sebuleni_srs.html
- sebuleni_master_implementation_plan.html

The plan must include exact treatment of:
- 1m source OHLC ingestion
- active timeframe stack: 1m, 5m, 15m, 1h, 4h, 1d
- deterministic resampling from 1m into 5m, 15m, 1h, 4h, 1d
- user-configurable source-data timezone
- user-configurable runtime timezone
- market session boundaries
- weekend handling
- holiday handling
- DST handling
- daily bar close at 5:00 PM New York time
- duplicate timestamp failure
- missing bar handling
- no silent correction

Pause after generating the plan.
```

---

## Prompt 8 - Execute U2

```text
Implement U2 exactly as approved.

Requirements:
- follow U1 contracts
- do not start U3
- validate OHLC schema strictly
- validate timestamps strictly
- implement deterministic resampling
- preserve causal correctness
- no silent corrections
- add structured logging
- add pytest unit tests
- add integration tests for instrument config + OHLC input -> aligned multi-timeframe output

Before editing, list:
- files to modify
- interfaces to implement
- validation cases
- resampling rules
- tests to add

Then implement U2 completely, run relevant tests, and report:
- changed files
- test results
- unresolved risks
```

---

## Prompt 9 - Verify U2 and activate U3

```text
Review U2 against the SRS and implementation plan.

Verify:
- 1m and 5m are both included correctly
- 5m, 15m, 1h, 4h, and 1d are resampled deterministically from 1m
- DST and session handling are explicit
- duplicate timestamps fail early
- timestamp misalignment is not silently repaired
- unit and integration tests pass

If complete, mark U2 completed and move only U3 to active.
Do not start U4 or later.
```

---

## Prompt 10 - Plan U3

```text
/plan Implement U3 Deterministic feature engineering using the approved U1 and U2 outputs.

Use the SRS and master implementation plan as authoritative.

The plan must include:
- features for every modeled timeframe: 1m, 5m, 15m, 1h, 4h, 1d
- raw OHLC usage
- deterministic derived price-action features
- calendar and time features
- causal rolling logic only
- no future leakage
- training and inference feature parity
- feature registry or equivalent mechanism
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 11 - Execute U3

```text
Implement U3 exactly as approved.

Requirements:
- do not start U4
- all features must be deterministic
- all rolling logic must be causal
- no future information may be used
- preserve train/inference parity
- add unit tests for feature correctness and determinism
- add integration tests for aligned feature window generation

Before editing, list:
- files to modify
- feature families to implement
- leakage risks
- planned tests

Then implement U3, run relevant tests, and report:
- changed files
- added feature groups
- test results
- remaining risks
```

---

## Prompt 12 - Verify U3 and activate U4

```text
Review U3 against the approved plan and SRS.

Verify:
- all feature generation is deterministic
- no future leakage exists
- feature behavior is consistent across modeled timeframes
- unit tests pass
- integration tests pass

If complete, mark U3 completed and move only U4 to active.
```

---

## Prompt 13 - Plan U4

```text
/plan Implement U4 Label generation and horizon engine using the approved semantics from sebuleni_srs.html and sebuleni_master_implementation_plan.html.

The plan must preserve:
- absolute move magnitude regardless of direction
- latest closed-bar close as the reference price
- future highest high / lowest low excursion semantics
- event start = first threshold-cross bar
- maturity = maximum excursion within horizon
- boundary outputs = lowest and highest reachable prices within horizon
- support for both single-horizon and multi-horizon modes
- config-driven fixed horizons list
- ambiguity handling
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 14 - Execute U4

```text
Implement U4 exactly as approved.

Requirements:
- do not start U5
- preserve label semantics exactly from the SRS
- use config-driven horizons
- support both single-horizon and multi-horizon modes
- classify ambiguity explicitly
- add unit tests for threshold crossing, event start, maturity, boundaries, and ambiguity
- add integration tests for stable label generation from approved feature windows and future bars

Before editing, list:
- files to modify
- exact label semantics you will implement
- ambiguity rules
- planned tests

Then implement U4, run tests, and report:
- changed files
- label semantics implemented
- test results
- unresolved risks
```

---

## Prompt 15 - Verify U4 and activate U5

```text
Review U4 against the SRS and implementation plan.

Verify:
- labels use latest closed-bar close as reference
- event is absolute excursion regardless of direction
- event start is first threshold-cross bar
- maturity is maximum excursion within horizon
- boundary outputs are lowest and highest reachable prices
- ambiguity handling is explicit
- unit and integration tests pass

If complete, mark U4 completed and move only U5 to active.
```

---

## Prompt 16 - Plan U5

```text
/plan Implement U5 Walk-forward split and evaluation scaffolding using the authoritative documents and the approved outputs of U1-U4.

The plan must include:
- official rolling walk-forward validation
- strict temporal isolation
- no overlap leakage
- fold metadata
- reusable evaluation contract
- unit tests and integration tests

Pause after generating the plan.
```

---

## Prompt 17 - Execute U5

```text
Implement U5 exactly as approved.

Requirements:
- do not start U6
- preserve strict temporal isolation
- no reused future information
- no reused scaler state across fold boundaries unless explicitly allowed by the SRS
- add unit tests for chronology and overlap prevention
- add integration tests for full walk-forward partition generation

Before editing, list:
- files to modify
- fold semantics
- isolation guarantees
- planned tests

Then implement U5, run tests, and report:
- changed files
- fold behavior
- test results
- remaining risks
```

---

## Prompt 18 - Foundation gate before model work

Paste this before allowing any model implementation:

```text
Perform a hard gate review for U1 through U5.

Use sebuleni_srs.html and sebuleni_master_implementation_plan.html as the source of truth.

Verify all of the following:
- U1 is complete and tested
- U2 is complete and tested
- U3 is complete and tested
- U4 is complete and tested
- U5 is complete and tested
- no leakage has been found in preprocessing, labels, or fold generation
- train/inference feature parity is preserved by design
- all interfaces needed for model work are stable

If any item is incomplete, do not start U6.
List only the missing items and fix them first.
If all items pass, mark U1-U5 complete and move only U6 to active.
```

---

## Prompt 19 - Plan U6

```text
/plan Implement U6 Shared multi-timeframe backbone using the approved outputs of U1-U5 and the authoritative documents.

The plan must preserve:
- active timeframe stack: 1m, 5m, 15m, 1h, 4h, 1d
- direct participation of both 1m and 5m in the learned hierarchy
- deterministic behavior under fixed seed
- CPU-first support with graceful CUDA detection
- no serving or API work yet
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 20 - Execute U6

```text
Implement U6 exactly as approved.

Requirements:
- do not start U7
- preserve fixed-seed determinism as far as allowed
- support CPU and graceful CUDA detection
- keep interfaces narrow and testable
- add shape tests, device tests, and determinism-oriented tests

Before editing, list:
- files to modify
- model interfaces
- device behavior
- planned tests

Then implement U6, run relevant tests, and report:
- changed files
- backbone interfaces
- test results
- remaining risks
```

---

## Prompt 21 - Plan U7

```text
/plan Implement U7 Prediction heads and regime logic using the approved backbone and label semantics.

The plan must cover:
- event head
- boundary head
- timing head
- confidence head
- regime logic
- task losses
- acceptance-aligned metrics
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 22 - Execute U7

```text
Implement U7 exactly as approved.

Requirements:
- do not start U8
- preserve approved event and boundary semantics
- keep outputs typed and stable
- expose interfaces required for offline evaluation and later inference
- add tests for output ranges, shapes, loss behavior, and interface stability

Before editing, list:
- files to modify
- heads to implement
- outputs to expose
- planned tests

Then implement U7, run tests, and report:
- changed files
- head interfaces
- test results
- remaining risks
```

---

## Prompt 23 - Plan U8

```text
/plan Implement U8 Explanation retrieval and grounded narration using training-only retrieval memory and the approved SRS requirements.

The plan must preserve:
- top-K analog retrieval
- summary statistics
- short grounded natural-language explanation
- deterministic output for identical input
- no future contamination in retrieval
- no unexplained high-confidence predictions
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 24 - Execute U8

```text
Implement U8 exactly as approved.

Requirements:
- do not start U9
- retrieval index must remain training-only during evaluation
- explanation output must be grounded, deterministic, and auditable
- add tests for retrieval determinism, explanation consistency, and contradiction prevention

Before editing, list:
- files to modify
- retrieval contract
- explanation contract
- planned tests

Then implement U8, run tests, and report:
- changed files
- explanation behavior
- test results
- remaining risks
```

---

## Prompt 25 - Plan U9

```text
/plan Implement U9 Training orchestration and evaluation runner using the approved outputs of U1-U8.

The plan must include:
- training orchestration
- walk-forward fold execution
- checkpointing
- evaluation summaries
- metrics logging
- candidate model selection
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 26 - Execute U9

```text
Implement U9 exactly as approved.

Requirements:
- do not start U10
- preserve walk-forward evaluation as the official standard
- keep checkpointing reproducible
- log metrics in a structured and traceable way
- add tests for metrics, checkpoints, and candidate selection

Before editing, list:
- files to modify
- training runner interfaces
- evaluation outputs
- planned tests

Then implement U9, run tests, and report:
- changed files
- training/evaluation interfaces
- test results
- remaining risks
```

---

## Prompt 27 - Offline model gate before inference work

```text
Perform an offline candidate gate review before any inference or serving implementation begins.

Verify:
- U6 through U9 are complete and tested
- walk-forward evaluation completes successfully
- metrics and artifacts are reproducible
- explanation output is grounded and deterministic
- all interfaces required for inference are stable

If any item is incomplete, fix only those items.
If complete, mark U6-U9 complete and move only U10 to active.
```

---

## Prompt 28 - Plan U10

```text
/plan Implement U10 Inference service and prediction pipeline using the approved outputs of U1-U9.

The plan must preserve:
- inference from fully closed bars only
- runtime feature parity with training
- low-confidence advisory output
- typed prediction payload
- latency awareness
- no API layer yet beyond the inference contract
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 29 - Execute U10

```text
Implement U10 exactly as approved.

Requirements:
- do not start U11
- inference must reject incomplete bars
- preserve training/inference parity
- produce low-confidence advisory output when evidence is weak
- add tests for closed-bar validation, runtime feature parity, degraded mode, and payload formatting

Before editing, list:
- files to modify
- inference interfaces
- degraded-mode behavior
- planned tests

Then implement U10, run tests, and report:
- changed files
- inference behavior
- test results
- remaining risks
```

---

## Prompt 30 - Plan U11

```text
/plan Implement U11 REST API, CLI, and batch reporting using the approved U10 inference contract and the authoritative documents.

The plan must preserve:
- API endpoints defined by the SRS
- CLI workflows defined by the SRS
- batch report generation
- shared contract parity between API and CLI
- structured errors
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 31 - Execute U11

```text
Implement U11 exactly as approved.

Requirements:
- do not start U12
- API and CLI must use the same inference and training contracts
- report generation must use approved outputs only
- add tests for API validation, CLI parsing, and report serialization
- add end-to-end integration tests across training, prediction, and reporting flows where applicable

Before editing, list:
- files to modify
- public endpoints and commands
- report outputs
- planned tests

Then implement U11, run tests, and report:
- changed files
- API and CLI interfaces
- test results
- remaining risks
```

---

## Prompt 32 - Plan U12

```text
/plan Implement U12 Monitoring, scheduled jobs, and controlled retraining review using the approved documents.

The plan must preserve:
- scheduled reporting
- scheduled retraining review
- manual retraining requests
- no auto-promotion
- operational monitoring
- review-gated approval flow
- unit and integration tests

Pause after generating the plan.
```

---

## Prompt 33 - Execute U12

```text
Implement U12 exactly as approved.

Requirements:
- preserve review-gated retraining
- do not introduce auto-promotion
- add structured logging for alerts, review decisions, and scheduled jobs
- add tests for thresholds, schedules, approval guards, and rollback metadata behavior

Before editing, list:
- files to modify
- monitoring interfaces
- review flow behavior
- planned tests

Then implement U12, run tests, and report:
- changed files
- monitoring and retraining-review behavior
- test results
- remaining risks
```

---

## Prompt 34 - Full-system integration review

```text
Perform a full-system integration review against:
- sebuleni_srs.html
- sebuleni_master_implementation_plan.html

Verify:
- all units U1 through U12 are complete
- dependency order was respected
- no unauthorized scope was added
- no later unit bypassed an earlier dependency
- train/inference parity holds
- deterministic preprocessing is preserved
- no future leakage exists
- API, CLI, and batch reports are consistent
- retraining remains review-gated
- no trading behavior exists anywhere in the system

List any gap by:
1. unit
2. file
3. issue
4. required fix

If gaps exist, fix them without adding new scope.
If no gaps exist, proceed to the final verification stage.
```

---

## Prompt 35 - Final verification and definition of done

```text
Run the final completion review for the entire project using the SRS and master implementation plan as the source of truth.

A unit is complete only if:
- implementation is complete
- static analysis passes
- type checking passes
- unit tests pass
- integration tests pass
- documentation is updated
- public APIs are documented
- code review issues are resolved

Perform this review across U1-U12 and report:
- completed units
- static analysis status
- type checking status
- unit test status
- integration test status
- documentation status
- public API documentation status
- remaining risks or technical debt

If anything is incomplete, fix only the incomplete items.
If everything is complete, produce a final project completion summary referencing the SRS and implementation plan.
```

---

## Prompt 36 - Final project completion summary

Paste this only when everything else is done:

```text
Prepare the final project completion summary.

The summary must:
- reference sebuleni_srs.html and sebuleni_master_implementation_plan.html as the governing documents
- confirm that U1 through U12 were completed in dependency order
- confirm that deterministic preprocessing and no-future-leakage constraints were preserved
- confirm that API, CLI, and batch reports were implemented
- confirm that retraining remains review-gated
- list all major created modules and public interfaces
- list final verification results
- list any approved limitations that remain explicitly out of scope for v1

Do not invent achievements that were not actually completed.
```

---

## Best usage note

If you want the cleanest execution flow in Trae Code, use this pattern for every unit:

1. `/plan` prompt for that unit
2. approve the plan
3. execution prompt for that unit
4. verification prompt for that unit
5. only then activate the next unit

This is the safest way to keep the implementation aligned with:

- `sebuleni_srs.html`
- `sebuleni_master_implementation_plan.html`

