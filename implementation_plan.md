# Sebuleni Pro Max AI - Implementation Plan

## Phase 0: Project Setup & Foundation (Weeks 1-2)

### 0.1 Project Structure
Create the project directory structure as defined in the original document:
```
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── latent/
├── models/
│   ├── dynamics.py
│   ├── regime.py
│   ├── event.py
│   ├── boundary.py
│   ├── timing.py
│   ├── confidence.py
│   └── explanation.py
├── training/
├── inference/
├── api/
├── dashboard/
├── experiments/
├── checkpoints/
└── notebooks/
```
Initialize git repository and set up DVC (Data Version Control) for large time-series datasets

### 0.2 Environment Setup
- Install Python 3.10+
- Set up a virtual environment
- Install core dependencies: pandas, numpy, torch, pytorch-lightning, scikit-learn, plotly, ruptures, faiss-cpu (or faiss-gpu), dvc, fastapi, uvicorn, dash
- Set up git version control

### 0.3 Data Pipeline Foundation
- **Data Format Specification**:
  - Base source is **1-minute OHLC** for one selected instrument at a time
  - Required raw columns: `timestamp`, `open`, `high`, `low`, `close`
  - Higher timeframes are created by **resampling the 1-minute stream** into `15min`, `1hr`, `4hr`, and `1day`
  - The architecture stays generic, but each training/validation/testing run is instrument-specific
- Build CSV import functionality for the 1-minute OHLC source data
- Implement basic data validation (check temporal ordering for each timeframe)
- Add simple data cleaning (handle missing values with forward fill/backward fill/interpolation)
- **Add temporal feature engineering**: Extract time, date, day of week, month, etc. as independent variables for all timeframes
- Create rolling window generation for each timeframe after resampling
- **Implement deterministic OHLC resampling rules**:
  - `open` = first open in interval
  - `high` = highest high in interval
  - `low` = lowest low in interval
  - `close` = last close in interval
- **Implement timeframe alignment logic**:
  - Each lower timeframe window is aligned to the current (most recent) higher timeframe bar
  - Maintain lookback windows across all timeframes (e.g., last 100 1min bars, last 50 15min bars, last 30 1hr bars, last 20 4hr bars, last 10 1day bars)

**Deliverable**: Working pipeline that ingests 1-minute OHLC, resamples it into all higher timeframes, and aligns all views with temporal features

---

## Phase 1: Multi-Timeframe Hierarchical Dynamics Model (Weeks 3-8)

### 1.1 Model Selection & Architecture Design
- Design a **hierarchical multi-timeframe Transformer with Mixture of Experts (MoE)**
- Lower timeframes (1min, 15min, 1hr, 4hr) feed into higher timeframes (bottom-up context)
- Higher timeframes (1day, 4hr) send guidance to lower timeframes (top-down direction)
- PatchTST is a good starting point for individual timeframe encoders
- Add Mixture of Experts: multiple expert networks, with a gating network that selects the best expert(s) for each detected regime

### 1.2 Initial Implementation
- Implement separate Transformer encoders for each timeframe in PyTorch using OHLC bar sequences as the core inputs
- Build cross-timeframe attention layers to enable bidirectional information flow
- Set up PyTorch Lightning for training organization
- Implement learning of short-term and higher-timeframe price dynamics across all resampled horizons

### 1.3 Progressive Training & Validation
**Phase 1.3.0: Data Splitting & Look-Ahead Bias Prevention (Week 3)**
- **Time-Based Train/Val/Test Split** to prevent future data leakage:
  - Train: First 80% of 5+ years of data (chronological order)
  - Validation: Next 10%
  - Test: Final 10% (holdout, no peeking!)
- **Look-Ahead Bias Safeguards**:
  - All feature engineering uses only past data up to each timestamp
  - No future data allowed in any window generation or training step
  - Explicit tests to verify no leakage

**Phase 1.3.1: Independent Timeframe Training (Weeks 4-5)**
- Train each timeframe encoder independently first to establish baselines
- Use train/val splits
- Validate using MAE and RMSE per timeframe

**Phase 1.3.2: Add Cross-Timeframe Attention (Weeks 6-7)**
- Freeze individual timeframe encoders, add and train cross-timeframe attention layers
- Validate using MAE/RMSE + cross-timeframe prediction consistency score

**Phase 1.3.3: Full Joint Training (Weeks 8-9)**
- Unfreeze all layers, train full hierarchical model end-to-end
- Use mixed precision training for efficiency
- Implement early stopping and learning rate scheduling
- **Synthetic Data Augmentation**: Generate synthetic edge cases (novel regimes, extended missing data) to test robustness

**Deliverable**: Working hierarchical multi-timeframe dynamics model with bidirectional information flow, Mixture of Experts, and look-ahead bias safeguards

---

## Phase 2: Core Prediction Heads (Weeks 9-12)

### 2.1 Event Detection Head
- Implement binary classification head on top of the shared backbone
- **Implement self-supervised event labeling and pre-event pattern learning**:
  - 1. Automatically scan historical OHLC data to find when price moves exceed the user-defined **price-difference threshold**
  - 2. For each detected event, extract the N-step window *before* the move started as the positive training sample
  - 3. Extract non-event windows as negative samples
- Train the model to recognize the pre-event OHLC structures that typically precede significant price moves
- Train using binary cross-entropy loss
- Evaluate using precision, recall, F1-score, and ROC-AUC

### 2.2 Boundary Prediction Head
- Implement quantile regression for lower, median, and upper **future price levels**
- Estimate expected move magnitude as a price difference from the starting point
- Use quantile loss function
- Evaluate using Prediction Interval Coverage Probability (PICP) and Mean Interval Width (MPIW)

### 2.3 Timing Prediction Head
- Implement regression heads for event start, maturity, and duration of the price move
- Use Huber loss (robust to outliers)
- Evaluate using mean absolute timing error

**Deliverable**: Working multi-task OHLC-based model with price-move event, boundary, and timing predictions

---

## Phase 3: Historical Analogy & Explanation System (Weeks 13-16)

### 3.1 Hierarchical Latent Storage
- Modify the training pipeline to save **hierarchical latent vectors** (one per timeframe, plus a combined vector)
- Store vectors in FAISS with timestamp, instrument identifier, and event outcome metadata

### 3.2 Hierarchical Similarity Search
- Integrate FAISS (use HNSW index for fast approximate nearest neighbors)
- Implement two-step retrieval:
  1. First search for similar high-timeframe (1day/4hr) states
  2. Refine results using lower-timeframe (1hr/15min/1min) context
- Retrieve top-N similar historical situations

### 3.3 Explanation Generation
- Calculate statistics from retrieved analogs:
  - Success rate of events
  - Distribution of magnitudes
  - Distribution of timing
- Generate evidence-based explanations including all stats from the original document

**Deliverable**: Working hierarchical explanation system that retrieves and summarizes historical analogs

---

## Phase 4: Regime Detection & Confidence Estimation (Weeks 17-20)

### 4.1 Per-Timeframe Regime Detection + Cross-Timeframe Consistency
- Implement HDBSCAN or DBSCAN clustering **separately per timeframe** on latent vectors
- Add Bayesian online change-point detection using ruptures library for each timeframe
- Check for cross-timeframe regime consistency to reduce false positives
- Create regime identifier and drift score outputs per timeframe and overall

### 4.2 Confidence Estimation
- Implement Monte Carlo Dropout for uncertainty estimation
- Add confidence calibration post-processing using Platt scaling or isotonic regression
- Generate confidence intervals for all predictions (event probability, boundaries, timing, duration)

**Deliverable**: Working per-timeframe regime detection with cross-timeframe consistency and confidence estimation system

---

## Phase 5: API & Dashboard (Weeks 21-24)

### 5.1 API Development
- Build FastAPI backend with /predict endpoint
- Add endpoints for model management and retraining
- Implement request validation and error handling
- Define the inference contract around recent OHLC history for one selected instrument

### 5.2 Dashboard Development
- Create interactive Dash dashboard
- Add visualizations for:
  - Current predictions with confidence intervals
  - Regime history
  - Historical analogs
  - Prediction performance over time
- Make it explicit in the UI that the system is a **predictive assistant only**, not a trade execution engine

**Deliverable**: Working API and interactive dashboard

---

## Phase 6: Continuous Learning & Productionization (Weeks 25-28)

### 6.1 Online & Continuous Learning (Novelty Handling)
- **Implement the original online learning workflow**:
  - Continuously monitor prediction errors across all timeframes
  - If error spikes significantly, flag as possible regime change/novel behavior
  - **Detailed "Adapt Model/Create New Regime" Workflow**:
    1. **Adapt**: Small, incremental update to existing model weights using only recent data (low learning rate, frozen old weights)
    2. **Create New Regime**: If adaptation doesn't reduce error, initialize new expert network (for MoE) or new regime cluster, train on recent data
- Add concept drift detection (for gradual changes) using ADWIN or DDM
- For gradual changes, use time-varying models (adaptive Kalman filters) to track evolving parameters
- Create retraining triggers and pipelines for continuous learning
- **Prevent catastrophic forgetting using**:
  - Elastic Weight Consolidation (EWC)
  - Experience Replay (replay buffer of historical examples)
  - Learning rate constraints for older weights
- **Rollback Strategy** (from original document's "recovery after interruption" and "automatic checkpointing"):
  - Always keep last 5 good checkpoints
  - If online/continuous update degrades performance, automatically roll back to most recent good checkpoint
- **Event Threshold Adaptation**:
  - Allow user to change magnitude threshold without full retraining
  - Recompute event labels and fine-tune only event detection head
- **Production Monitoring & Alerting**:
  - Monitor per-timeframe prediction errors, regime changes, novelty scores, latency
  - Alert if error exceeds threshold, novelty score is high, or latency exceeds budget
- **Latency Budget Breakdown**:
  - Data loading/alignment: <10ms
  - Hierarchical dynamics model: <30ms
  - Regime detection: <5ms
  - All prediction heads: <25ms
  - Historical analogy retrieval: <20ms
  - Explanation generation: <10ms
  - Total: <100ms
- **Extended Missing Data Strategy**:
  - If any timeframe has >1 hour of missing data, flag and use only available higher timeframes for predictions with reduced confidence
  - For gaps >1 day, use last known good state and widen prediction intervals significantly

### 6.2 Production Setup
- Set up PostgreSQL database for metadata and predictions
- Configure MLflow for experiment tracking
- Add logging and monitoring
- Create deployment documentation

**Deliverable**: Production-ready system with continuous learning capabilities

---

## Phase 7: Testing & Validation (Weeks 29-32)

### 7.1 Component Testing
- Test each component individually
- Validate all evaluation metrics

### 7.2 End-to-End Testing
- Test the complete pipeline from data ingestion to explanation
- Validate with the available 5+ years of historical data

### 7.3 User Acceptance Testing
- Gather feedback from potential users
- Iterate on dashboard and API usability

---

## Success Criteria

1. **Dynamics Model**:
   - Per-timeframe MAE/RMSE competitive with baseline time-series models
   - Cross-timeframe prediction consistency score > 0.8
2. **Event Detection**:
   - Per-timeframe F1-score > 0.75 on test data
   - Hierarchical voting F1-score > 0.8
3. **Boundary Prediction**: PICP > 0.9 for 90% confidence intervals
4. **Regime Detection**:
   - Per-timeframe: Detect known regime changes with < 5 time steps delay
   - Cross-timeframe consistency > 0.75
5. **Latency**: End-to-end inference < 100ms
6. **Scalability**: Handle datasets > 10 million observations
7. **Novelty Detection**: Flag 95% of truly novel sequences with < 5% false positive rate

---

## Risk Mitigation

| Risk | Mitigation Strategy |
|------|---------------------|
| Insufficient data | **Not a risk**: 5+ years of historical data is available |
| Overfitting | Implement strong regularization, use dropout, early stopping |
| Computational complexity | Start with smaller models, use mixed precision training |
| Complexity of full system | Build and validate components independently before integration |
