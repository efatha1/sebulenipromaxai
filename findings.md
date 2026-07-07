# Sebuleni Pro Max AI - Key Findings & Analysis

## 1. Overview of the Project Vision

After carefully reviewing the complete conversation and the later clarifications, the core idea is to build an AI platform that learns from **1-minute OHLC price data** for a **single instrument at a time**. The same architecture should be reusable across stocks, forex pairs, or commodities, but each training, validation, testing, and live deployment cycle is specific to one selected instrument. The system should be able to detect changes in price behavior, predict when significant price moves are likely to happen, and explain its reasoning using real historical analogs instead of opaque outputs.

## 2. Key Points Identified (Including Previously Missed Details)

### 2.1 Multi-Task Learning is Central
The original conversation emphasizes using a shared backbone model instead of training separate models for each task. This is important because:
- All tasks learn from the same underlying dynamics
- Reduces the total amount of data needed
- Makes the system more efficient
- Allows knowledge learned for one task to help other tasks

### 2.2 The Importance of Historical Analogs
One of the most unique parts of the proposal is the explanation engine, which doesn't just make up reasons for predictions. Instead, it:
1. Stores the hidden "state" representations of every historical window of data
2. When a new prediction is made, it finds the most similar historical situations
3. It then calculates statistics from those similar situations to explain the prediction
This is a powerful approach because it grounds predictions in actual past events, making the system more trustworthy.

### 2.3 Generic Architecture, Instrument-Specific Execution
The architecture is meant to stay generic enough to handle many instruments and markets, but the actual execution context is now much more precise than the earlier generic framing. The model will operate on one instrument at a time using **timestamp, open, high, low, close** as the raw inputs. That means the system is reusable across domains like stocks, forex, and commodities, but it is not being trained as one cross-instrument model that mixes all pairs together.

### 2.4 The Three Modes of Learning & Novelty Handling
The proposal outlines three distinct learning modes, which also serve as mitigation for novel behavior:
1. **Offline Learning**: Initial training on historical datasets
2. **Online Learning**: Small, periodic updates as new data arrives. The workflow here is key:
   - New data comes in
   - System tries to predict
   - If prediction error suddenly jumps (a sign of novel behavior), it infers a possible regime change
   - It then adapts the model or creates a new regime representation
3. **Continuous Learning**: Full retraining when concept drift exceeds thresholds, retaining previous knowledge to avoid catastrophic forgetting
4. **Gradual Change Handling**: If the change is slow instead of abrupt, it uses time-varying models (like adaptive Kalman filters) to track the evolving parameters
This layered approach allows the system to balance stability and adaptability, and directly addresses the "completely novel behavior" concern by detecting prediction error spikes and adapting accordingly.

### 2.5 The Tech Stack is Well-Considered
Every tool and library mentioned is open-source, well-maintained, and widely used in industry and research:
- PyTorch for deep learning
- pandas/NumPy for data handling
- FAISS for similarity search
- PostgreSQL for metadata storage
- FastAPI for the API
- Dash for the dashboard
- MLflow for experiment tracking

### 2.6 Multi-Timeframe Hierarchical Architecture (Critical Detail I Missed)
One of the most important architectural details is the multi-timeframe design with bidirectional relationships. The system uses five timeframes:
1. 1 day (highest, "galaxy" level—determines main direction)
2. 4 hours ("systems" level)
3. 1 hour ("worlds" level)
4. 15 minutes ("continents" level)
5. 1 minute (lowest level)

The relationships work like this:
- Higher timeframes (like 1 day) give the main direction and stability
- Lower timeframes (like 1 minute) provide detailed context and early signals
- Changes at higher timeframes almost certainly affect lower timeframes
- Lower timeframe patterns can indicate coming changes at higher levels

An important implementation clarification is that the higher timeframes are not assumed to come from independent datasets. They are created by **resampling the 1-minute OHLC stream** into 15-minute, 1-hour, 4-hour, and 1-day bars. This makes timeframe alignment a deterministic part of the pipeline rather than a loose data integration problem.

This structure directly addresses the "multiple regime changes in quick succession" concern because the slower-changing higher timeframes act as anchors, helping the system distinguish between noise and real, sustained regime shifts.

### 2.7 Detailed Pipeline Specifications
The conversation includes clear specifications for both the training pipeline and the live inference pipeline, which is crucial for actual implementation.

## 3. Updated Understanding & Considerations

### 3.1 Clarification on Inputs and Labels
The raw input structure is now clear:
1. **Base data**: `timestamp, open, high, low, close` at the 1-minute level
2. **Derived data**: Higher timeframes built by resampling the 1-minute bars
3. **Temporal features**: time, date, day of week, month, and similar calendar context can still be added as model inputs

The label side is also clearer than before:
1. **Main event definition**: a significant move in price from one point to another, measured as **price difference**
2. **Optional few input labels**: small manual annotations can still help regime understanding, but the main event logic is self-supervised

**Caveat about missing variables**: the project is intentionally restricted to OHLC plus temporal structure. That keeps the model clean and generic, but it also means the system is learning price behavior without volume, order flow, or macro context unless those are explicitly added later.

### 3.2 Self-Supervised Event Detection & Prediction Tasks
- Event detection: Predict whether price will move by more than a **user-specified price difference threshold** within a **user-specified future horizon**
- Boundary prediction: Estimate likely lower and upper **price levels** reachable during the relevant horizon, plus expected move magnitude
- Timing prediction: Estimate when the price move is likely to begin
- Maturity prediction: Estimate when the move is likely to reach its main expansion or maturity point
- All predictions should carry uncertainty bounds and confidence scores

### 3.3 Data Availability
Excellent news: there is 5+ years of historical 1-minute OHLC data available for the relevant instrument. That is strong coverage for training and validation because it should expose the model to multiple market conditions, volatility states, and structural regimes for that one instrument.

### 3.4 Hyperparameter Optimization
This will be discussed separately, so we can defer that planning for now.

### 3.5 Full Original Architecture & Details
The original document has a **7-component modular architecture** (before multi-task refinement):
1. **Dynamics Model** (input: recent OHLC bars across the active timeframes; learns momentum, consolidation, expansion, reversal structure, and regime-sensitive price behavior; outputs hidden state representation)
2. **Regime Model** (detects statistically abnormal prediction errors; outputs regime ID + confidence; sends "don't trust old patterns" signal to others if needed)
3. **Event Detector** (classifier; uses user-defined price-difference threshold + future horizon)
4. **Boundary Predictor** (conditional on Event Detector; outputs lower and upper likely price levels plus expected magnitude)
5. **Timing Model** (predicts expected event start)
6. **Maturity Model** (predicts time to max magnitude after start)
7. **Confidence Model** (assigns confidence intervals to all predictions)

**Communication Flow**: All components share the same hidden representation from Dynamics Model; no redundant pattern learning

**Additional Details**:
- **Project Structure**: Defined (data/, models/, training/, inference/, api/, dashboard/, experiments/, checkpoints/, notebooks/)
- **Mixture of Experts (MoE)**: Multiple experts with gating network; experts can continue learning online
- **Symbolic Regression (Optional)**: If interpretability is important, discover mathematical rules (e.g., y = 0.37*x² + sin(x)) per detected regime
- **Internal Behavioral Modes**: Discovers unsupervised (e.g., stable accumulation, accelerating growth, peak formation, reversal, recovery)
- **Live Operation Example**: Step-by-step scenario of how predictions evolve over time
- **Example Explanation Text**: Exact wording to use for evidence-based explanations
- **Non-trading role**: The model is not a trade execution system. It serves only as a predictive and interpretive assistant.

### 3.6 Remaining Edge Case
- How to handle extended periods of missing data?

## 4. Overall Assessment

The proposed system is well-designed and addresses real limitations of current forecasting and anomaly detection systems. The focus on explainability through historical analogs is particularly innovative and valuable. The modular architecture makes the project manageable, as components can be built and validated independently before being integrated into the full system.
