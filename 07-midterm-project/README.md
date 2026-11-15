# Mobile Price Classification API Midterm Project ML Zoomcamp 2026

An end-to-end Machine Learning classification project for Midterm Project of ML Zoocamp 2026. This project focus on predicting the commercial price tier of mobile phones based on their hardware specifications (from 0 to 3), packaged as a containerized **FastAPI** web service.

---

## 1. Problem Statement
In a competitive consumer electronics market, pricing new mobile devices accurately is critical to balancing profit margins and market share. Rather than estimating exact retail prices—which fluctuate heavily by brand and region—this project frames pricing as a **4-class classification problem** (`price_range`: `0` = Low Cost, `1` = Intermediate-Low Cost, `2` = Intermediate-High Cost, `3` = High Cost).

The goal is to build an interpretable, reproducible ML pipeline and deploy it as a REST API capable of classifying any mobile device into its corresponding price tier from its technical specifications.

---

## 2. Architecture Overview

```text
+-------------------+       +------------------------+       +-------------------------+

|   Raw Dataset     |       |   Training Pipeline    |       |   Production Web API    |
|   (train.csv)     | ----> |      (train.py)        | ----> |  (predict.py/serve.py)  |
+-------------------+       +------------------------+       +-------------------------+
  • 2,000 devices             • Drop 0-dim anomalies           • FastAPI + Uvicorn
  • 20 hardware specs         • 1.5*IQR outlier filter         • Port 9696 exposed
  • 4 balanced classes        • StandardScaler + SVC           • POST /predict
                              • Exports pipeline.bin &         • GET  /health
                                feature_defaults.json
```

---

## 3. Dataset Description
The dataset comes from Kaggle's [Mobile Price Classification](https://kaggle.com) repository (`train.csv`):
* **Size:** 2,000 rows and 21 columns (20 input features + 1 target).
* **Target Variable (`price_range`):** `0` (Low Cost), `1` (Intermediate-Low Cost), `2` (Intermediate-High Cost), `3` (High Cost).
* **Features (20 hardware & connectivity attributes):**
  * *Continuous/Discrete Numeric:* `ram` (MB), `battery_power` (mAh), `int_memory` (GB), `px_height`, `px_width`, `sc_h` (cm), `sc_w` (cm), `mobile_wt` (g), `clock_speed` (GHz), `m_dep` (cm), `n_cores`, `pc` (primary camera MP), `fc` (front camera MP), `talk_time` (hrs).
  * *Binary Flags (`0`/`1`):* `blue` (Bluetooth), `dual_sim`, `four_g`, `three_g`, `touch_screen`, `wifi`.

---

## 4. EDA & Preprocessing Summary
* **Class Balance & Missing Values:** The dataset contains 0 missing values, 0 duplicate rows, and a perfectly balanced target distribution (500 samples per class).
* **Key Predictive Signal:** `ram` exhibits an overwhelming positive correlation with `price_range` ($r \approx 0.92$), followed by `battery_power`, `px_width`, and `px_height` ($r \approx 0.15\text{–}0.20$).
* **Anomaly & Outlier Removal (preprocessing):**
  * Removed rows with physically impossible zero values in screen dimensions (`px_height == 0` and `sc_w == 0`).
  * Filtered extreme statistical outliers across continuous features using the $1.5 \times \text{IQR}$ rule.
* **Feature Engineering Decision:** No synthetic features were added. All 20 features are already clean numerical quantities or binary indicators in standard real-world units, preserving maximum interpretability without introducing multicollinearity.


---

## 5. Modeling Approach & Metrics
The cleaned dataset was split using stratified sampling into **80% Train / 10% Validation / 10% Test**. Hyperparameters were tuned on the Train/Validation sets optimizing for **Multi-class One-vs-Rest AUC-ROC (`roc_auc_ovr`)**, and final generalization was evaluated on the untouched Test set:

| Model | Best Hyperparameters | Val AUC-ROC | Test AUC-ROC | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Support Vector Machine (SVC)** | `C=1`, `kernel='linear'`, `gamma='scale'` | **0.9969** | **1.0000** | **98.89%** |
| **Gradient Boosting** | `learning_rate=0.1`, `max_depth=3`, `n_estimators=200` | 0.9812 | 0.9942 | 93.33% |
| **Random Forest** | `max_depth=None`, `min_samples_split=2`, `n_estimators=300` | 0.9709 | 0.9884 | 93.33% |
| **Baseline (Shallow Tree)** | `max_depth=3` | 0.8885 | 0.9270 | 79.44% |

* **Winner Selection:** **Linear SVC (`C=1`)** achieved the highest Validation AUC-ROC (`0.9969`), a perfect **1.0000 Test AUC-ROC**, and **98.89% Test Accuracy** (178/180 test samples classified correctly), confirming that the underlying hardware-to-price boundary is strongly linear.

* **Winner Selection:** **Linear SVC (`C=1`)** achieved the highest Validation AUC-ROC (`0.9969`), a perfect **1.0000 Test AUC-ROC**, and **98.89% Test Accuracy** (178/180 test samples classified correctly), confirming that the underlying hardware-to-price boundary is strongly linear.

### Feature Importance Analysis (Winning Model)
To verify model interpretability, both **Standardized Linear Coefficients** ($\text{Mean } |\text{coef}|$) and **Permutation Importance** (drop in Test `roc_auc_ovr` over 15 repeats) were computed on the hold-out test set:

| Rank | Feature | Mean \|Coefficient\| (Scaled) | Permutation Importance (AUC Drop) |
| :---: | :--- | :---: | :---: |
| **1** | `ram` | **4.4949** | **0.4613 ± 0.0312** |
| **2** | `battery_power` | **1.0959** | **0.0877 ± 0.0152** |
| **3** | `px_width` | **0.6678** | **0.0240 ± 0.0050** |
| **4** | `px_height` | **0.6668** | **0.0249 ± 0.0057** |
| **5** | `mobile_wt` | **0.1958** | **0.0015 ± 0.0006** |

* **Key Takeaway:** `ram` alone accounts for a ~0.46 drop in Test AUC-ROC when permuted, followed by `battery_power` and screen resolution (`px_width`, `px_height`). Secondary attributes (`mobile_wt`, `int_memory`, connectivity flags) act as fine-grained tie-breakers near class decision boundaries, directly validating the initial EDA correlations.

---

## 6. Project Structure

```text
.
├── notebook.ipynb          # EDA, preprocessing experiments, and model comparison
├── train.py                # Reproducible training pipeline & artifact exporter
├── predict.py              # FastAPI app, Pydantic validation, and inference logic
├── serve.py                # Uvicorn web server entrypoint (port 9696)
├── requirements.txt          # Production API & training dependencies
├── requirements-notebook.txt # Notebook EDA & visualization dependencies
├── Dockerfile              # Container definition
├── train.csv               # Raw training dataset
├── pipeline.bin            # Serialized StandardScaler + SVC model (generated by train.py)
├── feature_defaults.json   # Training medians for inference fallback (generated by train.py)
└── model_metadata.json     # SHA-256 hashes & evaluation metrics (generated by train.py)
```

---

## 7. How to Run Locally

1. **Install dependencies:**
   * *For training and running the API (production):*
     ```bash
     pip install -r requirements.txt
     ```
   * *Optional — To reproduce the EDA, plots, and model selection in `notebook.ipynb`:*
     ```bash
     pip install -r requirements-notebook.txt
     ```


2. **Train the model and generate artifacts (`pipeline.bin`, `feature_defaults.json`, `model_metadata.json`):**
   ```bash
   python train.py --data train.csv
   ```

3. **Start the web service on port `9696`:**
   ```bash
   python serve.py
   ```
   Interactive API documentation (Swagger UI) is available at `http://localhost:9696/docs`.

---

## 8. How to Run via Docker

1. **Build the Docker image:**
   ```bash
   docker build -t mobile-price-api .
   ```

2. **Run the container:**
   ```bash
   docker run -it --rm -p 9696:9696 mobile-price-api
   ```

---

## 9. API Usage Examples

### Health Check (`GET /health`)
```bash
curl http://localhost:9696/health
```

### Predict Price Tier (`POST /predict`)
*(Note: Any omitted fields automatically fall back to training-set medians via `feature_defaults.json`.)*

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
       "battery_power": 1850,
       "blue": 1,
       "clock_speed": 2.4,
       "dual_sim": 1,
       "fc": 8,
       "four_g": 1,
       "int_memory": 64,
       "m_dep": 0.5,
       "mobile_wt": 145,
       "n_cores": 8,
       "pc": 16,
       "px_height": 1080,
       "px_width": 1920,
       "ram": 3600,
       "sc_h": 15,
       "sc_w": 7,
       "talk_time": 18,
       "three_g": 1,
       "touch_screen": 1,
       "wifi": 1
     }' \
     http://localhost:9696/predict
```

**Example Response:**
```json
{
  "price_range": 3,
  "price_label": "High Cost",
  "confidence": 0.998412,
  "class_probabilities": {
    "0": 0.0,
    "1": 0.000015,
    "2": 0.001573,
    "3": 0.998412
  }
}
```

---

## 10. Known Limitations & Next Steps
* **Synthetic Dataset Dynamics:** The underlying dataset exhibits synthetic, near-linear boundaries heavily dominated by RAM capacity. Real-world smartphone pricing also depends on brand equity, processor generation, OS ecosystem, and release year.
* **Calibration of SVC Probabilities:** `SVC(probability=True)` relies on internal Platt scaling (5-fold cross-validation), which adds minor training overhead and can slightly smooth probability estimates on small datasets.
* **Next Steps:**
  * Add automated unit and integration tests (`pytest` + FastAPI `TestClient`) in a CI workflow.
  * Implement request logging and data drift monitoring (e.g., tracking incoming `ram` and `battery_power` distributions against `feature_defaults.json`).
  * Deploy the container to a cloud service (such as AWS ECS, Google Cloud Run, or Render) with HTTPS enabled.
