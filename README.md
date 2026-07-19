# Enhancing DDoS Detection & Mitigation Model for Network

---

## Project Structure

```
DDoS Detection Model/
├── scripts/
│   ├── ddos-svm-detection.py          # Sprint 3: Detection Model
│   ├── ddos-mitigation-model.py       # Sprint 4: Mitigation V1 (Dataset)
│   ├── ddos-ns3-mitigation.py         # Sprint 5: Mitigation V2 (NS-3)
│   └── NS3-OUTPUT/                    # Sprint 5 outputs (logs + charts)
├── dataset/
│   └── cicddos2019_dataset.csv        # CICDDoS2019 training dataset
├── flows.csv                          # NS-3 extracted traffic (215 flows)
├── output/                            # Sprint 3 outputs (pkl files + charts)
├── Output Mitigation Dataset/         # Sprint 4 outputs (logs + charts)
├── requirements.txt
└── README.md
```

> **Note:** Trained model files (`.pkl`) and the raw dataset are excluded from this repo via `.gitignore` due to file size. Run Sprint 3 locally to regenerate them before running Sprints 4 and 5.

---

## Prerequisites

- Windows 10/11 (or any OS with Python support)
- Python 3.12

---

## Setup

Clone the repository:

```
git clone https://github.com/<your-username>/<your-repo>.git
cd "DDoS Detection Model"
```

Create and activate a virtual environment:

```
python -m venv venv
.\venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal prompt.

Install dependencies:

```
pip install -r requirements.txt
```

Then navigate to the scripts folder:

```
cd scripts
```

---

### Sprint 3 — Detection Model

```
python ddos-svm-detection.py
```

- Loads and preprocesses the CICDDoS2019 dataset
- Selects 23 features and trains an SVM classifier (RBF kernel)
- Training takes approximately 3–5 minutes
- Saves model, scaler, and feature names as `.pkl` files to `output/`
- Generates confusion matrix, ROC curve, and metrics chart to `output/`

---

### Sprint 4 — Mitigation Model V1 (Dataset Traffic)

```
python ddos-mitigation-model.py
```

- Loads the trained `.pkl` files from `output/`
- Simulates 4,000 flows (2,000 benign + 2,000 attack) with 50 IPs
- Classifies each flow and applies threshold-based IP blocklist
- Saves logs, report, and visualizations to `Output Mitigation Dataset/`

---

### Sprint 5 — Mitigation Model V2 (NS-3 Traffic)

```
python ddos-ns3-mitigation.py
```

- Loads the trained `.pkl` files from `output/`
- Loads NS-3 extracted traffic from `flows.csv`
- Retrains the SVM on combined CICDDoS2019 + NS-3 data (50,215 samples)
- Classifies each NS-3 flow and applies the blocklist mechanism
- Saves logs, report, and visualizations to `NS3-OUTPUT/`

---

## Run Order

The scripts must be run in this order:

1. `ddos-svm-detection.py` (generates the `.pkl` files needed by the next scripts)
2. `ddos-mitigation-model.py` (loads `.pkl` files from step 1)
3. `ddos-ns3-mitigation.py` (loads `.pkl` files from step 1 + `flows.csv`)

---

## Expected Results

| Metric       | Sprint 3 (Detection) | Sprint 4 (V1) | Sprint 5 (V2) |
|--------------|----------------------|----------------|----------------|
| Accuracy     | 99.42%               | 99.24%         | 99.53%         |
| Precision    | 98.68%               | 98.68%         | 100.00%        |
| Recall       | 99.72%               | 99.72%         | 97.14%         |
| F1-Score     | 99.20%               | 99.20%         | 98.55%         |
| FPR          | 1.20%                | 1.20%          | 0.00%          |

---

## Notes

- All scripts use `random_state=42` for reproducibility — results will be identical on every run.
- The NS-3 simulation script (`syn-flood-sim.cc`) and PCAP feature extraction script (`pcap_to_csv.py`) were run separately on Ubuntu/WSL2 and are not included in this repo. The extracted output (`flows.csv`) is included.
- If you encounter a `ModuleNotFoundError`, ensure the virtual environment is activated (`.\venv\Scripts\activate`) and dependencies are installed (`pip install -r requirements.txt`).
