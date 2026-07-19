# Enhancing DDoS Detection & Mitigation Model for Network
## Final Year Project — Ahmed Zain (s082256)
## Supervisor: Ibrahim Thoriq | MNU FEST | July 2026

---

## Project Structure

```
DDoS Detection Model/
├── scripts/
│   ├── ddos-svm-detection.py          # Sprint 3: Detection Model
│   ├── ddos-mitigation-model.py       # Sprint 4: Mitigation V1 (Dataset)
│   └── ddos-ns3-mitigation.py         # Sprint 5: Mitigation V2 (NS-3)
├── dataset/
│   └── cicddos2019_dataset.csv        # CICDDoS2019 training dataset
├── flows.csv                          # NS-3 extracted traffic (215 flows)
├── venv/                              # Python virtual environment
├── output/                            # Sprint 3 outputs (pkl files + charts)
├── Output Mitigation Dataset/         # Sprint 4 outputs (logs + charts)
├── NS3-OUTPUT/                        # Sprint 5 outputs (logs + charts)
└── README.md
```

---

## Prerequisites

- Windows 10/11
- Python 3.12 (included in venv)

No additional installations are needed. All dependencies are included in the virtual environment.

---

## How to Run

Open Command Prompt or PowerShell. Navigate to the project folder:

```
cd "D:\DDoS Detection Model"
```

Activate the virtual environment:

```
.\venv\Scripts\activate
```

You should see `(venv)` appear at the beginning of your terminal prompt.

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
- Saves model, scaler, and feature names as .pkl files to `output/`
- Generates confusion matrix, ROC curve, and metrics chart to `output/`

---

### Sprint 4 — Mitigation Model V1 (Dataset Traffic)

```
python ddos-mitigation-model.py
```

- Loads the trained .pkl files from `output/`
- Simulates 4,000 flows (2,000 benign + 2,000 attack) with 50 IPs
- Classifies each flow and applies threshold-based IP blocklist
- Saves logs, report, and visualizations to `Output Mitigation Dataset/`

---

### Sprint 5 — Mitigation Model V2 (NS-3 Traffic)

```
python ddos-ns3-mitigation.py
```

- Loads the trained .pkl files from `output/`
- Loads NS-3 extracted traffic from `flows.csv`
- Retrains the SVM on combined CICDDoS2019 + NS-3 data (50,215 samples)
- Classifies each NS-3 flow and applies the blocklist mechanism
- Saves logs, report, and visualizations to `NS3-OUTPUT/`

---

## Run Order

The scripts must be run in this order:

1. `ddos-svm-detection.py` (generates the .pkl files needed by the next scripts)
2. `ddos-mitigation-model.py` (loads .pkl files from step 1)
3. `ddos-ns3-mitigation.py` (loads .pkl files from step 1 + flows.csv)

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
- The NS-3 simulation script (`syn-flood-sim.cc`) and PCAP feature extraction script (`pcap_to_csv.py`) were run separately on Ubuntu/WSL2 and are not included in this artifact. The extracted output (`flows.csv`) is included.
- If you encounter a `ModuleNotFoundError`, ensure the virtual environment is activated (`.\venv\Scripts\activate`).
