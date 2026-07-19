import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import joblib
import time
import warnings
import os

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STEP 1: Load and Explore the Dataset
# ============================================================================
print("=" * 70)
print("STEP 1: Loading and Exploring the Dataset")
print("=" * 70)


DATASET_PATH = "../dataset/cicddos2019_dataset.csv"  

print(f"\nLoading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]:,}")
print(f"Total Columns: {df.shape[1]}")

# Show first few rows
print(f"\nFirst 5 rows:")
print(df.head())

# Show column names 
print(f"\nColumn Names ({len(df.columns)} total):")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# Show data types
print(f"\nData Types:")
print(df.dtypes.value_counts())


label_candidates = [col for col in df.columns if "label" in col.lower()]
print(f"\nPossible label columns found: {label_candidates}")

# Strip whitespace from column names 
df.columns = df.columns.str.strip()
print(f"\nAfter stripping whitespace from column names:")
label_candidates = [col for col in df.columns if "label" in col.lower()]
print(f"Label column(s): {label_candidates}")

# Use the label column
LABEL_COL = label_candidates[0] if label_candidates else "Label"
print(f"\nUsing '{LABEL_COL}' as the target label column")

# Show class distribution
print(f"\nClass Distribution:")
print(df[LABEL_COL].value_counts())
print(f"\nClass Distribution (%):")
print(df[LABEL_COL].value_counts(normalize=True).round(4) * 100)


# ============================================================================
# STEP 2: Data Preprocessing (Cleaning)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing (Cleaning)")
print("=" * 70)

# 2a. Drop columns that are NOT useful for ML

columns_to_drop = []
for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp",
            "SimillarHTTP", "Unnamed: 0", "Class"]:
    if col in df.columns:
        columns_to_drop.append(col)

print(f"\nDropping non-useful columns: {columns_to_drop}")
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# 2b. Convert all feature columns to numeric (some may be strings)
feature_cols = [col for col in df.columns if col != LABEL_COL]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2c. Handle infinite values — replace with NaN, then handle
print(f"\nInfinite values found: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2d. Handle missing values
print(f"Missing values found: {df.isnull().sum().sum()}")

# Show columns with missing values
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"\nColumns with missing values:")
    print(missing_cols)

# Drop rows with missing values (safest approach for SVM)
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)
print(f"\nRows before cleaning: {rows_before:,}")
print(f"Rows after cleaning:  {rows_after:,}")
print(f"Rows removed: {rows_before - rows_after:,}")

# 2e. Convert labels to binary for Binary Classification
#     BENIGN = 0, All DDoS attacks = 1
print(f"\nConverting labels to binary (BENIGN=0, DDoS=1)...")
print(f"Original labels: {df[LABEL_COL].unique()}")

df["Binary_Label"] = df[LABEL_COL].apply(lambda x: 0 if x.lower() == "benign" else 1)

print(f"\nBinary Class Distribution:")
print(df["Binary_Label"].value_counts())
print(f"  0 = BENIGN (Normal Traffic)")
print(f"  1 = DDoS (Attack Traffic)")


# ============================================================================
# STEP 3: Feature Selection (Domain-Knowledge-Driven)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Feature Selection (Domain-Knowledge-Driven)")
print("=" * 70)


# Separate all features and labels first
X_all = df.drop(columns=[LABEL_COL, "Binary_Label"])
y = df["Binary_Label"]
total_features_before = X_all.shape[1]

print(f"\nTotal features available before selection: {total_features_before}")


PROTOCOL_FEATURES = [
    "Protocol",            # Transport protocol identifier (TCP/UDP/ICMP)
    "SYN Flag Count",      # SYN flood detection — abnormally high in attacks
    "ACK Flag Count",      # ACK flood / incomplete handshake detection
    "Fwd PSH Flags",       # Push flag behavior in forward direction
    "Fwd Header Length",   # Forward packet header size (crafted packet indicator)
    "Bwd Header Length",   # Backward packet header size (asymmetry indicator)
    "Init Fwd Win Bytes",  # Initial TCP window size (tool fingerprinting)
]

STATISTICAL_FEATURES = [
    "Total Fwd Packets",       # Forward packet count (flood volume)
    "Total Backward Packets",  # Backward packet count (asymmetry detection)
    "Fwd Packets Length Total", # Total forward bytes (attack payload volume)
    "Fwd Packet Length Max",   # Largest forward packet (bandwidth exhaustion)
    "Fwd Packet Length Min",   # Smallest forward packet (minimal packet floods)
    "Fwd Packet Length Std",   # Forward packet size variation (uniformity check)
    "Bwd Packet Length Mean",  # Average backward packet size (response indicator)
    "Flow Bytes/s",            # Throughput rate (volumetric spike detection)
    "Flow Packets/s",          # Packet rate (packets-per-second anomaly)
]


BEHAVIORAL_FEATURES = [
    "Flow Duration",    # Total flow duration (burst vs sustained pattern)
    "Flow IAT Mean",    # Mean packet inter-arrival time (rapid = DDoS)
    "Flow IAT Std",     # Inter-arrival time variation (uniform = DDoS)
    "Bwd IAT Mean",     # Backward inter-arrival time (server response timing)
    "Down/Up Ratio",    # Download/upload asymmetry (key DDoS indicator)
    "Active Mean",      # Mean active time (continuous = DDoS)
    "Active Std",       # Active time variation (uniform = DDoS)
]

# =======================================================================
# COMBINE ALL SELECTED FEATURES
# =======================================================================
SELECTED_FEATURES = PROTOCOL_FEATURES + STATISTICAL_FEATURES + BEHAVIORAL_FEATURES

# Verify all selected features exist in the dataset
print(f"\nVerifying selected features exist in dataset...")
missing_features = [f for f in SELECTED_FEATURES if f not in X_all.columns]
available_features = [f for f in SELECTED_FEATURES if f in X_all.columns]

if missing_features:
    print(f"\n  WARNING: The following expected features are MISSING from the dataset:")
    for f in missing_features:
        print(f"    - {f}")
    print(f"  These features will be skipped. Proceeding with {len(available_features)} features.")
else:
    print(f"  All {len(SELECTED_FEATURES)} selected features found in dataset.")

# Filter X to only use selected features
X = X_all[available_features]

# =======================================================================
# SUMMARY OF DOMAIN-KNOWLEDGE FEATURE SELECTION
# =======================================================================
print(f"\n{'='*60}")
print(f"  FEATURE SELECTION SUMMARY (Domain-Knowledge-Driven)")
print(f"{'='*60}")

print(f"\n  GROUP 1 — PROTOCOL FEATURES ({len(PROTOCOL_FEATURES)} features)")
print(f"  Purpose: Network protocol rules, connection setup, TCP flags")
for i, f in enumerate(PROTOCOL_FEATURES):
    status = "OK" if f in X.columns else "MISSING"
    print(f"    {i+1}. {f} [{status}]")

print(f"\n  GROUP 2 — STATISTICAL FEATURES ({len(STATISTICAL_FEATURES)} features)")
print(f"  Purpose: Volume, size, and rate measurements of traffic flows")
for i, f in enumerate(STATISTICAL_FEATURES):
    status = "OK" if f in X.columns else "MISSING"
    print(f"    {i+1}. {f} [{status}]")

print(f"\n  GROUP 3 — BEHAVIORAL FEATURES ({len(BEHAVIORAL_FEATURES)} features)")
print(f"  Purpose: Timing patterns, inter-arrival intervals, flow behavior")
for i, f in enumerate(BEHAVIORAL_FEATURES):
    status = "OK" if f in X.columns else "MISSING"
    print(f"    {i+1}. {f} [{status}]")

print(f"\n  Total features BEFORE selection: {total_features_before}")
print(f"  Total features AFTER selection:  {len(available_features)}")
print(f"  Features removed: {total_features_before - len(available_features)}")
print(f"{'='*60}")


# ============================================================================
# STEP 4: Train-Test Split (80/20 — Stratified)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Train-Test Split (80% Training, 20% Testing)")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           # Maintain class balance in both sets
)

print(f"\nTraining set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTesting set class distribution:")
print(y_test.value_counts())


# ============================================================================
# STEP 5: Feature Scaling (StandardScaler)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Feature Scaling using StandardScaler")
print("=" * 70)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on training
X_test_scaled = scaler.transform(X_test)         # only transform on testing

print("Feature scaling complete.")
print(f"Training data — Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")


# ============================================================================
# STEP 6: SVM Model Training
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training SVM Model")
print("=" * 70)


MAX_TRAIN_SAMPLES = 50000  # Adjust based on your computer's capability

if len(X_train_scaled) > MAX_TRAIN_SAMPLES:
    print(f"\nDataset is large ({len(X_train_scaled):,} samples).")
    print(f"Using a stratified sample of {MAX_TRAIN_SAMPLES:,} for SVM training.")
    print(f"(SVM has O(n²) to O(n³) time complexity — training on full dataset")
    print(f" would take very long. This is a known SVM limitation mentioned")
    print(f" in your literature review by Sadhwani et al., 2023)")

    # Take a stratified sample
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=MAX_TRAIN_SAMPLES, random_state=42)
    for train_idx, _ in sss.split(X_train_scaled, y_train):
        X_train_subset = X_train_scaled[train_idx]
        y_train_subset = y_train.iloc[train_idx]
else:
    X_train_subset = X_train_scaled
    y_train_subset = y_train

print(f"\nTraining SVM with RBF kernel on {len(X_train_subset):,} samples...")
print(f"This may take several minutes depending on your hardware...\n")

# Initialize and train the SVM model
svm_model = SVC(
    kernel="rbf",       # RBF kernel (best based on literature review)
    C=1.0,              # Regularization strength
    gamma="scale",      # Kernel coefficient
    random_state=42,    # Reproducibility
    probability=True,   # Enable probability estimates (needed for ROC curve)
    verbose=True        # Show training progress
)

start_time = time.time()
svm_model.fit(X_train_subset, y_train_subset)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")


# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Model Evaluation on Test Set")
print("=" * 70)

# Predict on the test set
start_time = time.time()
y_pred = svm_model.predict(X_test_scaled)
prediction_time = time.time() - start_time

print(f"Prediction completed in {prediction_time:.2f} seconds")

# --- 7a. Core Metrics ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")
f1 = f1_score(y_test, y_pred, average="binary")

# False Positive Rate (FPR) — from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)

print(f"\n{'='*50}")
print(f"  EVALUATION RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:           {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision:          {precision:.4f}  ({precision*100:.2f}%)")
print(f"  Recall:             {recall:.4f}  ({recall*100:.2f}%)")
print(f"  F1-Score:           {f1:.4f}  ({f1*100:.2f}%)")
print(f"  False Positive Rate: {fpr:.4f}  ({fpr*100:.2f}%)")
print(f"  Training Time:      {training_time:.2f} seconds")
print(f"  Prediction Time:    {prediction_time:.2f} seconds")
print(f"{'='*50}")

print(f"\nConfusion Matrix:")
print(f"  True Negatives (TN):  {tn:,}  — Benign correctly classified as Benign")
print(f"  False Positives (FP): {fp:,}  — Benign incorrectly classified as DDoS")
print(f"  False Negatives (FN): {fn:,}  — DDoS incorrectly classified as Benign")
print(f"  True Positives (TP):  {tp:,}  — DDoS correctly classified as DDoS")

# --- 7b. Full Classification Report ---
print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"]))


# ============================================================================
# STEP 8: Visualization — Confusion Matrix
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Generating Visualizations")
print("=" * 70)

# 8a. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=["BENIGN", "DDoS"],
    yticklabels=["BENIGN", "DDoS"]
)
plt.title("Confusion Matrix — SVM (RBF Kernel)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print("Saved: confusion_matrix.png")

# 8b. ROC Curve
y_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
fpr_roc, tpr_roc, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr_roc, tpr_roc)

plt.figure(figsize=(8, 6))
plt.plot(fpr_roc, tpr_roc, color="blue", lw=2, label=f"SVM (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve — SVM (RBF Kernel)", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
plt.close()
print("Saved: roc_curve.png")

# 8c. Metrics Bar Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_names, metrics_values, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
for bar, val in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.ylim(0, 1.1)
plt.title("SVM Model Performance Metrics", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metrics_bar_chart.png", dpi=150)
plt.close()
print("Saved: metrics_bar_chart.png")


# ============================================================================
# STEP 9: Save the Trained Model (for Mitigation Sprint)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 9: Saving Trained Model and Scaler")
print("=" * 70)

# Save the model — you'll load this in your mitigation script later
joblib.dump(svm_model, f"{OUTPUT_DIR}/svm_ddos_model.pkl")
print(f"Saved: svm_ddos_model.pkl")

# Save the scaler — needed to scale new incoming traffic the same way
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
print(f"Saved: scaler.pkl")

# Save the feature names — needed to know which features to extract
feature_names = X.columns.tolist()
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")
print(f"Saved: feature_names.pkl")

# Save results to a text file for your report
with open(f"{OUTPUT_DIR}/evaluation_results.txt", "w") as f:
    f.write("DDoS Detection Model — Evaluation Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model: Support Vector Machine (SVM)\n")
    f.write(f"Kernel: RBF (Radial Basis Function)\n")
    f.write(f"Dataset: CICDDoS2019\n")
    f.write(f"Split: 80% Training / 20% Testing (Stratified)\n")
    f.write(f"Training Samples: {len(X_train_subset):,}\n")
    f.write(f"Testing Samples: {len(X_test):,}\n")
    f.write(f"Features Used: {X.shape[1]}\n\n")
    f.write(f"Results:\n")
    f.write(f"  Accuracy:            {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Precision:           {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"  Recall:              {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"  F1-Score:            {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)\n")
    f.write(f"  ROC AUC:             {roc_auc:.4f}\n")
    f.write(f"  Training Time:       {training_time:.2f} seconds\n")
    f.write(f"  Prediction Time:     {prediction_time:.2f} seconds\n\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"  TN={tn:,}  FP={fp:,}\n")
    f.write(f"  FN={fn:,}  TP={tp:,}\n\n")
    f.write(f"Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"]))
print(f"Saved: evaluation_results.txt")


print("\n" + "=" * 70)
print("ALL DONE! Check the 'output' folder for:")
print("  - svm_ddos_model.pkl      (trained model for mitigation)")
print("  - scaler.pkl              (feature scaler)")
print("  - feature_names.pkl       (list of features used)")
print("  - confusion_matrix.png    (for your report)")
print("  - roc_curve.png           (for your report)")
print("  - metrics_bar_chart.png   (for your report)")
print("  - evaluation_results.txt  (summary for your report)")
print("=" * 70)