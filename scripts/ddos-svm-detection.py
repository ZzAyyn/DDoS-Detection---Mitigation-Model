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

print(f"\nFirst 5 rows:")
print(df.head())

# Show column names
print(f"\nColumn Names ({len(df.columns)} total):")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# Show data types
print(f"\nData Types:")
print(df.dtypes.value_counts())

# Show the label column 
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

columns_to_drop = []
for col in ["Flow ID", "Source IP", "Destination IP", "Timestamp",
            "SimillarHTTP", "Unnamed: 0", "Class"]:
    if col in df.columns:
        columns_to_drop.append(col)

print(f"\nDropping non-useful columns: {columns_to_drop}")
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Convert all feature columns to numeric (some may be strings)
feature_cols = [col for col in df.columns if col != LABEL_COL]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Handle infinite values — replace with NaN, then handle
print(f"\nInfinite values found: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handle missing values
print(f"Missing values found: {df.isnull().sum().sum()}")

# Show columns with missing values
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print(f"\nColumns with missing values:")
    print(missing_cols)

# Drop rows with missing values
rows_before = len(df)
df.dropna(inplace=True)
rows_after = len(df)
print(f"\nRows before cleaning: {rows_before:,}")
print(f"Rows after cleaning:  {rows_after:,}")
print(f"Rows removed: {rows_before - rows_after:,}")


print(f"\nConverting labels to binary (BENIGN=0, DDoS=1)...")
print(f"Original labels: {df[LABEL_COL].unique()}")

df["Binary_Label"] = df[LABEL_COL].apply(lambda x: 0 if x.lower() == "benign" else 1)

print(f"\nBinary Class Distribution:")
print(df["Binary_Label"].value_counts())
print(f"  0 = BENIGN (Normal Traffic)")
print(f"  1 = DDoS (Attack Traffic)")


# ============================================================================
# STEP 3: Feature Selection
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Feature Selection")
print("=" * 70)

# Separate features (X) and labels (y)
X = df.drop(columns=[LABEL_COL, "Binary_Label"])
y = df["Binary_Label"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label vector shape: {y.shape}")

# Remove any remaining non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\nRemoving non-numeric columns: {non_numeric}")
    X.drop(columns=non_numeric, inplace=True)

# Remove constant features (features that have only one value — useless)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"Removing constant features: {constant_cols}")
    X.drop(columns=constant_cols, inplace=True)

# Remove highly correlated features (keep one of each pair with >0.95 correlation)
print(f"\nRemoving highly correlated features (threshold > 0.95)...")
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
print(f"Highly correlated features to remove: {len(high_corr_cols)}")
X.drop(columns=high_corr_cols, inplace=True)

print(f"\nFinal number of features: {X.shape[1]}")
print(f"Features used:")
for i, col in enumerate(X.columns):
    print(f"  {i+1}. {col}")


# ============================================================================
# STEP 4: Train-Test Split
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Train-Test Split (80% Training, 20% Testing)")
print("=" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # For reproducibility
    stratify=y           
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

# Save the model 
joblib.dump(svm_model, f"{OUTPUT_DIR}/svm_ddos_model.pkl")
print(f"Saved: svm_ddos_model.pkl")

# Save the scaler 
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
print(f"Saved: scaler.pkl")

# Save the feature names 
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