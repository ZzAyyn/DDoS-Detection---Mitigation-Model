
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
import time
import random
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# PKL_DIR: where detection model saved its .pkl files (Sprint 3 output)
PKL_DIR = "output"

# OUTPUT_DIR: where this mitigation model saves its own outputs
OUTPUT_DIR = "Output Mitigation Dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# STEP 1: Load Saved Models from Detection Model (Sprint 3)
# ============================================================================
print("=" * 70)
print("STEP 1: Loading Saved Models from Detection Sprint")
print("=" * 70)

# Load the trained SVM model
try:
    svm_model = joblib.load(f"{PKL_DIR}/svm_ddos_model.pkl")
    print("  Loaded: svm_ddos_model.pkl")
except FileNotFoundError:
    print("  ERROR: svm_ddos_model.pkl not found in output/ folder!")
    print("  Please run the detection model (ddos_svm_detection.py) first.")
    exit(1)

# Load the feature scaler
try:
    scaler = joblib.load(f"{PKL_DIR}/scaler.pkl")
    print("  Loaded: scaler.pkl")
except FileNotFoundError:
    print("  ERROR: scaler.pkl not found in output/ folder!")
    print("  Please run the detection model (ddos_svm_detection.py) first.")
    exit(1)

# Load the feature names (23 domain-knowledge-selected features)
try:
    feature_names = joblib.load(f"{PKL_DIR}/feature_names.pkl")
    print(f"  Loaded: feature_names.pkl ({len(feature_names)} features)")
except FileNotFoundError:
    print("  ERROR: feature_names.pkl not found in output/ folder!")
    print("  Please run the detection model (ddos_svm_detection.py) first.")
    exit(1)

print(f"\n  Model type: {type(svm_model).__name__}")
print(f"  Kernel: {svm_model.kernel}")
print(f"  Features expected: {len(feature_names)}")
print(f"  Feature names: {feature_names}")


# ============================================================================
# STEP 2: Prepare Simulation Data
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Preparing Simulation Data")
print("=" * 70)

# Load the dataset (same one used for training)
DATASET_PATH = "../dataset/cicddos2019_dataset.csv"

print(f"\n  Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

# Strip whitespace from column names (same as detection model)
df.columns = df.columns.str.strip()

# Get the label column
LABEL_COL = "Label"
print(f"  Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Drop non-useful columns
columns_to_drop = [col for col in ["Flow ID", "Source IP", "Destination IP",
                                    "Timestamp", "SimillarHTTP", "Unnamed: 0",
                                    "Class"] if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Convert feature columns to numeric
for col in [c for c in df.columns if c != LABEL_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace infinite values with NaN, then drop rows with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Create binary ground truth label
df["Binary_Label"] = df[LABEL_COL].apply(lambda x: 0 if x.lower() == "benign" else 1)

print(f"  After preprocessing: {df.shape[0]:,} rows")

# Sample 2000 benign + 2000 attack flows to simulate incoming traffic
BENIGN_SAMPLE = 2000
ATTACK_SAMPLE = 2000
TOTAL_FLOWS = BENIGN_SAMPLE + ATTACK_SAMPLE

benign_df = df[df["Binary_Label"] == 0].sample(n=BENIGN_SAMPLE, random_state=42)
attack_df = df[df["Binary_Label"] == 1].sample(n=ATTACK_SAMPLE, random_state=42)

# Combine and shuffle to simulate real mixed traffic
sim_df = pd.concat([benign_df, attack_df], ignore_index=True)
sim_df = sim_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Simulation batch created:")
print(f"    Benign flows: {BENIGN_SAMPLE}")
print(f"    Attack flows: {ATTACK_SAMPLE}")
print(f"    Total flows:  {TOTAL_FLOWS}")
print(f"    Shuffled:     Yes (mixed order like real traffic)")

# Assign simulated source IPs
NUM_LEGIT_IPS = 40
NUM_ATTACKER_IPS = 10

legit_ips = [f"192.168.1.{i}" for i in range(1, NUM_LEGIT_IPS + 1)]
attacker_ips = [f"10.0.0.{i}" for i in range(1, NUM_ATTACKER_IPS + 1)]

random.seed(42)

sim_df["Simulated_IP"] = sim_df["Binary_Label"].apply(
    lambda label: random.choice(attacker_ips) if label == 1
    else random.choice(legit_ips)
)

print(f"\n  Simulated IP assignment:")
print(f"    Legitimate IPs (192.168.1.x): {NUM_LEGIT_IPS} IPs for benign flows")
print(f"    Attacker IPs (10.0.0.x):      {NUM_ATTACKER_IPS} IPs for attack flows")
print(f"    Unique IPs in simulation: {sim_df['Simulated_IP'].nunique()}")


# ============================================================================
# STEP 3: Flow Classifier
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Setting Up Flow Classifier")
print("=" * 70)

def classify_flow(flow_features, scaler, model):
    features_2d = np.array(flow_features).reshape(1, -1)
    features_scaled = scaler.transform(features_2d)
    prediction = model.predict(features_scaled)[0]
    return prediction

sample_flow = sim_df[feature_names].iloc[0]
sample_pred = classify_flow(sample_flow, scaler, svm_model)
print(f"  Flow classifier initialized and tested.")
print(f"  Sample flow prediction: {'DDoS' if sample_pred == 1 else 'BENIGN'}")
print(f"  Features per flow: {len(feature_names)}")


# ============================================================================
# STEP 4: Blocklist Manager
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Setting Up Blocklist Manager")
print("=" * 70)

BLOCK_THRESHOLD = 3
WINDOW_SIZE = 10
COOLDOWN_PERIOD = 50

blocked_ips = set()
ip_flag_history = {}
ip_flag_counts = {}
ip_blocked_at = {}
total_ips_unblocked = 0

def check_and_block(ip, flow_index, flagged):
    if ip in blocked_ips:
        return True, False

    if ip not in ip_flag_history:
        ip_flag_history[ip] = []
        ip_flag_counts[ip] = 0

    if flagged:
        ip_flag_history[ip].append(flow_index)
        ip_flag_counts[ip] += 1

        recent_flags = [idx for idx in ip_flag_history[ip]
                        if idx > flow_index - WINDOW_SIZE]

        if len(recent_flags) >= BLOCK_THRESHOLD:
            blocked_ips.add(ip)
            ip_blocked_at[ip] = flow_index
            return True, True

    return False, False

print(f"  Blocklist Manager initialized.")
print(f"  Blocking threshold: {BLOCK_THRESHOLD} flags within {WINDOW_SIZE} flows")
print(f"  Cooldown period:    {COOLDOWN_PERIOD} flows (blocked IPs auto-unblock after this)")


# ============================================================================
# STEP 5: Process All Flows (Classification + Mitigation)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Processing Flows (Classification + Mitigation)")
print("=" * 70)

flow_log = []
first_attack_time = None
first_block_time = None
response_time = None
total_passed = 0
total_blocked_by_list = 0
total_classified = 0
flows_before_first_block = 0
first_block_happened = False

print(f"\n  Processing {TOTAL_FLOWS} flows...\n")

processing_start = time.time()

for idx in range(len(sim_df)):
    row = sim_df.iloc[idx]
    ip = row["Simulated_IP"]
    actual_label = int(row["Binary_Label"])
    actual_label_str = "BENIGN" if actual_label == 0 else "DDoS"
    timestamp = time.time()

    if actual_label == 1 and first_attack_time is None:
        first_attack_time = timestamp

    # Check for timeout — unblock IPs that have served their cooldown
    ips_to_unblock = []
    for blocked_ip, blocked_at_idx in ip_blocked_at.items():
        if blocked_ip in blocked_ips and (idx - blocked_at_idx) >= COOLDOWN_PERIOD:
            ips_to_unblock.append(blocked_ip)

    for unblock_ip in ips_to_unblock:
        blocked_ips.discard(unblock_ip)
        del ip_blocked_at[unblock_ip]
        ip_flag_history[unblock_ip] = []
        ip_flag_counts[unblock_ip] = 0
        total_ips_unblocked += 1
        print(f"    >>> TIMEOUT: IP {unblock_ip} unblocked at flow {idx} "
              f"(cooldown of {COOLDOWN_PERIOD} flows expired)")

    # Check if IP is already blocked
    if ip in blocked_ips:
        total_blocked_by_list += 1
        flow_log.append({
            "Flow_Index": idx,
            "Simulated_IP": ip,
            "Predicted_Label": "DROPPED",
            "Actual_Label": actual_label_str,
            "Action": "BLOCKED",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        })
        if idx % 50 == 0:
            print(f"    Flow {idx:4d} | IP: {ip:15s} | DROPPED (IP blocked)")
        continue

    # Classify the flow using the SVM model
    flow_features = row[feature_names]
    prediction = classify_flow(flow_features, scaler, svm_model)
    predicted_label_str = "DDoS" if prediction == 1 else "BENIGN"
    total_classified += 1

    is_flagged = (prediction == 1)
    is_blocked, newly_blocked = check_and_block(ip, idx, is_flagged)

    if newly_blocked:
        action = "BLOCKED"
        if first_block_time is None:
            first_block_time = timestamp
            first_block_happened = True
    else:
        action = "PASSED"
        total_passed += 1
        if actual_label == 1 and not first_block_happened:
            flows_before_first_block += 1

    flow_log.append({
        "Flow_Index": idx,
        "Simulated_IP": ip,
        "Predicted_Label": predicted_label_str,
        "Actual_Label": actual_label_str,
        "Action": action,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    })

    if idx % 50 == 0:
        print(f"    Flow {idx:4d} | IP: {ip:15s} | Pred: {predicted_label_str:6s} "
              f"| Actual: {actual_label_str:6s} | Action: {action}")

processing_time = time.time() - processing_start

if first_attack_time and first_block_time:
    response_time = first_block_time - first_attack_time
else:
    response_time = None

print(f"\n  Processing complete in {processing_time:.2f} seconds.")
print(f"  Total flows classified by SVM: {total_classified}")
print(f"  Total flows dropped by blocklist: {total_blocked_by_list}")
print(f"  Total unique IPs blocked: {len(blocked_ips)}")
print(f"  Total IPs unblocked after timeout: {total_ips_unblocked}")

log_df = pd.DataFrame(flow_log)
log_df.to_csv(f"{OUTPUT_DIR}/mitigation_log.csv", index=False)
print(f"\n  Saved: {OUTPUT_DIR}/mitigation_log.csv")


# ============================================================================
# STEP 6: Generate Mitigation Report
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Generating Mitigation Report")
print("=" * 70)

classified_log = log_df[log_df["Predicted_Label"] != "DROPPED"].copy()

y_true = classified_log["Actual_Label"].apply(lambda x: 0 if x == "BENIGN" else 1).values
y_pred = classified_log["Predicted_Label"].apply(lambda x: 0 if x == "BENIGN" else 1).values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

total_tp = int(tp)
total_fp = int(fp)
total_tn = int(tn)
total_fn = int(fn)

blocked_attacker_ips = [ip for ip in blocked_ips if ip.startswith("10.0.0.")]
blocked_legit_ips = [ip for ip in blocked_ips if ip.startswith("192.168.1.")]

attack_flows_passed = log_df[
    (log_df["Actual_Label"] == "DDoS") & (log_df["Action"] == "PASSED")
].shape[0]

attack_flows_blocked = log_df[
    (log_df["Actual_Label"] == "DDoS") & (log_df["Action"] == "BLOCKED")
].shape[0]

report_lines = []
report_lines.append("=" * 60)
report_lines.append("  DDoS MITIGATION SYSTEM — REPORT")
report_lines.append("  Project: Enhancing DDoS Detection & Mitigation Model")
report_lines.append("  Author: Ahmed Zain")
report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("=" * 60)

report_lines.append("\n--- SIMULATION CONFIGURATION ---")
report_lines.append(f"  Total flows in simulation:     {TOTAL_FLOWS}")
report_lines.append(f"  Benign flows:                  {BENIGN_SAMPLE}")
report_lines.append(f"  Attack flows:                  {ATTACK_SAMPLE}")
report_lines.append(f"  Simulated legitimate IPs:      {NUM_LEGIT_IPS}")
report_lines.append(f"  Simulated attacker IPs:        {NUM_ATTACKER_IPS}")
report_lines.append(f"  Blocking threshold:            {BLOCK_THRESHOLD} flags in {WINDOW_SIZE} flows")
report_lines.append(f"  Cooldown period:               {COOLDOWN_PERIOD} flows")

report_lines.append("\n--- CLASSIFICATION RESULTS ---")
report_lines.append(f"  Flows classified by SVM:       {total_classified}")
report_lines.append(f"  True Positives (TP):           {total_tp}  — DDoS correctly detected")
report_lines.append(f"  True Negatives (TN):           {total_tn}  — Benign correctly passed")
report_lines.append(f"  False Positives (FP):          {total_fp}  — Benign wrongly flagged as DDoS")
report_lines.append(f"  False Negatives (FN):          {total_fn}  — DDoS missed (classified as Benign)")

report_lines.append("\n--- MITIGATION RESULTS ---")
report_lines.append(f"  Total unique IPs blocked:      {len(blocked_ips)}")
report_lines.append(f"    Attacker IPs blocked:        {len(blocked_attacker_ips)} / {NUM_ATTACKER_IPS}")
report_lines.append(f"    Legitimate IPs blocked:      {len(blocked_legit_ips)} / {NUM_LEGIT_IPS}")
report_lines.append(f"  Flows dropped by blocklist:    {total_blocked_by_list}")
report_lines.append(f"  Attack flows that bypassed:    {attack_flows_passed}  — got through before block")
report_lines.append(f"  Attack flows blocked:          {attack_flows_blocked}")
attack_bypass_rate = (attack_flows_passed / ATTACK_SAMPLE * 100) if ATTACK_SAMPLE > 0 else 0
attack_block_rate = (attack_flows_blocked / ATTACK_SAMPLE * 100) if ATTACK_SAMPLE > 0 else 0
system_effectiveness = ((total_tp + attack_flows_blocked) / ATTACK_SAMPLE * 100) if ATTACK_SAMPLE > 0 else 0
report_lines.append(f"  Attack bypass rate:            {attack_bypass_rate:.2f}%  — bypassed before IP was blocked")
report_lines.append(f"  Attack block rate:             {attack_block_rate:.2f}%  — blocked after IP was identified")
report_lines.append(f"  System effectiveness:          {system_effectiveness:.2f}%  — of total attack flows were stopped")

report_lines.append("\n--- RESPONSE TIME ---")
if response_time is not None:
    report_lines.append(f"  First attack flow arrived:     Yes")
    report_lines.append(f"  First IP blocked:              Yes")
    report_lines.append(f"  Response time:                 {response_time*1000:.2f} ms")
    report_lines.append(f"                                 ({response_time:.6f} seconds)")
else:
    report_lines.append(f"  Response time:                 N/A (no blocks triggered)")

report_lines.append("\n--- TIMEOUT MECHANISM ---")
report_lines.append(f"  Cooldown period:               {COOLDOWN_PERIOD} flows")
report_lines.append(f"  IPs re-evaluated after timeout: {total_ips_unblocked}")
report_lines.append(f"  Behavior: Blocked IPs are automatically unblocked after")
report_lines.append(f"  {COOLDOWN_PERIOD} flows. Flag counter resets to zero upon")
report_lines.append(f"  unblocking. If re-flagged, threshold applies from scratch.")

report_lines.append("\n--- PERFORMANCE METRICS ---")
report_lines.append(f"  Accuracy:                      {accuracy:.4f}  ({accuracy*100:.2f}%)")
report_lines.append(f"  Precision:                     {precision:.4f}  ({precision*100:.2f}%)")
report_lines.append(f"  Recall:                        {recall:.4f}  ({recall*100:.2f}%)")
report_lines.append(f"  F1-Score:                      {f1:.4f}  ({f1*100:.2f}%)")
report_lines.append(f"  False Positive Rate:           {fpr_val:.4f}  ({fpr_val*100:.2f}%)")
report_lines.append(f"  Processing time:               {processing_time:.2f} seconds")

report_lines.append("\n--- BLOCKED IP BREAKDOWN ---")
if blocked_ips:
    for ip in sorted(blocked_ips):
        ip_type = "ATTACKER" if ip.startswith("10.0.0.") else "LEGITIMATE"
        flags = ip_flag_counts.get(ip, 0)
        report_lines.append(f"  {ip:15s}  | Type: {ip_type:10s} | Flags: {flags}")
else:
    report_lines.append("  No IPs were blocked.")

report_lines.append("\n--- VERSION NOTES ---")
report_lines.append(f"  Current version:               2.0 (Dataset-based simulation)")
report_lines.append(f"  PKL files loaded from:         {PKL_DIR}/")
report_lines.append(f"  Outputs saved to:              {OUTPUT_DIR}/")
report_lines.append(f"  Simulation method:             CICDDoS2019 dataset sampling")
report_lines.append(f"  Next version:                  NS-3 generated traffic will replace dataset sampling")

report_lines.append("\n" + "=" * 60)

report_text = "\n".join(report_lines)
print(report_text)

with open(f"{OUTPUT_DIR}/mitigation_report.txt", "w") as f:
    f.write(report_text)
print(f"\n  Saved: {OUTPUT_DIR}/mitigation_report.txt")


# ============================================================================
# STEP 7: Generate Visualizations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Generating Visualizations")
print("=" * 70)

# 7a. Mitigation Timeline
fig, ax = plt.subplots(figsize=(14, 5))

colors = []
y_values = []
for _, row in log_df.iterrows():
    if row["Action"] == "BLOCKED":
        colors.append("#E53935")
        y_values.append(2)
    elif row["Predicted_Label"] == "DDoS":
        colors.append("#FF9800")
        y_values.append(1)
    else:
        colors.append("#4CAF50")
        y_values.append(0)

ax.scatter(log_df["Flow_Index"], y_values, c=colors, s=12, alpha=0.7, edgecolors="none")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["BENIGN\n(Passed)", "DDoS DETECTED\n(Flagged)", "BLOCKED\n(Dropped)"])
ax.set_xlabel("Flow Index (Simulation Order)", fontsize=12)
ax.set_title("Mitigation Timeline — Flow Processing Over Time", fontsize=14)
ax.grid(axis="x", alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", markersize=8, label="Benign (Passed)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF9800", markersize=8, label="DDoS Detected (Flagged)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#E53935", markersize=8, label="Blocked (Dropped)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mitigation_timeline.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/mitigation_timeline.png")

# 7b. IP Activity Chart
all_ips_in_sim = sim_df["Simulated_IP"].unique()
ip_data = []
for ip in sorted(all_ips_in_sim):
    flags = ip_flag_counts.get(ip, 0)
    ip_type = "Attacker" if ip.startswith("10.0.0.") else "Legitimate"
    ip_data.append({"IP": ip, "Flags": flags, "Type": ip_type})

ip_chart_df = pd.DataFrame(ip_data)
ip_chart_df = ip_chart_df[ip_chart_df["Flags"] > 0].sort_values("Flags", ascending=False)

if len(ip_chart_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_colors = ["#E53935" if t == "Attacker" else "#2196F3" for t in ip_chart_df["Type"]]
    bars = ax.bar(range(len(ip_chart_df)), ip_chart_df["Flags"], color=bar_colors, edgecolor="white")
    ax.axhline(y=BLOCK_THRESHOLD, color="#333333", linestyle="--", linewidth=2,
               label=f"Blocking Threshold ({BLOCK_THRESHOLD} flags)")
    ax.set_xticks(range(len(ip_chart_df)))
    ax.set_xticklabels(ip_chart_df["IP"], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Simulated Source IP", fontsize=12)
    ax.set_ylabel("Number of Flagged Flows", fontsize=12)
    ax.set_title("IP Activity — Flagged Flows per Source IP", fontsize=14)
    legend_elements = [
        Line2D([0], [0], color="#E53935", lw=8, label="Attacker IPs"),
        Line2D([0], [0], color="#2196F3", lw=8, label="Legitimate IPs"),
        Line2D([0], [0], color="#333333", linestyle="--", lw=2, label=f"Block Threshold ({BLOCK_THRESHOLD})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ip_activity_chart.png", dpi=150)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/ip_activity_chart.png")
else:
    print("  Skipped: ip_activity_chart.png (no flagged IPs)")

# 7c. Mitigation Metrics Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metrics_values = [accuracy, precision, recall, f1]

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
bars = ax.bar(metrics_names, metrics_values, color=bar_colors, edgecolor="white")
for bar, val in zip(bars, metrics_values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.set_title("Mitigation System — Performance Metrics", fontsize=14)
ax.set_ylabel("Score", fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mitigation_metrics_chart.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/mitigation_metrics_chart.png")

# 7d. Mitigation Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=["BENIGN", "DDoS"],
    yticklabels=["BENIGN", "DDoS"]
)
plt.title("Confusion Matrix — Mitigation System (SVM + Blocklist)", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mitigation_confusion_matrix.png", dpi=150)
plt.close()
print(f"  Saved: {OUTPUT_DIR}/mitigation_confusion_matrix.png")

print("\n" + "=" * 70)
print("ALL DONE! Mitigation model complete.")
print(f"Check the '{OUTPUT_DIR}' folder for:")
print("  - mitigation_log.csv")
print("  - mitigation_report.txt")
print("  - mitigation_timeline.png")
print("  - ip_activity_chart.png")
print("  - mitigation_metrics_chart.png")
print("  - mitigation_confusion_matrix.png")
print("=" * 70)
print("\nNOTE: This is Version 2.0.")
print(f"      PKL files loaded from: {PKL_DIR}/")
print(f"      Outputs saved to:      {OUTPUT_DIR}/")