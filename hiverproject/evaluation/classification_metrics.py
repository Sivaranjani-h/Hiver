import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

INPUT_FILE = "classifier_predictions.csv"
OUTPUT_FILE = "classification_metrics.csv"

print("Loading classifier predictions...")
df = pd.read_csv(INPUT_FILE)

# Remove invalid predictions if any
df = df[
    df["predicted_label"].notna() &
    ~df["predicted_label"].isin(["", "ERROR", "EMPTY_RESPONSE"])
].copy()

y_true = df["true_label"]
y_pred = df["predicted_label"]

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n====================================")
print("CLASSIFICATION EVALUATION")
print("====================================")
print(f"Accuracy: {accuracy:.2%} ({(y_true == y_pred).sum()}/{len(df)})")

# Precision, recall, F1, and support for each intent
report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

# Save the complete report
report_df.to_csv(OUTPUT_FILE)

print("\nPrecision / Recall / F1 report:")
print(report_df.round(3))

print(f"\nSaved metrics to {OUTPUT_FILE}")