import pandas as pd
import numpy as np

# Load detected actions
detected_df = pd.read_csv("data/my_test_set/detected_actions.csv")

# Actual ground-truth bat impact timestamps observed in each clip
actual_ground_truth_seconds = [
    5.10,  # Delivery 1: Bat impact timestamp
    0.65,  # Delivery 2: Bat impact timestamp
    3.30,  # Delivery 3: Bat impact timestamp
    6.40   # Delivery 4: Bat impact timestamp
]

tolerance_seconds = 0.50  # Standard temporal action spotting tolerance window (+/- 0.5s)
results = []

for index in range(min(4, len(detected_df))):
    det_row = detected_df.iloc[index]
    delivery_name = f"Delivery {index + 1}"
    detected_time = float(det_row["impact_timestamp_seconds"])
    true_time = actual_ground_truth_seconds[index]
    
    error_delta = abs(detected_time - true_time)
    status = "PASS (Hit)" if error_delta <= tolerance_seconds else "FAIL (Miss)"

    results.append({
        "Delivery": delivery_name,
        "Detected Time (s)": detected_time,
        "Ground Truth (s)": true_time,
        "Error Delta (s)": round(error_delta, 2),
        "Status (<=0.5s)": status
    })

eval_df = pd.DataFrame(results)

print("\n" + "=" * 68)
print(f"{'ACTION SPOTTING TEMPORAL EVALUATION METRICS':^68}")
print("=" * 68)
print(eval_df.to_string(index=False))
print("-" * 68)

mean_error = round(eval_df["Error Delta (s)"].mean(), 3)
accuracy = (eval_df["Status (<=0.5s)"].str.contains("PASS").sum() / len(eval_df)) * 100

print(f"Mean Absolute Temporal Error (MAE): {mean_error} seconds")
print(f"Temporal Spotting Accuracy Rate:    {accuracy:.1f}%")
print("=" * 68 + "\n")

# Save final evaluation table
output_path = "data/my_test_set/final_evaluation_metrics.csv"
eval_df.to_csv(output_path, index=False)
print(f"Saved final metrics table to: {output_path}")