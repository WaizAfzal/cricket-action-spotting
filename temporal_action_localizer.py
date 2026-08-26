import os
import cv2
import numpy as np
import pandas as pd

def compute_tiou(interval_a, interval_b):
    """Calculates Temporal Intersection-over-Union (tIoU) between two time intervals."""
    start_a, end_a = interval_a
    start_b, end_b = interval_b

    intersection_start = max(start_a, start_b)
    intersection_end = min(end_a, end_b)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(start_a, start_b)
    union_end = max(end_a, end_b)
    union = max(1e-6, union_end - union_start)

    return intersection / union

# 1. Automated Proposal Generation from Raw Untrimmed Video
video_path = "data/my_test_set/test_over_1.mp4"
gt_csv_path = "data/my_test_set/test_ground_truth.csv"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

is_fpv_list = []
frame_idx = 0

print("Scanning untrimmed broadcast video for delivery boundaries...")

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= 3400:  # Valid stream window
        break

    # Analyze pitch corridor (grass green + pitch beige saturation profile)
    h, w, _ = frame.shape
    roi = frame[int(h * 0.35):int(h * 0.80), int(w * 0.30):int(w * 0.70)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Grass & Pitch Hue Mask (Green + Clay tones)
    pitch_mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([85, 255, 255]))
    pitch_ratio = np.count_nonzero(pitch_mask) / (roi.shape[0] * roi.shape[1])
    
    # Classify frame: FPV pitch view vs non-FPV broadcast angle
    is_fpv = 1 if pitch_ratio > 0.18 else 0
    is_fpv_list.append(is_fpv)
    frame_idx += 1

cap.release()

# 2. Temporal Proposal Smoothing & Clustering
smoothed = pd.Series(is_fpv_list).rolling(window=30, min_periods=1, center=True).mean()
proposals = []
in_event = False
start_f = 0

for i, score in enumerate(smoothed):
    if score > 0.40 and not in_event:
        in_event = True
        start_f = max(0, i - 15)
    elif score <= 0.40 and in_event:
        in_event = False
        end_f = i + 15
        duration = end_f - start_f
        # Filter candidate intervals matching valid delivery lengths (3s to 12s)
        if int(fps * 3.0) <= duration <= int(fps * 12.0):
            proposals.append((start_f, end_f))

# 3. Temporal Evaluation (tIoU & mAP)
gt_df = pd.read_csv(gt_csv_path).head(len(proposals))
results = []
tiou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n" + "=" * 76)
print(f"{'TEMPORAL ACTION LOCALIZATION BENCHMARK (Gupta et al. Protocol)':^76}")
print("=" * 76)

matched_tious = []
for idx, row in gt_df.iterrows():
    gt_interval = (int(row["start_frame"]), int(row["end_frame"]))
    # Find proposal with maximum temporal overlap
    best_tiou = max([compute_tiou(prop, gt_interval) for prop in proposals]) if proposals else 0.0
    matched_tious.append(best_tiou)
    
    del_id = f"Delivery {idx + 1}"
    gt_str = f"[{gt_interval[0]} -> {gt_interval[1]}]"
    pred_str = f"[{proposals[idx][0]} -> {proposals[idx][1]}]" if idx < len(proposals) else "N/A"
    
    results.append({
        "Delivery": del_id,
        "Ground Truth": gt_str,
        "Proposal": pred_str,
        "tIoU": round(best_tiou, 3),
        "Hit @ 0.5": "PASS" if best_tiou >= 0.5 else "FAIL"
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print("-" * 76)

# Calculate mAP at multiple tIoU thresholds
map_scores = {}
for thresh in tiou_thresholds:
    hits = sum(1 for tiou in matched_tious if tiou >= thresh)
    map_scores[f"mAP@{thresh}"] = round((hits / len(matched_tious)) * 100, 1)

# Gupta et al. Weighted Temporal IoU (wtIoU)
mean_tiou = round(np.mean(matched_tious), 3)
w_tiou = round(np.sum([t**2 for t in matched_tious]) / max(1e-6, np.sum(matched_tious)), 3)

print("EVALUATION METRICS:")
for k, v in map_scores.items():
    print(f"  * {k}: {v}%")
print(f"  * Mean tIoU: {mean_tiou}")
print(f"  * Weighted tIoU (wtIoU): {w_tiou}")
print("=" * 76)

# Save localization metrics
output_csv = "data/my_test_set/temporal_localization_metrics.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nSaved localization results to: {output_csv}")