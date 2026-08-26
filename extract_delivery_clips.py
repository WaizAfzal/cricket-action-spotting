import os
import cv2
import pandas as pd

video_path = "data/my_test_set/test_over_1.mp4"
csv_path = "data/my_test_set/test_ground_truth.csv"
output_dir = "data/fps_frames/deliveries"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

deliveries = []
for _, row in df.iterrows():
    delivery_id = int(row["delivery_id"])
    output_clip_path = os.path.join(output_dir, f"delivery_{delivery_id}.mp4")
    deliveries.append({
        "id": delivery_id,
        "start": int(row["start_frame"]),
        "end": int(row["end_frame"]),
        "frames_written": 0,
        "writer": cv2.VideoWriter(
            output_clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
    })

print("Extracting delivery clips with fault tolerance...")
current_frame = 0
consecutive_failures = 0
max_frame_target = int(df["end_frame"].max())

while current_frame <= max_frame_target + 30:
    ret, frame = cap.read()

    if not ret:
        consecutive_failures += 1
        current_frame += 1
        # Stop only after 150 consecutive dead frames (5 seconds of dead data)
        if consecutive_failures > 150:
            print(f"Reached end of readable stream at frame {current_frame}.")
            break
        continue

    consecutive_failures = 0

    for d in deliveries:
        if d["start"] <= current_frame <= d["end"]:
            d["writer"].write(frame)
            d["frames_written"] += 1

    current_frame += 1

for d in deliveries:
    d["writer"].release()
    print(f"delivery_{d['id']}.mp4 -> Total frames captured: {d['frames_written']}")

cap.release()
print("Extraction finished.")