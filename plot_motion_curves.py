import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

deliveries_dir = "data/fps_frames/deliveries"
delivery_files = [f"delivery_{i}.mp4" for i in range(1, 5)]

plt.figure(figsize=(14, 8))

for idx, file_name in enumerate(delivery_files, 1):
    video_path = os.path.join(deliveries_dir, file_name)
    cap = cv2.VideoCapture(video_path)
    
    motion_scores = []
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        pitch = gray[int(h * 0.30):int(h * 0.85), int(w * 0.30):int(w * 0.70)]
        
        if prev_gray is not None:
            diff = cv2.absdiff(pitch, prev_gray)
            motion_scores.append(np.sum(diff))
        else:
            motion_scores.append(0)
        prev_gray = pitch
        
    cap.release()
    
    peak_frame = int(np.argmax(motion_scores))
    time_axis = np.arange(len(motion_scores)) / 30.0
    
    plt.subplot(2, 2, idx)
    plt.plot(time_axis, motion_scores, color="#1f77b4", linewidth=2.0, label="Motion Energy")
    plt.axvline(
        x=peak_frame / 30.0, 
        color="red", 
        linestyle="--", 
        linewidth=1.8, 
        label=f"Detected Impact ({peak_frame / 30.0:.2f}s)"
    )
    plt.title(f"Delivery {idx} - Pitch Motion Energy Curve", fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temporal Pixel Difference")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

plt.tight_layout()
output_chart = "data/my_test_set/motion_energy_curves.png"
plt.savefig(output_chart, dpi=300)
print(f"Motion energy visualization saved to: {output_chart}")
plt.show()