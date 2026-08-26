import os
import glob
import cv2
import numpy as np
import pandas as pd


def detect_bat_impact_peak(delivery_clip_path):
    clip_capture = cv2.VideoCapture(delivery_clip_path)
    previous_frame_grayscale = None
    motion_energy_scores = []

    frame_index = 0
    while True:
        success, current_frame = clip_capture.read()
        if not success:
            break

        # Convert frame to grayscale
        gray_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape

        # Focus motion-differencing specifically on pitch & batsman region
        pitch_region = gray_image[
            int(height * 0.30):int(height * 0.85),
            int(width * 0.30):int(width * 0.70)
        ]

        if previous_frame_grayscale is not None:
            # Absolute pixel difference calculates motion velocity
            pixel_difference = cv2.absdiff(pitch_region, previous_frame_grayscale)
            motion_energy = np.sum(pixel_difference)
            motion_energy_scores.append(motion_energy)
        else:
            motion_energy_scores.append(0)

        previous_frame_grayscale = pitch_region
        frame_index += 1

    clip_capture.release()

    if len(motion_energy_scores) == 0:
        return 0, 0.0

    # The frame with maximum pixel change corresponds to the bat swing / impact
    peak_local_frame = int(np.argmax(motion_energy_scores))
    peak_timestamp_seconds = round(peak_local_frame / 30.0, 2)

    return peak_local_frame, peak_timestamp_seconds


def main():
    deliveries_directory = "data/fps_frames/deliveries"
    delivery_video_files = sorted(glob.glob(os.path.join(deliveries_directory, "delivery_*.mp4")))

    if not delivery_video_files:
        print(f"No delivery clips found in {deliveries_directory}!")
        return

    detection_results = []

    print(f"\n{'='*55}")
    print(f"{'Delivery':<15}{'Impact Frame':<18}{'Clip Time (s)':<15}")
    print(f"{'='*55}")

    for video_path in delivery_video_files:
        clip_name = os.path.basename(video_path)
        peak_frame, peak_second = detect_bat_impact_peak(video_path)

        detection_results.append({
            "delivery_clip": clip_name,
            "detected_impact_frame": peak_frame,
            "impact_timestamp_seconds": peak_second
        })

        print(f"{clip_name:<15}{peak_frame:<18}{peak_second:<15}")

    print(f"{'='*55}\n")

    # Save results to CSV
    output_dataframe = pd.DataFrame(detection_results)
    output_csv_path = "data/my_test_set/detected_actions.csv"
    output_dataframe.to_csv(output_csv_path, index=False)
    print(f"Action spotting completed! Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()