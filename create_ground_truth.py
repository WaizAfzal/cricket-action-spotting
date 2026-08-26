import pandas as pd
import os

# 1. Define the video details
name_of_video_file = "test_over_1.mp4"
video_frames_per_second = 30  # Standard YouTube broadcast frame rate

# 2. WATCH THE VIDEO: Plug in the start and end SECONDS for all 6 balls.
# Example: If he starts running at 1 minute and 5 seconds, write 65.
delivery_timestamps = [
    {"ball_number": 1, "runup_start_second": 3, "ball_dead_second": 10},
    {"ball_number": 2, "runup_start_second": 35, "ball_dead_second": 42},  # Example timings
    {"ball_number": 3, "runup_start_second": 68, "ball_dead_second": 75},
    {"ball_number": 4, "runup_start_second": 98, "ball_dead_second": 106},
    {"ball_number": 5, "runup_start_second": 132, "ball_dead_second": 140},
    {"ball_number": 6, "runup_start_second": 165, "ball_dead_second": 174},
]

# 3. We will create a clean list to store our final formatted data
formatted_dataset_rows = []

for delivery in delivery_timestamps:
    
    # Calculate the exact frame numbers by multiplying seconds by the frame rate
    start_frame_number = delivery["runup_start_second"] * video_frames_per_second
    end_frame_number = delivery["ball_dead_second"] * video_frames_per_second
    
    # Create a dictionary for this specific row of data
    current_row_data = {
        "video_name": name_of_video_file,
        "delivery_id": delivery["ball_number"],
        "start_frame": start_frame_number,
        "end_frame": end_frame_number
    }
    
    formatted_dataset_rows.append(current_row_data)

# 4. Convert our list into a Pandas DataFrame
ground_truth_dataframe = pd.DataFrame(formatted_dataset_rows)

# 5. Save it safely to your test folder
save_directory = r"C:\Users\mwab2\cricket_action_spotting\data\my_test_set\test_ground_truth.csv"
ground_truth_dataframe.to_csv(save_directory, index=False)

print(f"Success! Your ground truth CSV has been saved to: {save_directory}")