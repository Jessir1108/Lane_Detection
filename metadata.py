from pymediainfo import MediaInfo
from datetime import datetime, timedelta
import pytz
import csv

def extract_metadata(file_path, utc_offset):
    media_info = MediaInfo.parse(file_path)
    frame_times = []

    for track in media_info.tracks:
        if track.track_type == "Video":
            duration = track.duration / 1000  # Convert to seconds
            frame_rate = float(track.frame_rate)
            total_frames = int(duration * frame_rate)
            start_time = datetime.strptime(track.encoded_date, '%Y-%m-%d %H:%M:%S %Z')
            start_time = start_time + timedelta(hours=utc_offset)

            for frame_number in range(total_frames):
                frame_time = start_time + timedelta(seconds=frame_number / frame_rate)
                frame_times.append((frame_number, frame_time))

    return frame_times

def save_to_csv(frame_times, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame Number", "Timestamp"])
        for frame_number, timestamp in frame_times:
            writer.writerow([frame_number, timestamp])

if __name__ == "__main__":
    file_path = r"c:\Users\danie\Documents\capstone_project\plate_detection\videos\Raw\2.MP4"
    utc_offset = -5
    output_csv = "frame_times.csv"
    
    frame_times = extract_metadata(file_path, utc_offset)
    save_to_csv(frame_times, output_csv)