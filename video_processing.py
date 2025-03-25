import whisper
import cv2
import os

# Load Whisper model
model = whisper.load_model("base")  # Use "small" or "large" for better accuracy

def extract_audio_from_video(video_path):
    """Extracts text from video audio using Whisper ASR"""
    try:
        result = model.transcribe(video_path)
        return result["text"]
    except Exception as e:
        return f"Error processing video: {e}"


def extract_frames(video_path, output_folder="data/frames", frame_interval=30):
    """Extracts frames from video at a given interval"""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        count += 1
    cap.release()
    return f"Extracted {frame_count} frames from video."

# Example usage
if __name__ == "__main__":
    video_text = extract_audio_from_video("data/sample_video.mp4")
    print("Extracted Video Text:\n", video_text)
    extract_frames("data/sample_video.mp4")
