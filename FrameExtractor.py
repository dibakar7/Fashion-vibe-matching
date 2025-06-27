import cv2
import os

def FrameExtractor(video_path, output_folder, every_n_frames=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % every_n_frames == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        frame_count += 1

    cap.release()
    return saved
