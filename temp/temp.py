import cv2, sys

VIDEO_PATH = "data/OK/clip_1762493991.mp4"
OUTPUT_PATH = "config/sample_frame.jpg"

# >>> CHANGE THIS VARIABLE <<<
TARGET_TIME_SEC = 5   # e.g., 2.5 seconds into the video

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: cannot open {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = int(TARGET_TIME_SEC * fps)

# Seek to frame
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ok, frame = cap.read()
if not ok:
    print("Error: could not read frame at the specified timestamp.")
    cap.release()
    sys.exit(1)

cv2.imwrite(OUTPUT_PATH, frame)
cap.release()
print(f"Wrote {OUTPUT_PATH} (timestamp: {TARGET_TIME_SEC}s, frame {frame_index})")