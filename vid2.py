# save as capture_dataset.py
import cv2, os, time, pathlib

CAM_INDEX = 1
WIDTH, HEIGHT = 1280, 720
FPS = 30
CLASS_NAMES = ["OK", "Defect", "Defect_Backup", "OK_Backup"]
OUT_DIR = pathlib.Path("data"); OUT_DIR.mkdir(exist_ok=True)

# Default camera settings
exposure_val = -10
brightness_val = 90
TARGET_WB_TEMP = 4500

def _set_and_confirm(cap, prop, value, retries=3, delay=0.1):
    """Set a camera property and retry a few times until it sticks."""
    for _ in range(retries):
        cap.set(prop, value)
        time.sleep(delay)
        got = cap.get(prop)
        if abs(float(got) - float(value)) < 1.5:
            return True
    return False

def set_cam_props(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    # Manual mode
    _set_and_confirm(cap, cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    _set_and_confirm(cap, cv2.CAP_PROP_AUTO_WB, 0)
    _set_and_confirm(cap, cv2.CAP_PROP_WB_TEMPERATURE, TARGET_WB_TEMP)

def apply_current_settings(cap, exp, bright):
    cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)

def ensure_dirs():
    for cname in CLASS_NAMES:
        (OUT_DIR / cname).mkdir(parents=True, exist_ok=True)

def main():
    global exposure_val, brightness_val

    ensure_dirs()
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    time.sleep(0.2)
    set_cam_props(cap)
    apply_current_settings(cap, exposure_val, brightness_val)

    counters = {c: len(list((OUT_DIR/c).glob("*.jpg"))) for c in CLASS_NAMES}

    info = ("Keys: [1..n]=class | [SPACE]=capture | [v]=video | "
            "[w/s]=exposure +/- | [a/d]=brightness -/+ | [q]=quit")
    last_class = CLASS_NAMES[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Overlay info and live settings
        ui = frame.copy()
        txt = (f"Class: {last_class} | Counts: {counters} | "
               f"Exp={exposure_val} | Bright={brightness_val} | {info}")
        cv2.rectangle(ui, (0,0), (ui.shape[1], 35), (0,0,0), -1)
        cv2.putText(ui, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        cv2.imshow("Dataset Capture", ui)
        key = cv2.waitKey(1) & 0xFF

        # ---- CLASS SELECTION ----
        if ord('1') <= key <= ord(str(min(9, len(CLASS_NAMES)))):
            last_class = CLASS_NAMES[key - ord('1')]

        # ---- CAPTURE IMAGE ----
        elif key == ord(' '):
            cname = last_class
            counters[cname] += 1
            outfile = OUT_DIR / cname / f"img_{counters[cname]:05d}.jpg"
            cv2.imwrite(str(outfile), frame)
            print("Saved", outfile)

        # ---- TOGGLE VIDEO RECORD ----
        elif key == ord('v'):
            if writer is None:
                clip_dir = OUT_DIR / last_class
                clip_dir.mkdir(exist_ok=True)
                ts = int(time.time())
                video_path = str(clip_dir / f"clip_{ts}.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, FPS,
                                         (frame.shape[1], frame.shape[0]))
                print("Recording to", video_path)
            else:
                writer.release(); writer = None
                print("Stopped recording")

        # ---- LIVE EXPOSURE / BRIGHTNESS ADJUST ----
        elif key == ord('w'):
            exposure_val += 1
            apply_current_settings(cap, exposure_val, brightness_val)
            print(f"Exposure -> {exposure_val}")
        elif key == ord('s'):
            exposure_val -= 1
            apply_current_settings(cap, exposure_val, brightness_val)
            print(f"Exposure -> {exposure_val}")
        elif key == ord('a'):
            brightness_val -= 10
            apply_current_settings(cap, exposure_val, brightness_val)
            print(f"Brightness -> {brightness_val}")
        elif key == ord('d'):
            brightness_val += 10
            apply_current_settings(cap, exposure_val, brightness_val)
            print(f"Brightness -> {brightness_val}")

        # ---- QUIT ----
        elif key == ord('q'):
            break

        # ---- RECORD VIDEO FRAME ----
        if writer is not None:
            writer.write(frame)

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
