# save as capture_dataset.py
import cv2, os, time, pathlib

CAM_INDEX = 1         # change if multiple cams
WIDTH, HEIGHT = 1280, 720
FPS = 30
CLASS_NAMES = ["OK", "Defect"]   # edit for your classes
OUT_DIR = pathlib.Path("data"); OUT_DIR.mkdir(exist_ok=True)

def set_cam_props(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    # Try to lock exposure/white balance to avoid flicker (may not work on all webcams)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25=manual on many drivers
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # tweak per camera
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)

def ensure_dirs():
    for cname in CLASS_NAMES:
        (OUT_DIR / cname).mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    set_cam_props(cap)
    counters = {c: len(list((OUT_DIR/c).glob("*.jpg"))) for c in CLASS_NAMES}

    info = "Keys: [1..n]=class | [SPACE]=capture last class | [v]=save video | [q]=quit"
    last_class = CLASS_NAMES[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    while True:
        ok, frame = cap.read()
        if not ok: break

        # overlay UI
        ui = frame.copy()
        txt = f"Class: {last_class} | Counts: {counters} | {info}"
        cv2.rectangle(ui, (0,0), (ui.shape[1], 30), (0,0,0), -1)
        cv2.putText(ui, txt, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("Dataset Capture", ui)
        key = cv2.waitKey(1) & 0xFF

        # number keys select class
        if ord('1') <= key <= ord(str(min(9, len(CLASS_NAMES)))):
            last_class = CLASS_NAMES[key - ord('1')]

        # space: capture one frame to last_class
        elif key == ord(' '):
            cname = last_class
            counters[cname] += 1
            outfile = OUT_DIR / cname / f"img_{counters[cname]:05d}.jpg"
            cv2.imwrite(str(outfile), frame)
            print("Saved", outfile)

        # 'v': toggle video recording of last_class (clips stored under that class)
        elif key == ord('v'):
            if writer is None:
                clip_dir = OUT_DIR / last_class
                clip_dir.mkdir(exist_ok=True)
                ts = int(time.time())
                video_path = str(clip_dir / f"clip_{ts}.mp4")
                writer = cv2.VideoWriter(video_path, fourcc, FPS, (frame.shape[1], frame.shape[0]))
                print("Recording to", video_path)
            else:
                writer.release(); writer = None
                print("Stopped recording")

        elif key == ord('q'):
            break

        if writer is not None:
            writer.write(frame)

    if writer is not None: writer.release()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
