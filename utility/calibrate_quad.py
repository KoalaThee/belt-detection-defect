import cv2, json, sys
from pathlib import Path

IMG_PATH = "config/sample_frame.jpg"
out_path = Path("config"); out_path.mkdir(exist_ok=True)
save_to = out_path / "quad.json"

pts = []
def on_mouse(event, x, y, flags, param):
    global pts, img_show
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([int(x), int(y)])
        cv2.circle(img_show, (x,y), 5, (0,255,0), -1)
        cv2.imshow("click 4 corners (TL,TR,BR,BL)", img_show)

img = cv2.imread(IMG_PATH)
img_show = img.copy()
cv2.imshow("click 4 corners (TL,TR,BR,BL)", img_show)
cv2.setMouseCallback("click 4 corners (TL,TR,BR,BL)", on_mouse)
while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
    if len(pts) == 4:
        with open(save_to, "w") as f:
            json.dump({"points": pts}, f)
        print("Saved:", save_to)
        break
cv2.destroyAllWindows()
