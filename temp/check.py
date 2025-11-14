import cv2

cap = cv2.VideoCapture(1)  # change index if needed
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

exp, bright = -6, 120
cap.set(cv2.CAP_PROP_EXPOSURE, exp)
cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)

print("Use keys: [w/s]=exposure up/down, [a/d]=brightness down/up, [q]=quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Tune Exposure/Brightness", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('w'):
        exp += 1; cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    elif key == ord('s'):
        exp -= 1; cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    elif key == ord('a'):
        bright -= 10; cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
    elif key == ord('d'):
        bright += 10; cap.set(cv2.CAP_PROP_BRIGHTNESS, bright)
    print(f"Exposure={exp}, Brightness={bright}")

cap.release()
cv2.destroyAllWindows()
