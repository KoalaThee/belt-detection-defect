import serial, time, threading

PORT = "COM5"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2)

print(f"Connected to {PORT}\n")

def read_from_pico():
    while True:
        if ser.in_waiting:
            print("<<", ser.readline().decode(errors="ignore").strip())

threading.Thread(target=read_from_pico, daemon=True).start()

print("Sending test messages every 5 seconds. Ctrl+C to exit.\n")

test_messages = ["TEST", "OK", "DEFECT"]
message_index = 0

try:
    while True:
        msg = test_messages[message_index % len(test_messages)]
        print(f">> {msg}")
        ser.write((msg + "\n").encode())
        message_index += 1
        time.sleep(5)
except KeyboardInterrupt:
    ser.close()
    print("\nClosed serial connection.")
