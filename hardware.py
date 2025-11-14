import serial
import sys
import os
import threading
import time

SERIAL_PORT = "COM5"
BAUD_RATE = 115200
TIMEOUT = 0.1

# Monitoring state
_monitoring_active = False

# Check if we should connect to serial port
_run_main = os.environ.get('WERKZEUG_RUN_MAIN')

# Check if Flask reloader is active
_is_running_flask = len(sys.argv) > 0 and 'app_flask' in sys.argv[0]
_is_flask_reloader = _is_running_flask and _run_main is None

_should_connect = _run_main == 'true' or (_run_main is None and not _is_flask_reloader)

# Try to connect to Pico via serial port
if _should_connect:
    try:
        pico = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        time.sleep(2)  # Wait for connection to stabilize (like monitor.py)
        print(f"[HARDWARE] Connected to Pico on {SERIAL_PORT} at {BAUD_RATE} baud")
        
        # Start monitoring thread to read Pico output
        _monitoring_active = True
        
        def read_from_pico():
            while _monitoring_active and pico:
                try:
                    if pico.in_waiting:
                        line = pico.readline().decode(errors="ignore").strip()
                        if line:
                            print(f"[PICO] {line}")
                except Exception as e:
                    if _monitoring_active:
                        pass
                    break
        
        _monitor_thread = threading.Thread(target=read_from_pico, daemon=True)
        _monitor_thread.start()
        print("[HARDWARE] Pico monitoring started (output will appear as [PICO] messages)")
        
    except Exception as e:
        pico = None
        _monitoring_active = False
        print(f"[HARDWARE] No hardware connected (simulation mode): {e}")
else:
    pico = None
    _monitoring_active = False
    print(f"[HARDWARE] Skipping connection in Flask reloader parent process (WERKZEUG_RUN_MAIN={_run_main})")

def send_command(msg: str):
    if pico:
        try:
            pico.write((msg + "\n").encode())
            print(f"[HARDWARE] Sent to Pico: {msg}")
        except Exception as e:
            print(f"[HARDWARE] Error sending command '{msg}': {e}")
    else:
        print(f"[SIM] Sent to Pico: {msg}")

def close():
    global pico, _monitoring_active
    _monitoring_active = False
    if pico and pico.is_open:
        pico.close()
        pico = None
        print("[HARDWARE] Serial connection closed")

# Cleanup on module exit
import atexit
atexit.register(close)

