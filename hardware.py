# hardware.py
"""
Hardware abstraction layer for communicating with actuators (Pico, MQTT, etc.).
Handles all physical-device communication, allowing the backend to work without hardware.
"""
import serial
import sys

# Configuration - can be moved to config.py later
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
TIMEOUT = 1

# Try to connect to Pico via serial port
try:
    pico = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
    print(f"[HARDWARE] Connected to Pico on {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    pico = None
    print(f"[HARDWARE] No hardware connected (simulation mode): {e}")

def send_command(msg: str):
    """
    Send a command string to the hardware actuator.
    
    Args:
        msg: Command string (e.g., "OK", "DEFECT")
    
    Example:
        send_command("OK")      # Sends "OK\\n" to Pico
        send_command("DEFECT")  # Sends "DEFECT\\n" to Pico
    """
    if pico:
        try:
            pico.write((msg + "\n").encode())
            pico.flush()  # Ensure data is sent immediately
            print(f"[HARDWARE] Sent to Pico: {msg}")
        except Exception as e:
            print(f"[HARDWARE] Error sending command '{msg}': {e}")
    else:
        print(f"[SIM] Sent to Pico: {msg}")

def close():
    """Close the serial connection if open."""
    global pico
    if pico and pico.is_open:
        pico.close()
        pico = None
        print("[HARDWARE] Serial connection closed")

# Cleanup on module exit
import atexit
atexit.register(close)

