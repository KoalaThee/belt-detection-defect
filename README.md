# **Pill Counting & Defect Detection System**

A computer-vision pill counting and defect detection pipeline for conveyor-belt inspection.  
The system performs HSV-based pill detection, temporal tracking, hardware actuation via Raspberry Pi Pico, and optional web dashboard visualization.


# **Information**

This system improves accuracy in high-speed pill inspection using **color segmentation**, **temporal analysis**, and **hardware integration**. Compared to traditional systems that rely only on thresholding or single-frame detection, this project focuses on **robust real-time tracking**, synchronized with **physical actuation** for defect removal.

## **Why This System?**

### **Unique Features**
1. **Accurate Color-Based Detection**  
   HSV segmentation tuned for bright yellow pills.
2. **Temporal Tracking for Stability**  
   Counts pills over multiple frames to reduce noise and flicker.
3. **Hardware Integration**  
   Sends `OK`/`DEFECT` signals to a Raspberry Pi Pico for real-time actuation.
4. **Web Dashboard**  
   Live updates of detection, results, counts, and system status.
5. **Calibration Tools & Parameter Optimizers**  
   Includes quad calibration, grid search, and Bayesian optimization.

### **What Other Systems Miss**
- Single-frame detection produces unstable counts  
- No hardware-actuated removal workflow  
- Limited support for parameter optimization  
- No synchronized real-time web interface  

<br>

# **Project Structure**

```bash
capture_vid/
├── app_flask.py                      # Flask app: detection thread + dashboard
├── app_state.py                      # Thread-safe shared state management
├── config_flask.py                   # Flask configuration (video source, port)
├── count_pills_color.py              # Main HSV + temporal detection script
├── hardware.py                       # Serial communication with Raspberry Pi Pico
├── requirements.txt                  # Dependencies
├── simulation_vid.mp4                # Sample test video
│
├── config/
│   ├── quad.json                     # Perspective calibration (warp points)
│   └── sample_frame.jpg              # Reference frame for quad calibration
│
├── templates/
│   └── dashboard.html                # Web dashboard UI
│
├── utility/
│   ├── calibrate_quad.py             # Click-based quad calibration
│   ├── vid2.py                       # Webcam video capture tool
│   ├── optimize_temporal_grid.py     # Grid search optimizer
│   ├── optimize_temporal_bayesian.py # Bayesian optimization script
│   ├── optimized_params_color.json   # Saved optimized parameters
│   ├── monitor.py                    # Pico serial tester (TEST/OK/DEFECT)
│   ├── count_pills_simple.py         # Base detection functions
│   └── eval_video_temporal_color.py  # Evaluate detection on individual video
│
└── data/
    ├── OK/
    │   └── clip_.mp4
    └── Defect/
        └── clip_.mp4
