# **Pill Counting & Defect Detection System**

A **computer-vision pill counting** and **defect detection pipeline** for pill on conveyor-belt connected to camera, Raspberry Pi Pico and pneumatic actuator.  
The system performs HSV-based pill detection, temporal tracking, hardware actuation via Raspberry Pi Pico, and optional web dashboard visualization.

<p align="center">
  <img src="https://github.com/KoalaThee/belt-detection-defect/blob/a404f017c6358e981516cc83a9556a3eec1fdaf1/resources/pico%20and%20hardware.png" width="400" alt="Architectural Diagram"/>
</p>
<p align="center"><i><b>Figure 1:</b> Mechanical System Set Up</i></p>

# **Codebase Information**

This system improves accuracy in high-speed pill inspection using **color segmentation**, **temporal analysis**, and **hardware integration**. Compared to traditional systems that rely only on thresholding or single-frame detection, this project focuses on **robust real-time tracking**, synchronized with **physical actuation** for defect removal. It also has a web application to display detection status, count and real-time detected images.

<p align="center">
  <img src="https://github.com/KoalaThee/belt-detection-defect/blob/a404f017c6358e981516cc83a9556a3eec1fdaf1/resources/defect%20detection%20interface.png" width="700" alt="Architectural Diagram"/>
</p>
<p align="center"><i><b>Figure 2:</b> GUI Displaying DEFECT Pill Detection </i></p>

<p align="center">
  <img src="https://github.com/KoalaThee/belt-detection-defect/blob/a404f017c6358e981516cc83a9556a3eec1fdaf1/resources/ok%20detection%20interface.png" width="700" alt="Architectural Diagram"/>
</p>
<p align="center"><i><b>Figure 3:</b> GUI Displaying OK Pill Detection </i></p>

## **Unique Features**
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
