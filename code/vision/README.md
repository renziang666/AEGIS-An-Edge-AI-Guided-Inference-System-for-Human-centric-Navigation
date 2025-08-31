# Vision Clients

## Overview
This module provides a collection of **Redis-based vision clients** that integrate with computer vision models and a camera feed.  
Each client runs as an independent thread, performing detection/recognition tasks and publishing results through Redis channels.  

Supported functions include:  
- **OCRClient**: Bus number recognition triggered by a gamepad button (L2).  
- **DETClient**: Continuous obstacle detection and warning broadcast.  
- **DESClient**: Detailed obstacle description triggered by a gamepad button (R1).  
- **StepDetectorClient**: Continuous stair detection and warning broadcast.  
- **PavementDetectorClient**: Detects tactile paving, curbs, potholes, and water surfaces (triggered by R2).  
- **QWenPhotoClient**: Captures and saves photos when triggered by the L1 button.  

A central **FrameGrabber** ensures efficient camera frame capture, shared among all clients.

---

## Features
- Multi-threaded vision client framework with Redis pub/sub integration.  
- Modular design: easily extendable for new vision tasks.  
- Automatic cooldown for repeated obstacle broadcasts (prevents spamming).  
- Configurable model paths (YOLOv8, PP-OCR, custom stair/pavement detectors).  
- Gamepad-triggered actions for interactive testing.  
- Built-in logging for debugging and monitoring.  

---

## Architecture
```

Camera → FrameGrabber → \[Multiple Clients] → Redis Pub/Sub → Downstream Services

```

Each client:
1. Captures frames from `FrameGrabber`.  
2. Runs model inference (detection/recognition).  
3. Processes results (OCR, warnings, JSON descriptions, etc.).  
4. Publishes structured messages to Redis.  

---

## Requirements
- **Python 3.8+**  
- **Redis** (running locally or accessible via host/port)  
- Python dependencies:
```bash
  pip install redis opencv-python-headless
```

* Model dependencies (compiled for **SOPHON BM1684X TPU**):

  * YOLOv8 detection models (`.bmodel`)
  * PP-OCR recognition models
  * Stair and pavement detection models

---

## Usage

### 1. Start Redis

Make sure a Redis server is running:

```bash
redis-server
```

### 2. Run the Vision Clients

```bash
python vision_clients.py
```

Expected console output:

```
🚀 [Main] Initializing camera...
✅ [Main] FrameGrabber ready.
🚀 [Main] Initializing vision clients...
✅ [Main] All clients started.
   - DETClient: running obstacle detection
   - OCRClient: waiting for 'L2' button press
   - StepDetectorClient: running stair detection
   - PavementDetectorClient: waiting for 'R1' button press
   - QWenPhotoClient: waiting for 'L1' button press
--- Press Ctrl+C to stop safely ---
```

### 3. Stopping

Press `Ctrl+C` to gracefully shut down all clients and release resources.

---

## Redis Channels

| Client                     | Subscribed Event      | Published Channel        | Payload Example                                                                       |
| -------------------------- | --------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| **OCRClient**              | `gamepad:events` (L2) | `bus:number:detect`      | `{"eventType": "bus_number_detected", "result": "123", "content": "Bus 123 arrived"}` |
| **DETClient**              | — (continuous)        | `channel:vision_to_tts`  | `{"eventType": "obstacle_detected", "content": "Obstacle ahead"}`                     |
| **DESClient**              | `gamepad:events` (R1) | `event:vision:detection` | `{"eventType": "obstacle_described", "result": "...json..."}`                         |
| **StepDetectorClient**     | — (continuous)        | `channel:vision_to_tts`  | `{"eventType": "stair_detected", "content": "Stair detected"}`                        |
| **PavementDetectorClient** | `gamepad:events` (R2) | `event:vision:detection` | `{"eventType": "pavement_detected", "content": "Nearby tactile paving detected"}`     |
| **QWenPhotoClient**        | `gamepad:events` (L1) | `event:vision:photo`     | `{"eventType": "image_capture", "path": "/path/to/captured.jpg"}`                     |

---

## Code Structure

```
vision_clients.py
├── RedisClient           # Base Redis client with threading control
├── FrameGrabber          # Camera capture manager (shared among clients)
├── OCRClient             # Bus number recognition (L2 trigger)
├── DETClient             # Continuous obstacle detection
├── DESClient             # Obstacle description (R1 trigger)
├── StepDetectorClient    # Continuous stair detection
├── PavementDetectorClient# Pavement/tactile paving detection (R2 trigger)
├── QWenPhotoClient       # Photo capture (L1 trigger)
└── main()                # Initializes and manages all clients
```


---

## Acknowledgments

This work was implemented by **Ziang Ren, Benhao Qin, and Yunzhuo Yang**,
with special thanks to **Prof. Hai Li** for guidance and support.




