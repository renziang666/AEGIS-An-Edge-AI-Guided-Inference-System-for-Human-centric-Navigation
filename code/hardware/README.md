# Real-Time Keyword Spotting System 🎙️

## Overview
This project implements a **real-time microphone-based Keyword Spotting (KWS) system** built with:
- **TensorFlow (custom trained model)**  
- **PyAudio** for microphone streaming  
- **Redis pub/sub** for inter-process communication  
- **Multiprocessing + Watchdog** for robust continuous operation  

It continuously listens to microphone input, performs **keyword detection**, and publishes recognized events to Redis channels for downstream use (e.g., ASR, navigation, chat, or RAG modules).

---

## Features
- **Custom KWS Model**  
  - Trained on **Google Speech Commands dataset** + **CommonVoice-ZH** + **environmental noise samples**.  
  - Supports both binary classification (e.g., detect "yes") and multi-class classification (e.g., `yes`, `no`, `up`, `down`, ...).  

- **Real-time microphone pipeline**  
  - 48kHz USB microphone input (stereo) → 16kHz mono preprocessing.  
  - Buffered streaming with adjustable window size.  

- **Keyword Spotting Engine**  
  - TensorFlow SavedModel inference.  
  - Predicts keywords directly from in-memory audio buffers.  
  - Confidence thresholding + refractory period to avoid repeated triggers.  

- **Redis Integration**  
  - Publishes detection results to `events:kws`.  
  - Publishes recorded audio segments to `events:audio` for downstream ASR/RAG modules.  
  - Listens to `gamepad:events` for external triggers (navigation/chat/RAG).  

- **System reliability**  
  - Watchdog thread monitors queue health and automatically restarts KWS if stalled.  
  - Multiprocessing ensures isolation between recording and keyword inference.  
  - Performance monitoring via `psutil` (CPU & memory tracking).  
  - End-to-end latency logging (from detection to Redis publish).  

---

## Architecture
```

┌─────────────────────────────┐
│         Microphone           │
│  - PyAudio recording         │
│  - 48kHz → 16kHz resampling  │
│  - Circular buffer           │
└───────────────┬─────────────┘
│
┌───────────────▼─────────────┐
│  KWS Process (TensorFlow)   │
│  - Model inference           │
│  - Keyword detection         │
│  - Publishes to Redis        │
└───────────────┬─────────────┘
│
┌───────▼───────────┐
│ Redis Pub/Sub      │
│ - events\:kws       │
│ - events\:audio     │
│ - gamepad\:events   │
└────────────────────┘

```

---

## Requirements
- **Python 3.8+**  
- **Redis** (must be running locally)  
- Install dependencies:
```bash
  pip install pyaudio numpy librosa redis tensorflow tensorflow-datasets psutil
```

* **Hardware**

  * USB microphone (named `"MINI"` by default, modify in `Microphone._find_device_index_by_name` if needed).

---

## Usage

### 1. Train or Provide a Model

Train a custom KWS model (binary or multi-class). Example training script:

```python
# Using TensorFlow Datasets + Speech Commands + CommonVoice-ZH
# Converts target keyword into binary classification
# Saves trained model into: ./tensorflow/model2/saved
```

After training, export your model to:

```
/home/linaro/smart_cane_project/hardware/microphone/tensorflow/model2/saved
```

### 2. Start Redis

```bash
redis-server
```

### 3. Run the KWS Service

```bash
python microphone_kws.py
```

Console output will include:

```
🚀 [KWS进程] 关键词识别进程已启动。
✅ [初始化] 成功找到 MINI 麦克风
✅ [KWS模型] TensorFlow 模型加载成功！
📡 [Redis监听] 线程已启动
```

### 4. Example Redis Output

When the system detects a keyword:

```json
{
  "type": "kws_detection",
  "keyword": "yes",
  "confidence": 0.91,
  "timestamp": 1725100000.123
}
```

Published to channel:

```
events:kws
```

When a full audio segment is saved (e.g., navigation):

```json
{
  "type": "navigation_recording",
  "path": "/data/RECOsys_data_cache/Microphone/capture_20230830_123456.wav",
  "timestamp": 1725100050.789
}
```

Published to channel:

```
events:audio
```

---

## Code Structure

```
kws-system/
├── microphone_kws.py         # Main microphone + KWS + Redis integration
├── tensorflow/
│   └── model2/saved/         # Saved TensorFlow KWS model
├── dataset/                  # Speech Commands + CommonVoice-ZH dataset
└── logs/                     # Runtime logs (optional)
```

---

## Performance Monitoring

The main loop tracks:

* **KWS Process CPU & Memory usage** (via `psutil`)
* **End-to-end latency** (from keyword timestamp → Redis publish → main process reception)

Sample log:

```
--- [性能监控] | KWS进程 | CPU: 12.45% | 内存: 48.32 MB ---
🎯 [端到端延迟测试] 收到关键词: 'yes'
   耗时: 85.31 ms
```

---



## Acknowledgments

* KWS model trained on **Google Speech Commands dataset**, **Mozilla CommonVoice-ZH**, and custom environmental samples.
* Developed by **Ziang Ren, Benhao Qing, and Yunzhuo Yang**.
* Special thanks to **Prof. Hai Li** for valuable guidance.


