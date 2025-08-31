# Streaming ASR System (WeNet + Redis)

## Overview
This project implements a **three-layer streaming Automatic Speech Recognition (ASR) system** built on **WeNet** and **Redis pub/sub**.  
It enables real-time audio recognition, event-driven processing, and flexible integration with downstream services.

### Architecture
```

┌──────────────────────────────────┐
│         Application Layer         │
│ (Navigation / Chat / RAG Modules) │
└──────────────────┬────────────────┘
│ Redis Pub/Sub
┌──────────────────▼────────────────┐
│          Service Layer            │
│        (service.py)               │
│ - Listens to `events:audio`       │
│ - Calls StreamingRecognizer        │
│ - Publishes ASR results            │
└──────────────────┬────────────────┘
│ API calls
┌──────────────────▼────────────────┐
│   Recognition Wrapper Layer        │
│ (streaming\_recoginizer.py)         │
│ - Loads WeNet encoder/decoder      │
│ - Manages dataset & inference      │
│ - Provides `recognizing()` API     │
└──────────────────┬────────────────┘
│ Core inference
┌──────────────────▼────────────────┐
│   Core Model Layer (WeNet)         │
│ - Sophon TPU inference engine      │
│ - CTC & Attention Rescoring        │
│ - Data pipeline (Dataset, Utils)   │
└───────────────────────────────────┘

```

---

## Features
- **Three-tier modular design**: service, wrapper, core model.  
- **Redis-driven workflow**: subscribes to audio events, publishes recognition results.  
- **Multi-task handling**:
  - `navigation_recording` → ASR → `events:asr_result`  
  - `chat_recording` → ASR → `events:asr_result`  
  - `RAG_recording` → ASR → `events:asr2rag`  
  - `RAG_input` → ASR → `events:asr2rag`  
- **Streaming + Non-streaming recognition** support.  
- **Attention rescoring** for improved accuracy.  
- **Sophon TPU acceleration** with `.bmodel` files.  

---

## Requirements
- **Python 3.8+**  
- **Redis** server  
- Dependencies:
```bash
    pip install redis torch pyyaml numpy
```

* Models:

  * `wenet_encoder_non_streaming_fp32.bmodel`
  * `wenet_decoder_fp32.bmodel`
  * `train_u2++_conformer.yaml`
  * `lang_char.txt`

---

## Usage

### 1. Start Redis

```bash
redis-server
```

### 2. Run ASR Service

```bash
python service.py
```

Console output:

```
Subscribed to Redis channel: events:audio
Waiting for new audio events...
```

### 3. Publish an Audio Event

Example:

```bash
redis-cli publish events:audio '{"type": "chat_recording", "path": "/path/to/audio.wav"}'
```

The ASR service will:

* Detect the new audio file
* Run speech recognition via `StreamingRecognizer`
* Publish results to the correct Redis channel

---

## Redis Workflow

| Input Event (to `events:audio`)                   | Recognition Output                             | Publish Channel     |
| ------------------------------------------------- | ---------------------------------------------- | ------------------- |
| `{"type": "navigation_recording", "path": "..."}` | `{"instruction": "navigation", "text": "..."}` | `events:asr_result` |
| `{"type": "chat_recording", "path": "..."}`       | `{"instruction": "chat", "text": "..."}`       | `events:asr_result` |
| `{"type": "RAG_recording", "path": "..."}`        | `{"instruction": "rag_query", "text": "..."}`  | `events:asr2rag`    |
| `{"type": "RAG_input", "path": "..."}`            | `{"instruction": "rag_input", "text": "..."}`  | `events:asr2rag`    |

---

## Code Structure

```
asr-system/
├── service.py               # Service Layer: Redis listener + dispatcher
├── streaming_recoginizer.py # Recognition Wrapper Layer
├── wenet.py                 # Core Model Layer (TPU inference)
├── config/
│   ├── train_u2++_conformer.yaml
│   └── lang_char.txt
├── models/
│   ├── wenet_encoder_non_streaming_fp32.bmodel
│   └── wenet_decoder_fp32.bmodel
└── result.txt               # Recognition outputs (for debugging)
```

---



## Acknowledgments

This project was developed by **Ziang Ren, Benhao Qing, and Yunzhuo Yang**,
with special thanks to **Prof. Hai Li** for guidance and support.


