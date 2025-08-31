# Qwen-VL Redis Clients 🧩

## Overview
This module implements **Redis-integrated clients** for the **Qwen2.5-VL multimodal model**, enabling:
- **Image comprehension** (describe captured photos)
- **Blind-assistant chat** (general conversational support)
- **RAG-enhanced Q&A** (retrieval-augmented dialogue)
- **Navigation reasoning** (structured JSON instructions)

The system listens to **Redis Pub/Sub events** and routes them to specialized clients. Each client processes user input (text and/or image) with Qwen2.5-VL, then publishes results back to Redis in a standardized format.

---

## Features
- **Modular Redis Client Framework**
  - `RedisClient`: Base pub/sub client with lifecycle management.  
  - `Qwen_2_5_VLRedisClient`: Specialized client integrating Qwen2.5-VL inference.  

- **Streaming Response Support**
  - `StreamingTextBaseClient`: Handles streaming text generation with sentence-level buffering.  
  - Subclasses:
    - `ImageComprehendClient` → image description  
    - `ChatClient` → blind-assistant conversational support  
    - `RAGChatClient` → retrieval-augmented Q&A  

- **Navigation Client**
  - Collects full LLM output, parses into structured **JSON navigation commands**, and publishes via Redis.  

- **Robust Runtime**
  - Thread-safe with `RLock`.  
  - Stop/resume inference mid-stream using stop filters.  
  - Event-based modular routing with fine-grained filters.  

---

## Architecture
```

┌───────────────┐   Redis pub/sub    ┌──────────────────────┐
│  Input Events  │  ───────────────▶ │   Qwen-VL Clients    │
│ (ASR, Camera)  │                   │ - ImageComprehend    │
│                │                   │ - Chat               │
└───────┬────────┘                   │ - RAGChat            │
│                                    │ - Navigation         │
▼                                    └───────────┬──────────┘
┌──────────────────┐                             │
│   Redis Channels  │   publish results (JSON)   │
│ - events\:qwen\_\*│ ◀────────────────────────┘
│ - events\:navigate│
└───────────────────┘

```

---

## Requirements
- **Python 3.8+**
- **Redis** (running locally)
- **Dependencies**:
```bash
  pip install redis numpy torch
```

* **Model**: Qwen2.5-VL (deployed on Sophgo BM1684X)

  * bmodel: `qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048.bmodel`
  * tokenizer + processor configs
  * `config.json`

---

## Usage

### 1. Start Redis

```bash
redis-server
```

### 2. Run the Clients

```bash
python QwenVLClients.py
```

Expected log:

```
✅ All services started successfully, listening to Redis...
✅ Startup audio command sent.
```

### 3. Send Test Input

Publish a launch event to Redis:

```bash
redis-cli publish events:asr_result '{"instruction": "chat", "text": "你好"}'
```

Expected result:

* Client `ChatClient` receives event.
* Qwen2.5-VL generates text.
* Final reply published to:

```
events:qwen_reply_result
```

---

## Redis Channels

### Input Events

* **Image Capture**

  ```json
  {"eventType": "image_capture", "path": "/tmp/photo.jpg"}
  ```

  → Channel: `event:vision:photo`

* **Chat / RAG / Navigation**

  ```json
  {"instruction": "chat", "text": "Hello!"}
  {"instruction": "chatrag", "text": "What is CRISPR?"}
  {"instruction": "navigation", "text": "Guide me to the exit"}
  ```

  → Channel: `events:asr_result`

* **Stop Keyword**

  ```json
  {"type": "kws_detection", "keyword": "stop"}
  ```

  → Channel: `events:kws`

### Output Events

* **Text Reply (Chat, RAG, Image)**

  ```json
  {"content": "This is a tree.", "priority": 2}
  ```

  → Channel: `events:qwen_reply_result`

* **Navigation Result**

  ```json
  {"action": "move_forward", "distance": 3.5}
  ```

  → Channel: `events:qwen_navigate_result`

---

## Code Structure

```
qwen_vl/
├── QwenVLClients.py         # Main client definitions
├── qwen2_5_vl_kou.py        # Qwen2.5-VL wrapper
├── prompt_manager.py        # System prompts
└── configs/                 # Tokenizer & processor configs
```

---

## Authors & Acknowledgments

Developed by:

* **Ziang Ren**, **Benhao Qing**, **Yunzhuo Yang**

Special thanks to **Prof. Hai Li** for invaluable guidance.

---




