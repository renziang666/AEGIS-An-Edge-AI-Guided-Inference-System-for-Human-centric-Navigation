import threading
from speaker import Speaker
import redis
import json
import time
from collections import deque

# --- 1. 全局队列和信号 ---
high_priority_queue = deque()
medium_priority_queue = deque()
low_priority_queue = deque()
shutdown_event = threading.Event()
# 这个“信号旗”是实现逻辑的关键
stop_requested_event = threading.Event() 

# --- 2. Redis 监听者线程 ---
def redis_listener(redis_conn):
    channels_to_subscribe = {
        "audio:playcommand": handle_play_command,
        "events:kws": handle_kws_command
    }
    pubsub = redis_conn.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(*channels_to_subscribe.keys())
    print(f"🎧 Redis listener thread started. Subscribed to: {list(channels_to_subscribe.keys())}")

    while not shutdown_event.is_set():
        try:
            message = pubsub.get_message(timeout=1.0)
            if message is None:
                continue
            
            channel = message['channel'].decode('utf-8')
            handler = channels_to_subscribe.get(channel)
            
            if handler:
                handler(message['data'])

        except Exception as e:
            if not shutdown_event.is_set():
                print(f"❌ An error occurred in listener thread: {e}")
    
    print("🎧 Redis listener thread finished.")

def handle_play_command(data):
    command = json.loads(data)
    action = command.get('action')
    if action == 'play':
        # ... (省略添加任务到队列的逻辑，这部分不变)
        path = command.get('path')
        priority = command.get('key', 3) 
        if not path: return
        if priority == 1: high_priority_queue.append(path); print(f"➕ Added to HIGH: {path}")
        elif priority == 2: medium_priority_queue.append(path); print(f"➕ Added to MEDIUM: {path}")
        else: low_priority_queue.append(path); print(f"➕ Added to LOW: {path}")
    elif action == 'stop':
        print("🚨 Received global stop command.")
        # **第一步: 产生“停止”信号**
        stop_requested_event.set()

def handle_kws_command(data):
    command = json.loads(data)
    keyword = command.get('keyword')
    stop_keywords = ["no", "stop", "off"]
    print(f"👁️ KWS received: '{keyword}'")
    if keyword in stop_keywords:
        print(f"🚨 Stop keyword '{keyword}' detected!")
        # **第一步: 产生“停止”信号**
        stop_requested_event.set()

# --- 3. 主程序 ---
speaker = Speaker()
redis_conn = redis.Redis(host='localhost', port=6379) 
speaker.play_voice(wav_path="/data/preaudio/002.wav")

listener_thread = threading.Thread(target=redis_listener, args=(redis_conn,))
listener_thread.start()
print("📢 Audio Player Service started. Supervisor loop is running.")
current_playing_priority = None

try:
    while not shutdown_event.is_set():
        # **第二步: 主循环最优先响应“停止”信号**
        if stop_requested_event.is_set():
            print("🛑 Stop request received. Stopping playback and clearing all queues.")
            
            # **第三步(A): 执行停止播放**
            speaker.stop_playback()

            # **第三步(B): 执行清空所有队列**
            high_priority_queue.clear()
            medium_priority_queue.clear()
            low_priority_queue.clear()
            
            # **第四步: 恢复正常 - 放下“信号旗”，准备接收新任务**
            stop_requested_event.clear()

            if current_playing_priority is not None:
                print(f"✅ Priority {current_playing_priority} task was stopped.")
                current_playing_priority = None
            
            time.sleep(0.1) 
            # `continue`让循环立即重新开始，此时系统是干净的空闲状态
            continue 

        # --- 正常播放逻辑 (只有在没有停止信号时才会执行) ---
        if speaker.is_playing():
            should_interrupt = False
            # 规则1：如果当前播放的不是最高优先级(>1)，且高优先级队列有新任务，则中断。
            if high_priority_queue and current_playing_priority > 1:
                should_interrupt = True
                new_task_priority = 1
                new_task_path = high_priority_queue.popleft()
                print("🔥 High priority task (1) arrived. Preparing to interrupt lower priority task.")

            # 规则2：如果当前播放的是最低优先级(==3)，且中等优先级队列有新任务，则中断。
            elif medium_priority_queue and current_playing_priority > 2: # 实际上只有 current_playing_priority == 3
                should_interrupt = True
                new_task_priority = 2
                new_task_path = medium_priority_queue.popleft()
                print("🔔 Medium priority task (2) arrived. Preparing to interrupt LOW priority task (3).")

            # 如果决定了要中断，就执行清晰的“先停后播”
            if should_interrupt:
                speaker.stop_playback() 
                # 为了立即响应，我们在这里直接播放
                print(f"▶️ Playing new priority {new_task_priority} task immediately: {new_task_path}")
                speaker.play_voice(new_task_path)
                current_playing_priority = new_task_priority

        else: # 如果扬声器空闲
            if current_playing_priority is not None:
                print(f"✅ Priority {current_playing_priority} task finished.")
                current_playing_priority = None

            next_task_path = None
            next_task_priority = None
            # 按优先级从队列取新任务
            if high_priority_queue:
                next_task_path, next_task_priority = high_priority_queue.popleft(), 1
            elif medium_priority_queue:
                next_task_path, next_task_priority = medium_priority_queue.popleft(), 2
            elif low_priority_queue:
                next_task_path, next_task_priority = low_priority_queue.popleft(), 3
            
            if next_task_path:
                print(f"▶️ Playing from priority {next_task_priority} queue: {next_task_path}")
                speaker.play_voice(next_task_path)
                current_playing_priority = next_task_priority
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nShutdown signal received (Ctrl+C).")
finally:
    print("Shutting down service...")
    shutdown_event.set()
    speaker.stop_playback() 
    print("Waiting for listener thread to exit...")
    listener_thread.join()
    redis_conn.close()
    print("✅ Service shut down gracefully.")