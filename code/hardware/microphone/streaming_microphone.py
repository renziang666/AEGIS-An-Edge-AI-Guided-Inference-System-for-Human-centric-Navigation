import pyaudio
import wave
import threading
import time
import multiprocessing
import collections
import tempfile
import os
import numpy as np
import pathlib
import tensorflow as tf
import queue
import librosa

import redis
import json
import uuid
from typing import Union

# ===================================================================
# 您的模型加载和预测函数 (增加了日志)
# ===================================================================
def load_my_model():
    """加载模型。"""
    print("[KWS模型] 正在加载 TensorFlow 模型...")
    saved_model_path = '/home/linaro/smart_cane_project/hardware/microphone/tensorflow/model2/saved'
    try:
        loaded_model = tf.saved_model.load(saved_model_path)
        print("✅ [KWS模型] TensorFlow 模型加载成功！")
        return loaded_model
    except Exception as e:
        print(f"❌ [KWS模型] 加载模型失败: {e}")
        return None

def run_model_prediction(audio_bytes: bytes, model) -> dict:
    """
    通用预测函数：【最终版】
    直接在内存中处理音频字节，不再读写文件。
    """
    label_names = ['backward' ,'down' ,'follow' ,'forward' ,'go' ,'left' ,'no' ,'noise' ,'off' ,'on' ,'right' ,'stop' ,'up' ,'yes']
    
    try:
        # 1. 字节 -> int16 NumPy 数组
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # 2. int16 -> float32 NumPy 数组 (归一化)
        audio_float = audio_np.astype(np.float32) / 32768.0

        # 3. 确保长度为16000 (1秒)，这是KWS模型的常见要求
        target_len = 16000
        if len(audio_float) > target_len:
            audio_float = audio_float[:target_len]
        elif len(audio_float) < target_len:
            audio_float = np.pad(audio_float, (0, target_len - len(audio_float)), 'constant')
        
        # 4. NumPy 数组 -> TensorFlow 张量
        audio_tensor = tf.constant(audio_float, dtype=tf.float32)
        
        # 5. 增加批次维度以匹配模型输入 (16000,) -> (1, 16000)
        audio_tensor = tf.expand_dims(audio_tensor, 0)

        # 6. 使用张量进行预测
        predictions = model(audio_tensor)

        # 7. 处理预测结果
        predicted_logits = predictions['predictions']
        probabilities = tf.nn.softmax(predicted_logits).numpy().flatten()
        predicted_class_id = np.argmax(probabilities)
        predicted_word = label_names[predicted_class_id]
        confidence = probabilities[predicted_class_id]

        return {"predicted_word": predicted_word, "confidence": float(confidence)}

    except Exception as e:
        print(f"❌ [KWS模型] 内存预测时出错: {e}")
        return {"predicted_word": None, "error": str(e)}

# ===================================================================
# KWS 目标进程 (增加了日志)
# ===================================================================
def keyword_spotting_process_target(
    audio_queue: multiprocessing.Queue,
    channels: int,
    rate: int,
    sample_width: int,
    prediction_interval: float = 0.35,
    buffer_duration: float = 1.0
):
    print("🚀 [KWS进程] 关键词识别进程已启动。")
    try:
        print("    [KWS进程] 正在连接到 Redis...")
        redis_conn = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_conn.ping()
        print("    ✅ [KWS进程] Redis 连接成功。")
    except redis.exceptions.ConnectionError as e:
        print(f"    ❌ [KWS进程] Redis 连接失败: {e}。进程退出。")
        return

    loaded_model = load_my_model()
    if loaded_model is None:
        print("    ❌ [KWS进程] 模型未能加载，进程退出。")
        return

    target_keywords = ['backward', 'down', 'follow', 'forward', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    print(f"    [KWS进程] 目标关键词: {target_keywords}")
    
    buffer_size_in_bytes = int(rate * buffer_duration * sample_width)
    audio_buffer = collections.deque()
    current_buffer_bytes = 0
    last_prediction_time = time.time()
    
    buffer_filled_once = False
    debug_counter = 0

    print("    [KWS进程] 初始化完毕，开始从队列获取音频数据，准备填充初始缓冲区...")
    while True:
        try:
            # 使用带超时的 get，避免在主进程异常退出时永久阻塞
            audio_chunk = audio_queue.get(timeout=1.0)
            if audio_chunk is None:
                print("    [KWS进程] 收到停止信号(None)，进程即将退出。")
                break

            # 缓冲区管理
            audio_buffer.append(audio_chunk)
            current_buffer_bytes += len(audio_chunk)
            while current_buffer_bytes > buffer_size_in_bytes:
                old_chunk = audio_buffer.popleft()
                current_buffer_bytes -= len(old_chunk)
                
                if not buffer_filled_once:
                    print("    ✅ [KWS进程] 初始缓冲区已满，开始进行循环预测。")
                    buffer_filled_once = True
            
            current_time = time.time()
            

                # 只有在缓冲区满过一次后才进行预测
            if buffer_filled_once and (current_time - last_prediction_time > prediction_interval):
                last_prediction_time = current_time
                
                # 1. 从缓冲区获取音频字节快照
                buffer_snapshot = b"".join(list(audio_buffer))
                
                # 2. 直接调用新的、基于内存的预测函数
                prediction = run_model_prediction(buffer_snapshot, loaded_model)

                predicted_word = prediction.get("predicted_word")
                confidence = prediction.get("confidence", 0)
                
                #print(f"    [KWS进程] 预测结果: '{predicted_word}', 置信度: {confidence:.2f}") # 日志太频繁，调试时开启

                if predicted_word in target_keywords and confidence > 0.5:
                    final_result = {
                        "type": "kws_detection",
                        "keyword": predicted_word,
                        "confidence": confidence,
                        "timestamp": time.time()
                    }
                    print(f"!!! 🎯 [KWS进程] 关键词已发现 !!! 结果: {final_result}")
                    try:
                        print("    [KWS进程] 正在通过 Redis 发布结果...")
                        redis_conn.publish("events:kws", json.dumps(final_result))
                        print("    ✅ [KWS进程] Redis 发布成功。")
                    except redis.exceptions.ConnectionError as e:
                        print(f"    ❌ [KWS进程] 发布到 Redis 失败: {e}")
                    
                    # 【关键修复 1】: 清空缓冲区
                    print("    [KWS进程] 清空音频缓冲区，为下一次识别做准备。")
                    audio_buffer.clear()
                    current_buffer_bytes = 0
                    buffer_filled_once = False # 重置状态，需要重新填满缓冲区
                    print("    [KWS进程] 状态已重置，将重新填充缓冲区。")
                    
                    # 【关键修复 2】: 设置不应期
                    print("    [KWS进程] 进入2秒不应期...")
                    time.sleep(2.0)
                    last_prediction_time = time.time() 
                    print("    [KWS进程] 不应期结束，恢复正常监听。")


        except queue.Empty:
            # 队列在1秒内是空的，这是正常现象，说明上游没有数据过来
            # print("    [KWS进程] 队列暂时为空，继续等待...") # 这条日志通常没必要，除非你想确认进程还活着
            continue
        except Exception as e:
            print(f"❌ [KWS进程] 在主循环中发生未知错误: {e}")
    
    print("🛑 [KWS进程] 进程已自然退出。")


# ===================================================================
# Microphone 类 (增加了日志)
# ===================================================================
class Microphone:

    def __init__(self, buffer_seconds=10, wav_path="/data/RECOsys_data_cache/Microphone"):
        print("🚀 [主程序] Microphone 服务开始初始化...")
        self.wav_path = wav_path
        os.makedirs(self.wav_path, exist_ok=True)
        
        print("    [初始化] 正在初始化 PyAudio...")
        self._pa = pyaudio.PyAudio()

        print("    [初始化] 正在查找麦克风设备...")
        self.device_index = self._find_device_index_by_name("MINI")
        if self.device_index is None:
            print("    ❌ [初始化] 致命错误：找不到名为 'MINI' 的USB麦克风！请检查设备连接。程序将无法录音。")
            # 在这种情况下，可以考虑直接 raise Exception 来中断程序
            return
        else:
            print(f"    ✅ [初始化] 成功找到 MINI 麦克风，设备索引为: {self.device_index}")

        # ... (其他初始化变量)
        self._is_recording = False
        self._recording_thread = None
        self._relay_thread = None
        self._kws_process = None
        self._redis_listener_thread = None

        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.HARDWARE_RATE = 48000
        self.TARGET_RATE = 16000
        self.SAMPLE_WIDTH = 2

        buffer_size_in_chunks = int((self.HARDWARE_RATE / self.CHUNK) * buffer_seconds)
        self.audio_buffer = collections.deque(maxlen=buffer_size_in_chunks)
        print(f"    [初始化] 滚动缓冲区已创建，将保留最近 {buffer_seconds} 秒的音频。")

        self._shutdown_event = threading.Event()
        self._save_now_event = threading.Event()
        self._enable_recognition = False

        self.thread_queue = queue.Queue(maxsize=100) # 给内部队列一个大小限制，防止意外的内存增长
        self.multiprocess_queue = multiprocessing.Queue(maxsize=100) # 跨进程队列也加上限制

        self.toggle_recognition(True)

        self.action = None

        print("    [初始化] 正在连接到 Redis...")
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            print("    ✅ [初始化] Redis 连接成功！")
        except redis.exceptions.ConnectionError as e:
            print(f"    ❌ [初始化] Redis 连接失败: {e}. 程序将无法响应外部命令。")
            self.redis_client = None
        print("✅ [主程序] Microphone 服务初始化完成。")

    def _watchdog_thread_target(self):
        """【新增】看门狗线程，监控KWS进程的健康状况"""
        print("🐶 [看门狗] 线程已启动。")
        
        # 队列持续满多少秒后触发重启
        queue_full_threshold_seconds = 10.0 
        
        time_queue_started_being_full = None

        while not self._shutdown_event.is_set():
            time.sleep(2) # 每2秒检查一次

            if not self._enable_recognition:
                # 如果识别功能是关闭的，则重置计时器并继续
                time_queue_started_being_full = None
                continue

            try:
                # 检查队列是否接近满载
                qsize = self.multiprocess_queue.qsize()
                if qsize > 90: # 队列使用率超过90%
                    if time_queue_started_being_full is None:
                        # 第一次发现队列满，记录当前时间
                        print(f"    ⚠️ [看门狗] 检测到队列拥堵 (大小: {qsize})，开始观察...")
                        time_queue_started_being_full = time.time()
                    else:
                        # 队列持续拥堵，检查是否已超过阈值
                        duration = time.time() - time_queue_started_being_full
                        print(f"    ⚠️ [看门狗] 队列持续拥堵 {duration:.1f} 秒...")
                        if duration > queue_full_threshold_seconds:
                            print("    🚨 [看门狗] KWS进程可能已卡死！触发自动重启...")
                            self.toggle_recognition(False)
                            time.sleep(1) # 等待清理完成
                            self.toggle_recognition(True)
                            print("    ✅ [看门狗] KWS服务重启完毕。")
                            # 重置计时器
                            time_queue_started_being_full = None
                else:
                    # 队列恢复正常，重置计时器
                    if time_queue_started_being_full is not None:
                        print("    👍 [看门狗] 队列拥堵已缓解。")
                    time_queue_started_being_full = None

            except NotImplementedError:
                # 在某些平台上，multiprocessing.Queue.qsize() 可能不可用
                # 这里可以留空或寻找其他健康检查方式
                pass
            except Exception as e:
                print(f"    ❌ [看门狗] 线程出现错误: {e}")

        print("🛑 [看门狗] 线程已退出。")

    def _find_device_index_by_name(self, name_keyword: str) -> Union[int, None]:
        # ... (这个函数逻辑很简单，暂时不需要加日志)
        num_devices = self._pa.get_device_count()
        for i in range(num_devices):
            info = self._pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and name_keyword.lower() in info['name'].lower():
                return i
        return None

    def _redis_listener_thread_target(self):
        """后台监听 Redis 的 'gamepad:events' 频道"""
        print("📡 [Redis监听] 线程已启动，正在订阅 'gamepad:events' 频道...")
        if not self.redis_client:
            print("    ❌ [Redis监听] Redis 未连接，线程自动退出。")
            return
        
        pubsub = self.redis_client.pubsub()
        pubsub.subscribe("gamepad:events")

        for message in pubsub.listen():
            if self._shutdown_event.is_set():
                break

            if message['type'] != 'message':
                continue

            print(f"    [Redis监听] 收到原始消息: {message}")
            try:
                event_data = json.loads(message['data'])
                print(f"    [Redis监听] 解析后事件: {event_data}")

                event_type = event_data.get('type')
                button_name = event_data.get('name')
                button_state = event_data.get('state')

                if event_type == 'button' and button_name == 'Y' and button_state == 'pressed':
                    print("    [Redis监听] 检测到 'X'(真实按键) 键按下，记录导航信息")
                    self.action = "navigation"
                    self.trigger_save()

                if event_type == 'button' and button_name == 'A' and button_state == 'pressed':
                    print("    [Redis监听] 检测到 'A' 键按下，将记录聊天信息")
                    # self.toggle_recognition(True)
                    self.action = "chat"
                    self.trigger_save()

                if event_type == 'button' and button_name == 'B' and button_state == 'pressed':
                    print("    [Redis监听] 检测到 'B' 键按下，将准备RAG输入！")
                    self.action = "RAG_serve"
                    self.trigger_save()

                if event_type == 'button' and button_name == 'X' and button_state == 'pressed':
                    print("    [Redis监听] 检测到 'Y' 键按下，将准备录入个人信息[RAG]！")
                    self.action = "RAG_input"
                    self.trigger_save()

            except (json.JSONDecodeError, TypeError):
                 # 'shutdown'消息不是json，会触发TypeError，这是正常的
                 if isinstance(message['data'], str) and "shutdown" in message['data']:
                     print("    [Redis监听] 收到关闭指令，准备退出。")
                 else:
                    print(f"    [Redis监听] 无法解析收到的消息: {message['data']}")
            except Exception as e:
                print(f"    ❌ [Redis监听] 处理消息时出错: {e}")

        print("🛑 [Redis监听] 线程已退出。")

    def _recorder_thread_target(self):
        """【录音主线程】处理音频读取、转换和降采样"""
        print("🎧 [录音线程] 线程已启动。")
        stream = None
        try:
            print(f"    [录音线程] 正在以 {self.HARDWARE_RATE} Hz, {self.CHANNELS} 声道模式打开设备...")
            stream = self._pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.HARDWARE_RATE,
                                   input=True,
                                   input_device_index=self.device_index,
                                   frames_per_buffer=self.CHUNK)
            print("    ✅ [录音线程] 设备成功打开，开始循环读取音频...")
            
            chunk_counter = 0
            while not self._shutdown_event.is_set():
                audio_data_48k_stereo = stream.read(self.CHUNK, exception_on_overflow=False)
                chunk_counter += 1
                
                # ... (音频处理逻辑)
                audio_array_48k_stereo_int16 = np.frombuffer(audio_data_48k_stereo, dtype=np.int16)
                audio_array_48k_mono_int16 = audio_array_48k_stereo_int16[::2]
                audio_array_48k_mono_float32 = audio_array_48k_mono_int16.astype(np.float32) / 32768.0
                resampled_audio_16k_float32 = librosa.resample(y=audio_array_48k_mono_float32, orig_sr=self.HARDWARE_RATE, target_sr=self.TARGET_RATE)
                clipped_audio_float = np.clip(resampled_audio_16k_float32, -1.0, 1.0)
                resampled_audio_16k_int16 = (clipped_audio_float * 32767).astype(np.int16)
                final_audio_data_16k = resampled_audio_16k_int16.tobytes()
                

                # 放入滚动缓冲区
                self.audio_buffer.append(final_audio_data_16k)

                # 如果识别开启，则放入内部队列
                if self._enable_recognition:
                    try:
                        self.thread_queue.put(final_audio_data_16k, block=False)
                    except queue.Full:
                        print("    ⚠️ [录音线程] 警告: 内部线程队列已满，丢弃一个数据块！这表明中继线程处理不过来。")

                # 检查是否需要保存文件
                if self._save_now_event.is_set():
                    print("    [录音线程] ---> 收到保存指令！")
                    try:
                        buffer_snapshot = list(self.audio_buffer)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        unique_id = uuid.uuid4().hex[:8]
                        filename = f"capture_{timestamp}_{unique_id}.wav"
                        filepath = os.path.join(self.wav_path, filename)
                        
                        with wave.open(filepath, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(self.SAMPLE_WIDTH)
                            wf.setframerate(self.TARGET_RATE)
                            wf.writeframes(b"".join(buffer_snapshot))
                        print(f"    ✅ [录音线程] 音频已成功保存至 {filepath}")

                        if self.redis_client:
                            if self.action == "navigation":
                                event_message = { "type": "navigation_recording", "path": filepath, "timestamp": time.time() }
                            elif self.action == "chat":
                                event_message = { "type": "chat_recording", "path": filepath, "timestamp": time.time() }
                            elif self.action == "RAG_serve":
                                event_message = { "type": "RAG_recording", "path": filepath, "timestamp": time.time() }
                            elif self.action == "RAG_input":
                                event_message = { "type": "RAG_input", "path": filepath, "timestamp": time.time() }
                            self.action = None # 清除动作状态
                            self.redis_client.publish("events:audio", json.dumps(event_message))
                            print(f"    [录音线程] 已向 Redis 广播新文件路径。")
                    except Exception as e:
                        print(f"    ❌ [录音线程] 在保存或广播文件时出错: {e}")
                    finally:
                        self._save_now_event.clear() # 清除指令
                        print("    [录音线程] 保存指令已处理完毕。")

        except Exception as e:
            print(f"    ❌ [录音线程] 发生致命错误: {e}") 
        finally:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
            print("🛑 [录音线程] 线程结束，音频流已关闭。")

    def _relay_thread_target(self):
        """中继线程：负责从内部队列取出数据，再放入跨进程队列。"""
        print("🚚 [中继线程] 线程已启动。")
        while self._enable_recognition and not self._shutdown_event.is_set():
            try:
                audio_data = self.thread_queue.get(timeout=0.5)
                # print("    [中继线程] 从内部队列获取到数据，准备放入跨进程队列...")
                try:
                    self.multiprocess_queue.put(audio_data, timeout=0.5)
                    # print("    [中继线程] 数据成功放入跨进程队列。")
                except queue.Full:
                    print("    ⚠️ [中继线程] 警告: 跨进程队列已满！KWS进程处理不过来，丢弃一个数据块。")
            except queue.Empty:
                continue
        print("🛑 [中继线程] 线程已退出。")

    def start_listening(self):
        print("[主程序] 调用 start_listening()...")
        if self._recording_thread is not None:
            print("    [主程序] 监听线程已在运行，忽略此次调用。")
            return
        self._shutdown_event.clear()
        
        print("    [主程序] 正在启动录音线程...")
        self._recording_thread = threading.Thread(target=self._recorder_thread_target)
        self._recording_thread.start()
        
        if self.redis_client:
            print("    [主程序] 正在启动 Redis 监听线程...")
            self._redis_listener_thread = threading.Thread(target=self._redis_listener_thread_target)
            self._redis_listener_thread.start()
        
        print("    [主程序] 正在启动看门狗线程...")
        self._watchdog_thread = threading.Thread(target=self._watchdog_thread_target)
        self._watchdog_thread.start()
        
        print("[主程序] 所有监听组件已启动。")

    def trigger_save(self):
        print("[主程序] 调用 trigger_save()...")
        if not self._recording_thread or not self._recording_thread.is_alive():
            print("    ❌ [主程序] 错误：监听尚未启动，无法保存。")
            return
        print("    [主程序] ---> 向录音线程发出保存指令...")
        self._save_now_event.set()

    def toggle_recognition(self, start: bool):
        """动态地开启或关闭关键词识别功能。"""
        print(f"[主程序] 调用 toggle_recognition(start={start})...")
        if start and not self._enable_recognition:
            print("    [主程序] ---> 正在开启识别功能...")
            self._enable_recognition = True
            
            print("        [主程序] 正在创建 KWS 进程...")
            self._kws_process = multiprocessing.Process(
                target=keyword_spotting_process_target,
                args=(self.multiprocess_queue, 1, self.TARGET_RATE, self.SAMPLE_WIDTH))
            
            print("        [主程序] 正在创建中继线程...")
            self._relay_thread = threading.Thread(target=self._relay_thread_target)
            
            print("        [主程序] 正在启动 KWS 进程和中继线程...")
            self._kws_process.start()
            self._relay_thread.start()
            print("    ✅ [主程序] 识别功能已开启。")

        elif not start and self._enable_recognition:
            print("    [主程序] ---> 正在关闭识别功能...")
            self._enable_recognition = False # 这会使中继线程的循环退出
            
            if self._relay_thread and self._relay_thread.is_alive():
                print("        [主程序] 正在等待中继线程退出...")
                self._relay_thread.join(timeout=2.0)
                if self._relay_thread.is_alive():
                    print("        ⚠️ [主程序] 中继线程超时未退出。")
                else:
                    print("        ✅ [主程序] 中继线程已退出。")

            if self._kws_process and self._kws_process.is_alive():
                print("        [主程序] 正在向 KWS 进程发送停止信号...")
                self.multiprocess_queue.put(None)
                print("        [主程序] 正在等待 KWS 进程退出...")
                self._kws_process.join(timeout=2.0)
                if self._kws_process.is_alive():
                     print("       ⚠️ [主程序] KWS 进程超时未退出，将尝试强制终止。")
                     self._kws_process.terminate()
                else:
                    print("        ✅ [主程序] KWS 进程已退出。")
            
            # 清理队列
            print("        [主程序] 正在清理队列...")
            while not self.multiprocess_queue.empty(): self.multiprocess_queue.get_nowait()
            print("    ✅ [主程序] 识别功能已关闭。麦克风仍在后台监听。")

    def shutdown(self):
        """安全地关闭整个系统。"""
        print("🛑 [主程序] 调用 shutdown()，开始彻底关闭系统...")
        if self._shutdown_event.is_set():
            print("    [主程序] 系统已在关闭中，忽略此次调用。")
            return
        
        # 1. 先关闭识别功能
        if self._enable_recognition:
            self.toggle_recognition(False)
        
        # 2. 设置全局关闭信号
        print("    [主程序] 设置全局关闭信号...")
        self._shutdown_event.set()
        
        # 3. 唤醒并关闭 Redis 监听线程
        if self.redis_client and self._redis_listener_thread and self._redis_listener_thread.is_alive():
            print("    [主程序] 正在向 Redis 监听线程发送关闭消息以唤醒它...")
            try:
                self.redis_client.publish("gamepad:events", "shutdown")
            except Exception as e:
                print(f"        ❌ [主程序] 向 Redis 发送关闭消息时出错: {e}")
            self._redis_listener_thread.join(timeout=2.0)
            if not self._redis_listener_thread.is_alive():
                print("    ✅ [主程序] Redis 监听线程已退出。")
            else:
                print("    ⚠️ [主程序] Redis 监听线程超时未退出。")

        # 4. 等待录音线程关闭
        if self._recording_thread and self._recording_thread.is_alive():
            print("    [主程序] 正在等待录音线程退出...")
            self._recording_thread.join(timeout=2.0)
            if not self._recording_thread.is_alive():
                print("    ✅ [主程序] 录音线程已退出。")
            else:
                print("    ⚠️ [主程序] 录音线程超时未退出。")
        
        # 5. 最后释放 PyAudio 资源
        if self._pa:
            print("    [主程序] 正在释放 PyAudio 资源...")
            self._pa.terminate()
            print("    ✅ [主程序] PyAudio 资源已释放。")
            
        print("🏁 [主程序] 系统已安全关闭。")

if __name__ == "__main__":
    # ===================================================================
    # 只需要修改这个 main 部分
    # ===================================================================
    import psutil # 引入性能监控库

    recorder = Microphone(buffer_seconds=10)
    
    # 检查初始化是否成功（例如，麦克风是否找到）
    if getattr(recorder, 'device_index', None) is None:
        print("❌ 主程序: 初始化失败，程序退出。")
        exit()

    print("\n[主程序] 初始化完成，准备启动监听...")
    recorder.start_listening()
    
    # ---- 性能监控设置 ----
    kws_process = None
    try:
        # 等待一小会儿，确保KWS进程已启动并获取其PID
        time.sleep(2) 
        if recorder._kws_process and recorder._kws_process.is_alive():
            kws_process = psutil.Process(recorder._kws_process.pid)
            print(f"✅ [性能监控] 成功锁定 KWS 进程 (PID: {kws_process.pid})。")
        else:
            print("⚠️ [性能监控] 未能找到正在运行的 KWS 进程。")
    except (psutil.NoSuchProcess, AttributeError) as e:
        print(f"⚠️ [性能监控] 无法附加到 KWS 进程: {e}")
    # ----------------------
    
    print("\n[主程序] 系统正在运行。性能数据将每5秒更新一次。按下 Ctrl+C 来关闭程序。")
    
    # ---- 延迟测试设置 ----
    # 在主进程中也连接Redis，用来监听KWS的结果
    redis_latency_tester = redis.Redis(host='localhost', port=6379)
    pubsub = redis_latency_tester.pubsub()
    pubsub.subscribe("events:kws")
    print("✅ [延迟测试] 已订阅 Redis 'events:kws' 频道，准备接收识别结果。")
    # ----------------------

    try:
        while True:
            # ---- 1. 持续监控 CPU 和 内存 ----
            if kws_process and kws_process.is_running():
                # .cpu_percent(interval=None) 是非阻塞的，它会返回自上次调用以来的CPU使用率
                # 首次调用返回0.0，之后才能获取到有效值
                cpu_usage = kws_process.cpu_percent(interval=1.0) 
                
                # 获取内存使用情况 (RSS: Resident Set Size)
                memory_info = kws_process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                print(f"--- [性能监控] | KWS进程 | CPU: {cpu_usage:.2f}% | 内存: {memory_mb:.2f} MB ---")
            else:
                 # 如果进程不在了，尝试重新获取
                if recorder._kws_process and recorder._kws_process.is_alive():
                    kws_process = psutil.Process(recorder._kws_process.pid)
                else:
                    print("--- [性能监控] | KWS进程未运行 ---")

            # ---- 2. 非阻塞地检查 KWS 耗时/延迟 ----
            message = pubsub.get_message()
            if message and message['type'] == 'message':
                event_data = json.loads(message['data'])
                # 计算从关键词被发现（由KWS进程打上时间戳）到主进程收到消息的时间差
                detection_timestamp = event_data.get("timestamp", 0)
                reception_timestamp = time.time()
                latency_ms = (reception_timestamp - detection_timestamp) * 1000
                
                print("\n" + "="*50)
                print(f"🎯 [端到端延迟测试] 收到关键词: '{event_data.get('keyword')}'")
                print(f"   耗时: {latency_ms:.2f} ms")
                print("="*50 + "\n")
            
            # 主循环的休眠时间
            time.sleep(4) # 减去上面cpu_percent的1秒，大概5秒更新一次

    except KeyboardInterrupt:
        print("\n[主程序] 检测到 Ctrl+C！")
    finally:
        # 确保无论如何都能执行关闭流程
        if pubsub:
            pubsub.close()
        recorder.shutdown()