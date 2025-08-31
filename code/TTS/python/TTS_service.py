import threading
import queue
import json
import redis
import time

from TTS import TTS
from number_converter import convert_sentence_numbers

# --- 1. 配置信息 (已修改) ---
REDIS_HOST = 'localhost'
REDIS_PORT = 6379

# 定义统一的输出频道，所有TTS结果都发到这里
UNIFIED_OUTPUT_CHANNEL = 'audio:playcommand' # 这个频道名与你的 speaker.py 监听的频道完全对应

# 定义不同来源的优先级 (key)
# 视觉信息最重要（如障碍物），设为最高优先级 1
# 导航信息次之，设为优先级 3
# Qwen理解的周围环境，设为优先级 2
PRIORITY_MAP = {
    'channel:vision_to_tts': 2,
    'event:vision:detection':1,
    'events:map:navigate_route': 3,
    'events:qwen_environment_result': 2,
    'events:qwen_reply_result': 2,
    'bus:number:detect':1,
}
# 输入频道就是优先级映射的所有键
INPUT_CHANNELS = list(PRIORITY_MAP.keys())


# --- 2. 生产者线程函数 (无需改变) ---
def redis_listener_producer(redis_conn, channel_name, msg_queue):
    """监听指定的Redis频道，并将收到的消息（包含来源频道）放入共享队列。"""
    p = redis_conn.pubsub(ignore_subscribe_messages=True)
    p.subscribe(channel_name)
    print(f"📡 (Producer) 已订阅频道: '{channel_name}'")

    for message in p.listen():
        try:
            # Redis的pubsub返回的消息中，channel是bytes，需要解码
            source_channel = message['channel'].decode('utf-8')
            print(f"\n📥 (Producer) 从 '{source_channel}' 收到消息，放入队列...")
            task = {
                "source_channel": source_channel,
                "data": message['data']
            }
            msg_queue.put(task)
        except Exception as e:
            print(f"❌ (Producer) 在监听 '{channel_name}' 时发生错误: {e}")


# --- 3. 主线程：初始化与消费 (已修改) ---
if __name__ == "__main__":
    tts_engine = None
    try:
        # --- 初始化 ---
        print("--- 🎬 主线程：初始化服务 ---")
        message_queue = queue.Queue()
        # 注意：这里的 decode_responses=False 很重要，因为生产者线程需要原始的bytes来解码channel名
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        print(f"✅ 成功连接到Redis服务器")
        
        tts_engine = TTS()

        # --- 为每个输入频道创建并启动一个生产者线程 ---
        threads = []
        for channel in INPUT_CHANNELS:
            thread = threading.Thread(
                target=redis_listener_producer,
                args=(r, channel, message_queue),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        print(f"\n--- 👷 主线程：进入消费者模式，已启动 {len(threads)} 个监听线程 ---")

        # --- 消费者主循环 ---
        while True:
            task = message_queue.get()
            
            print("\n" + "="*50)
            print(f"🛍️  (Consumer) 从队列中取出一个任务并开始处理 (来源: {task['source_channel']})")
            
            try:
                # 1. 解析需要转换的文本
                # task['data'] 是bytes，需要解码
                task_data = json.loads(task['data'].decode('utf-8'))
                original_text = task_data.get('content')

                if not original_text:
                    print("⚠️ 任务JSON中缺少 'content' 字段，已忽略。")
                    continue

                print(f"   原始文本: '{original_text}'")
                
                # 数字转换
                text_with_chinese_numbers = convert_sentence_numbers(original_text)
                print(f"   数字转换后: '{text_with_chinese_numbers}'")

                # 2. 调用TTS引擎获取音频路径生成器
                audio_path_generator = tts_engine.process_text(text_with_chinese_numbers)

                if audio_path_generator:
                    print(f"   准备流式发布音频任务到统一频道 '{UNIFIED_OUTPUT_CHANNEL}'...")
                    
                    # 3. 根据来源频道获取优先级
                    # .get(key, default_value) 是一个安全的方式，如果来源频道不在映射中，则默认为最低优先级3
                    priority = PRIORITY_MAP.get(task['source_channel'], 3)
                    print(f"   任务来源: {task['source_channel']} -> 优先级: {priority}")
                    
                    # 4. 循环从生成器中取出每个音频路径并发布
                    for path in audio_path_generator:
                        # 构造符合 speaker.py 期望的JSON格式
                        output_message_dict = {
                            "action": "play",
                            "path": path,
                            "key": priority
                        }
                        
                        output_message_json = json.dumps(output_message_dict)
                        
                        r.publish(UNIFIED_OUTPUT_CHANNEL, output_message_json)
                        print(f"   🚀 (Consumer) 成功将结果 '{output_message_json}' 发布到频道 '{UNIFIED_OUTPUT_CHANNEL}'")

            except json.JSONDecodeError:
                print(f"⚠️ 任务数据不是有效的JSON格式: {task.get('data')}")
            except Exception as e:
                print(f"❌ (Consumer) 处理任务时出错: {e}")
            finally:
                message_queue.task_done()

    except KeyboardInterrupt:
        print("\n🛑 检测到退出信号 (Ctrl+C)...")
    except Exception as e:
        print(f"❌ 主线程发生致命错误: {e}")
    finally:
        print("👋 程序已关闭。")