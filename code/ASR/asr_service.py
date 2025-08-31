import redis
import json
import time
from streaming_recoginzer import StreamingRecoginzer

r = redis.Redis(host='localhost', port=6379, db=0)

recognizer = StreamingRecoginzer()
recognizer.Model_Loader()


# 3. 创建 PubSub 对象并订阅频道
pubsub = r.pubsub()
channel_to_subscribe = 'events:audio'
pubsub.subscribe(channel_to_subscribe)
print(f"已订阅 Redis 频道: {channel_to_subscribe}，等待接收新录音通知...")

try:
    # 4. 循环监听消息
    for message in pubsub.listen():
        if message['type'] == 'message':
            try:
                # 消息数据通常是字节，需要解码为字符串
                message_data = message['data'].decode('utf-8')
                # 尝试将消息数据解析为 JSON
                event = json.loads(message_data)

                # 检查消息内容是否符合预期
                if "path" in event:
                    filepath = event["path"]
                    print(f"\n检测到新的录音文件通知: {filepath}")

                    # 5. 进行语音识别
                    print(f"正在识别文件: {filepath}...")
                    # 这里的 key 可以根据你的需求生成，例如使用时间戳或文件路径的一部分
                    asr_content = recognizer.recognizing(key=filepath, wav_path=filepath)
                    print(f"识别结果: {asr_content}")

                    if event.get("type") == "navigation_recording":
                        print("    [Redis监听] 处理导航录音...")
                        # 这里可以添加处理导航录音的逻辑
                        asr_message = { "instruction": "navigation", "text": asr_content }
                        publish_channel = 'events:asr_result'
                        asr_message_json = json.dumps(asr_message, ensure_ascii=False)
                        r.publish(publish_channel, asr_message_json)
                        print(f"识别结果{asr_message_json}已发布到频道: {publish_channel}")

                    elif event.get("type") == "chat_recording":
                        print("    [Redis监听] 处理聊天录音...")
                        # 这里可以添加处理聊天录音的逻辑
                        asr_message = { "instruction": "chat", "text": asr_content }
                        publish_channel = 'events:asr_result'
                        asr_message_json = json.dumps(asr_message, ensure_ascii=False)
                        r.publish(publish_channel, asr_message_json)
                        print(f"识别结果{asr_message_json}已发布到频道: {publish_channel}")

                    elif event.get("type") == "RAG_recording":
                        print("    [Redis监听] 处理聊天录音...")
                        # 这里可以添加处理聊天录音的逻辑
                        asr_message = { "instruction": "rag_query", "text": asr_content }
                        publish_channel = 'events:asr2rag'
                        asr_message_json = json.dumps(asr_message, ensure_ascii=False)
                        r.publish(publish_channel, asr_message_json)
                        print(f"识别结果{asr_message_json}已发布到频道: {publish_channel}")

                    elif event.get("type") == "RAG_input":
                        print("    [Redis监听] 处理聊天录音...")
                        # 这里可以添加处理聊天录音的逻辑
                        asr_message = { "instruction": "rag_input", "text": asr_content }
                        publish_channel = 'events:asr2rag'
                        asr_message_json = json.dumps(asr_message, ensure_ascii=False)
                        r.publish(publish_channel, asr_message_json)
                        print(f"识别结果{asr_message_json}已发布到频道: {publish_channel}")

                    # 6. 发布识别结果到另一个频道
                    
                else:
                    print(f"收到非 'new_recording' 类型或缺少 'path' 的消息: {event}")

            except json.JSONDecodeError:
                print(f"收到非 JSON 格式的消息: {message['data']}")
            except Exception as e:
                print(f"处理消息时发生错误: {e}")
        elif message['type'] == 'subscribe':
            print(f"成功订阅频道: {message['channel'].decode('utf-8')}")
        # 可以添加更多类型（如 'unsubscribe'）的判断，根据需要处理

except KeyboardInterrupt:
    print("\n程序被用户中断。")
finally:
    # 退出前取消订阅
    pubsub.unsubscribe(channel_to_subscribe)
    print(f"已取消订阅频道: {channel_to_subscribe}")
    pubsub.close()
    print("Redis PubSub 连接已关闭。")

