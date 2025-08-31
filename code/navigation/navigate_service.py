import redis
from online.navigate_new import get_transit_routes,summarize_transit_route
import json
from beijing_offline.subwaySystem import Subway_Navigator



r = redis.Redis(host='localhost', port=6379, db=0)
# 3. 创建 PubSub 对象并订阅频道
pubsub = r.pubsub()
channel_to_subscribe = 'events:qwen_navigate_result'
pubsub.subscribe(channel_to_subscribe)

navigator = Subway_Navigator()
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

                start = event.get("origin")
                end = event.get("destination")
                travel_modes = event.get("travel_modes", [])
                if not start or not end:
                    print("❌ 'start' 或 'end' 参数缺失，无法进行路径规划。")
                    continue
                travel_plan_raw = get_transit_routes(start, end)
                if travel_plan_raw.get("status") == "success":
                    print("✅ 在线路径规划成功，正在生成文本总结...")
                    travel_plan_text = summarize_transit_route(travel_plan_raw)
                    data_to_convert = {
                        "content": travel_plan_text
                    }
                else:
                    # 如果在线规划失败（比如没网），则调用本地备用方案
                    print(f"⚠️ 在线路径规划失败: {travel_plan_raw.get('message')}。切换到本地备用方案...")
                    data_to_convert = navigator.Route_Planning(start, end) # 假设这个函数返回的是你想要的字典
    

                #    使用 ensure_ascii=False 来确保中文字符在JSON中正确显示
                #    使用 indent=4 来格式化输出，使其更易读（可选）
                final_json_output = json.dumps(data_to_convert, ensure_ascii=False, indent=4)

                print(final_json_output)
                publish_channel = 'events:map:navigate_route'
                r.publish(publish_channel, final_json_output)
                print(f"识别结果已发布到频道: {publish_channel}")

            except json.JSONDecodeError:
                print(f"❌ 消息解码失败: 收到的数据不是有效的JSON格式。消息内容: '{message_data}'")
            except Exception as e:
                print(f"❌ 处理消息时发生未知错误: {e}")
    
except KeyboardInterrupt:
    print("\n程序被用户中断。")
finally:
    # 退出前取消订阅
    pubsub.unsubscribe(channel_to_subscribe)
    print(f"已取消订阅频道: {channel_to_subscribe}")
    pubsub.close()
    print("Redis PubSub 连接已关闭。")


