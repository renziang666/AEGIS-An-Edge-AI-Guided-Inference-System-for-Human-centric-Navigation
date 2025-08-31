# 这是测试代码

import redis
import logging
logging.basicConfig(level=logging.DEBUG)
import time
import json

#sub_channels = ["events:qwen_reply_result", "event:vision:photo", "events:kws"]#订阅频道，同时订阅qwen和自身
sub_channels = ["events:asr_result", "events:asr2rag"]
# pub_channels = ["event:vision:photo"]
pub_channels = ["A"]
r = redis.Redis(host="localhost", port=6379, decode_responses=True)
p = r.pubsub()
logging.debug("已建立客户端。")
p.subscribe(*sub_channels)
logging.debug(f"已订阅频道{sub_channels}。")

launch_msg = json.dumps({"eventType": "image_capture"})
stop_msg = json.dumps({
    "events:kws": [{"type": "kws_detection", "keyword": "stop"}]
})

# 发送启动推理消息
for channel in pub_channels:
    r.publish(channel, launch_msg)
    logging.debug(f"已发送消息{launch_msg}到频道{channel}。")

while True:
    msg = p.get_message()
    if msg and msg.get("type") == "message":
        channel = msg.get("channel")
        data = msg.get("data")
        logging.info(f"从频道{channel}收到新消息\n{data}。")
    time.sleep(0.5)
