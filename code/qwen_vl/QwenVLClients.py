import redis
from abc import abstractmethod
import threading
import numpy as np
import torch
import json
import time
import re

from qwen2_5_vl_kou import Qwen2_5_VL, Conversation
from prompt_manager import get_initial_prompt

from typing import List, Dict

# ===================================================================
# A. 辅助函数和依赖 (从你的旧代码中保留)
#    这些函数不属于类的一部分，但被客户端的功能所需要
# ===================================================================

def broadcast_message(redis_conn, wav_path):
    """广播播放音频的消息。"""
    command_data = {"action": "play", "path": wav_path, "key": 1}
    message = json.dumps(command_data)
    try:
        redis_conn.publish("audio:playcommand", message)
        print(f"✅ 成功发送启动音指令到频道 'audio:playcommand'")
    except Exception as e:
        print(f"❌ 发送指令时发生错误: {e}")

def extract_and_parse_json(text: str) -> dict or list or None:
    """
    [修正版] 从LLM返回的文本中提取JSON。
    此版本修复了不支持 `(?R)` 递归写法的正则表达式错误。
    """
    # 1. 优先匹配 ```json ... ``` 代码块，这是最可靠的方式
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 如果代码块内的内容不是有效的JSON，我们继续尝试下面的方法
            pass

    # 2. 如果没有代码块，则寻找第一个 '{' 或 '[' 到最后一个 '}' 或 ']'
    try:
        # 寻找JSON对象
        start_brace = text.find('{')
        end_brace = text.rfind('}')
        
        # 寻找JSON数组
        start_bracket = text.find('[')
        end_bracket = text.rfind(']')

        start_index = -1

        # 判断是对象还是数组先开始
        if start_brace != -1 and start_bracket != -1:
            start_index = min(start_brace, start_bracket)
        elif start_brace != -1:
            start_index = start_brace
        elif start_bracket != -1:
            start_index = start_bracket
        
        # 如果找到了起始符号
        if start_index != -1:
            # 根据起始符号确定结束符号
            if text[start_index] == '{':
                end_index = end_brace
            else:
                end_index = end_bracket
            
            if end_index > start_index:
                json_str = text[start_index : end_index + 1]
                return json.loads(json_str)

    except (json.JSONDecodeError, ValueError):
        # 如果在任何步骤解析失败，则返回None
        return None
    
    return None





# 模板类定义
Payload = dict[str, str]
Filter = Dict[str, List[Payload]] # channel, Payload

# Redis客户端封装类
class RedisClient():
    def __init__(
            self,
            host: str = "localhost",
            port: int = 6379
    ):
        self.host = host
        self.port = port
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)

        self._running = False #在client类里，请用此方法控制循环。在client类外，请显式调用start()和stop()方法管理线程。
        self.thread = None #线程被注册存在标志，并非线程存活标志。用keep_alive()查询存活。
    
    @abstractmethod
    def main_loop(self):
        # 需要手动实现方法
        pass

    @abstractmethod
    def process(self):
        # 需要手动实现方法
        pass

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("请避免重复启用推理循环。")
        self._running = True
        self.thread = threading.Thread(target=self.main_loop, daemon=True)
        self.thread.start()
        
    def stop(self, timeout: int = 5):        
        self._running = False
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError(f"在尝试终止线程{self.thread}时超时。")
        self.thread = None

# Qwen_2_5_VL特化Client类
class Qwen_2_5_VLRedisClient(RedisClient):
    def __init__(
            self,
            model: Qwen2_5_VL,
            lock: threading.RLock,
            host: str = "localhost",
            port: int = 6379,
            subed_launch_filter: Filter = {},
            subed_stop_filter: Filter = {},
            pub_chnls: List[str] = [],
    ):
        '''
            subed_launch_filter: 消息过滤器，用于过滤出那些指导启动的消息。
            subed_stop_filter: 同理，用于过滤出那些指导停止的消息。
            pub_chnls: 订阅频道列表。
        '''
        super().__init__(host=host, port=port)
        self.model = model

        #初始化双过滤器和发布频道列表
        self.subed_launch_filter = subed_launch_filter
        self.subed_stop_filter = subed_stop_filter
        self.pub_chnls = pub_chnls

        #初始化订阅组件
        self.launch_pubsub = self.client.pubsub()
        self.stop_pubsub = self.client.pubsub()

        #发起订阅
        if self.subed_launch_filter:
            self.launch_pubsub.subscribe(*list(self.subed_launch_filter.keys()))
        if self.subed_stop_filter:
            self.stop_pubsub.subscribe(*list(self.subed_stop_filter.keys()))
        
        #中断推理标志
        self._stop_current: bool = False

        #线程锁
        self.lock = lock
    
    @staticmethod
    def _match(payload: Payload, matchers: List[Payload]) -> bool:
        #存在一个全部命中的候选项
        for m in matchers:
            if all(payload.get(k) == v for k, v in m.items()):
                return True
        return False

    def _publish(self, event_type: str, data: Dict[str, str]) -> None:
        #发布函数
        to_publish = {"EventType": event_type, **data}
        msg = json.dumps(to_publish, ensure_ascii=False)
        for ch in self.pub_chnls:
            self.client.publish(ch, msg)
    
    def _check_stop(self) -> bool:
        #这个频道不要塞其它信息，否则降低推理速度
        msg = self.stop_pubsub.get_message(timeout=0)
        while msg is not None:
            if msg.get("type") == "message":
                channel, data = msg.get("channel"), msg.get("data")
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = None#不要发这种无效信息
                if payload is not None:
                    if channel in self.subed_stop_filter and self._match(payload, self.subed_stop_filter[channel]):
                        self._stop_current = True
                        return True
            msg = self.stop_pubsub.get_message(timeout=0)
        return False

    def main_loop(self):
        print(f"[{self.__class__.__name__}] 线程启动，开始监听频道: {list(self.launch_pubsub.channels.keys())}")
        for message in self.launch_pubsub.listen():
            if not self._running:
                break

            try:
                # 在新版本 redis-py 中，channel是 bytes 类型，需要解码
                channel = message['channel']
                payload = json.loads(message['data'])
            except (json.JSONDecodeError, TypeError, KeyError):
                # 忽略无法解析或格式不符的消息
                continue

            # 检查消息是否符合启动规则
            if channel in self.subed_launch_filter and self._match(payload, self.subed_launch_filter[channel]):
                # 在子类中实现具体的处理逻辑
                self.handle_message(payload)

    def handle_message(self, payload: Dict):
        # 这个方法由子类来实现，定义如何处理一个合法的消息
        raise NotImplementedError

    def process(
        self,
        # 将 text 和 image_urls 参数改为 conversation
        conversation: Conversation,
    ) -> None:
        #启动推理
        time_stamp = time.time()
        self._publish(
            "GenerationStarted",
            {"time_stamp": time_stamp}
        )
        self._stop_current = False

        #正式推理
        try:
            with self.lock:
                # 从传入的 conversation 对象中提取图像信息
                image_urls = []
                # 检查 user 的 content 部分
                for item in conversation:
                    if item.get("role") == "user":
                        for content_part in item.get("content", []):
                            if content_part.get("type") == "image":
                                image_urls.append(content_part.get("image"))

                # 直接将 conversation 列表传入底层 preprocess
                inputs = self.model.preprocess([conversation])
                token_len = inputs.input_ids.numel()
                self.model.forward_embed(inputs.input_ids.numpy())
                position_ids = np.tile(np.arange(token_len), 3)
                max_posid = token_len - 1

                if len(image_urls) > 0:
                    vit_token_list = torch.where(inputs.input_ids == self.model.ID_IMAGE_PAD)[1].tolist()
                    self.vit_offset = vit_token_list[0]
                    self.model.vit_process_image(self.vit_offset, inputs.image_grid_thw, inputs.pixel_values)
                    self.vision_embeds = self.model.output_tensors[self.model.name_embed][0].asnumpy()
                    position_ids = self.model.get_rope_index(inputs.input_ids, inputs.image_grid_thw, self.model.ID_IMAGE_PAD)
                    max_posid = int(position_ids.max())
                    position_ids = position_ids.numpy()
                    self.model.update_embeddings(inputs, self.vision_embeds, self.model.ID_IMAGE_PAD)

                #第一次推理
                token = self.model.forward_first(position_ids)
                
                full_word_tokens = []
                text = ""
                while True:
                    if self._stop_current or not self._running:
                        break

                    if self.model.is_end_with_reason(token)[0] or self.model.is_end_with_reason(token)[1]:
                        break
                    
                    full_word_tokens.append(token)
                    word = self.model.tokenizer.decode(full_word_tokens,
                                                skip_special_tokens=True)
                    if "�" not in word:
                        if len(full_word_tokens) == 1:
                            pre_word = word
                            word = self.model.tokenizer.decode(
                                [token, token],
                                skip_special_tokens=True)[len(pre_word):]#分词
                        text += word
                        print(word, flush=True, end="")
                        self._publish(
                            event_type="TokenGenerated",
                            data={
                                "Token": word,
                                "TimeStamp": time_stamp
                            }
                        )
                        full_word_tokens = []
                    max_posid += 1

                    token = self.model.forward_next(max_posid)

                    self._check_stop()
            
            # 收尾判断
            if self._stop_current or not self._running:
                self._publish(
                    "GenerationStopped",
                    {"TimeStamp": time_stamp}
                )
            else:
                self._publish(
                    "GenerationFinished",
                    {"TimeStamp": time_stamp}
                )

        except Exception as e:
            self._publish(
                "GenerationError",
                {"TimeStamp": time_stamp}
            )

# ===================================================================
# C. 各功能客户端的具体实现
#    这是我们需要修改和填充的核心部分
# ===================================================================

class StreamingTextBaseClient(Qwen_2_5_VLRedisClient):
    """
    一个基础客户端，用于处理需要“流式”返回文本的功能。
    (如：图像描述、普通闲聊、RAG问答)
    它统一处理了 "添加System Prompt" 和 "按redisprint格式转发结果" 的逻辑。
    """
    def __init__(self, *args, prompt_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        if not prompt_name:
            raise ValueError("StreamingTextBaseClient 需要一个 prompt_name")
        self.prompt_name = prompt_name
        self._sentence_buffer = "" # 为每个客户端实例创建一个句子缓冲区
        # 定义所有需要断句的标点符号
        self.DELIMITERS = ["。", "！", "？", "…", "；", "!", "?", ",", ";"]


    # def main_loop(self):
    #     """[修正版] 覆写 main_loop 来准备标准的 Conversation 对象。"""
    #     self._sentence_buffer = ""
    #     system_content = get_initial_prompt(self.prompt_name)['content']
        
    #     while self._running:
    #         message = self.launch_pubsub.get_message(timeout=1)
    #         if not message or message.get("type") != "message":
    #             continue
            
    #         try:
    #             payload = json.loads(message.get("data"))
    #         except (json.JSONDecodeError, ValueError):
    #             continue

    #         channel = message.get("channel")
    #         if channel not in self.subed_launch_filter or not self._match(payload, self.subed_launch_filter[channel]):
    #             continue
            
    #         # 从payload中获取用户输入和图片路径
    #         user_text = payload.get("text", "请描述我周围有什么？")
    #         image_path = payload.get("path")

    #         # 构造一个标准的 Conversation 对象列表
    #         conversation_to_process = [
    #             {"role": "system", "content": [{"type": "text", "text": system_content}]},
    #         ]
            
    #         user_content = [{"type": "text", "text": user_text}]
    #         if image_path:
    #             # 底层代码需要的是 'image' 键
    #             user_content.insert(0, {"type": "image", "image": image_path})
            
    #         conversation_to_process.append({"role": "user", "content": user_content})

    #         print(f"[{self.__class__.__name__}] 接收到任务，开始处理...")
    #         # 使用正确的参数 'conversation' 来调用 process 方法
    #         self.process(conversation=conversation_to_process)

    def handle_message(self, payload: Dict):
        self._sentence_buffer = "" # 为新任务重置缓冲区
        system_content = get_initial_prompt(self.prompt_name)['content']

        user_text = payload.get("text", "请描述我周围有什么？")
        image_path = payload.get("path")

        conversation_to_process = [
            {"role": "system", "content": [{"type": "text", "text": system_content}]},
        ]

        user_content = [{"type": "text", "text": user_text}]
        if image_path:
            user_content.insert(0, {"type": "image", "image": image_path})

        conversation_to_process.append({"role": "user", "content": user_content})

        print(f"[{self.__class__.__name__}] 接收到任务，开始处理...")
        # 直接调用 process，它继承自 Qwen_2_5_VLRedisClient
        self.process(conversation=conversation_to_process)

    def _publish(self, event_type: str, data: Dict[str, any]) -> None:
        """
        [核心修改]
        覆写 _publish 方法，实现句子/子句的缓冲和发送。
        """
        if event_type == "TokenGenerated":
            token = data.get("Token", "")
            if not token:
                return

            # 1. 将新生成的token加入缓冲区
            self._sentence_buffer += token

            # 2. 检查缓冲区末尾是否是我们定义的断句符号
            if any(self._sentence_buffer.endswith(p) for p in self.DELIMITERS):
                # 如果是，就发送整个缓冲区的内容
                print(f"[{self.__class__.__name__}] 发送断句: {self._sentence_buffer}")
                message_to_publish = {
                    "content": self._sentence_buffer,
                    "priority": 2
                }
                json_message = json.dumps(message_to_publish, ensure_ascii=False)
                for ch in self.pub_chnls:
                    self.client.publish(ch, json_message)
                
                # 3. 发送后，清空缓冲区
                self._sentence_buffer = ""

        # 当收到结束信号时，检查缓冲区是否还有剩余内容
        elif event_type in ["GenerationFinished", "GenerationStopped", "GenerationError"]:
            # 4. 如果缓冲区里还有没发送完的“半句话”，则立即发送（刷新缓冲区）
            if self._sentence_buffer:
                print(f"[{self.__class__.__name__}] 发送剩余内容: {self._sentence_buffer}")
                message_to_publish = {
                    "content": self._sentence_buffer,
                    "priority": 2
                }
                json_message = json.dumps(message_to_publish, ensure_ascii=False)
                for ch in self.pub_chnls:
                    self.client.publish(ch, json_message)
                self._sentence_buffer = "" # 清空

            # 打印任务结束日志
            if event_type == "GenerationFinished":
                print(f"[{self.__class__.__name__}] 任务处理完成。")
            else:
                print(f"[{self.__class__.__name__}] 任务处理中断或出错。")


class ImageComprehendClient(StreamingTextBaseClient):
    # 这个客户端的功能完全被基类覆盖，只需要在创建时传入正确的prompt_name即可
    pass

class ChatClient(StreamingTextBaseClient):
    # 同上
    pass

class RAGChatClient(StreamingTextBaseClient):
    # 同上
    # 如果未来RAG有更复杂的逻辑，可以在这里覆写 main_loop 或 process
    pass

class NavigationClient(StreamingTextBaseClient):
    """
    这个类继承自 StreamingTextBaseClient，以确保 main_loop 和 process 路径一致。
    唯一的区别是它覆写了 _publish 方法，用于收集token而不是发送它们。
    """
    def __init__(self, *args, **kwargs):
        # 初始化时，强制使用 "navigator" prompt
        # 并且把 pub_chnls 存起来备用
        super().__init__(*args, prompt_name="navigator", **kwargs)
        self.final_pub_chnls = self.pub_chnls
        self.full_response = ""

    # def main_loop(self):
    #     """
    #     每次调用 main_loop (即处理一个新请求) 时，
    #     必须重置 full_response，否则会累加之前的结果。
    #     """
    #     self.full_response = ""
    #     # 调用父类的 main_loop 来执行所有工作
    #     super().main_loop()

    def handle_message(self, payload: Dict):
        self.full_response = "" # 为新任务重置
        # 直接调用父类（StreamingTextBaseClient）的 handle_message 方法
        # 来完成准备数据和调用 self.process 的所有工作
        super().handle_message(payload)

    def _publish(self, event_type: str, data: Dict[str, any]) -> None:
        """
        【核心修改】
        这个 _publish 方法不再向 Redis 发送流式 token。
        它的任务是：
        1. 收集 token 到 self.full_response。
        2. 在收到结束信号时，处理并发送最终的JSON。
        """
        if event_type == "TokenGenerated":
            # 1. 不发送，只拼接
            self.full_response += data.get("Token", "")
        
        # 2. 当收到父类 process 方法发来的结束或错误信号时
        elif event_type in ["GenerationFinished", "GenerationStopped", "GenerationError"]:
            print(f"[{self.__class__.__name__}] LLM 完整回复: {self.full_response}")
            
            # 解析拼接好的完整回复
            parsed_json = extract_and_parse_json(self.full_response)

            if parsed_json:
                json_message = json.dumps(parsed_json, ensure_ascii=False)
                # 使用我们之前存好的频道列表来发布最终结果
                for ch in self.final_pub_chnls:
                    self.client.publish(ch, json_message)
                    print(f"✅ 成功发送导航JSON到频道 '{ch}': {json_message}")
            else:
                print(f"❌ 未能从回复中解析出有效的JSON: {self.full_response}")
            
            # 打印任务结束日志
            if event_type == "GenerationFinished":
                print(f"[{self.__class__.__name__}] 任务处理完成。")
            else:
                print(f"[{self.__class__.__name__}] 任务处理中断或出错。")


if __name__ == "__main__":
    print("正在初始化服务...")

    # 共享的LLM模型实例和线程锁
    lock = threading.RLock()
    # 确保这里的模型路径和配置是正确的
    model = Qwen2_5_VL(
        bmodel_path="/data/QWen_Model/qwen2.5-vl-3b-instruct-awq_w4bf16_seq2048_bm1684x_1dev_20250428_143625.bmodel",
        tokenizer_path="/home/linaro/smart_cane_project/qwenvl25/configs/token_config",
        processor_path="/home/linaro/smart_cane_project/qwenvl25/configs/processor_config",
        config="/home/linaro/smart_cane_project/qwenvl25/configs/config.json",
    )
    redis_connection = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # --- 1. 定义每个客户端的“路由”规则 ---
    
    # 图像描述客户端的配置
    image_client_filter = {
        "event:vision:photo": [{"eventType": "image_capture"}]
    }
    
    # 普通闲聊客户端的配置
    chat_client_filter = {
        "events:asr_result": [{"instruction": "chat"}]
    }

    # RAG问答客户端的配置
    rag_client_filter = {
        "events:asr_result": [{"instruction": "chatrag"}]
    }

    # 导航客户端的配置
    nav_client_filter = {
        "events:asr_result": [{"instruction": "navigation"}]
    }

    stop_filter = {
        "events:kws": [{"type": "kws_detection", "keyword": "stop"}]
    }

    # --- 2. 实例化所有客户端 ---
    
    clients = [
        ImageComprehendClient(
            model=model, lock=lock, prompt_name="image_describer",
            subed_launch_filter=image_client_filter,
            #subed_stop_filter=stop_filter,
            pub_chnls=["events:qwen_reply_result"]
        ),
        ChatClient(
            model=model, lock=lock, prompt_name="blind_assistant",
            subed_launch_filter=chat_client_filter,
            # subed_stop_filter=stop_filter,
            pub_chnls=["events:qwen_reply_result"]
        ),
        RAGChatClient(
            model=model, lock=lock, prompt_name="RAGnotepad",
            subed_launch_filter=rag_client_filter,
            # subed_stop_filter=stop_filter,
            pub_chnls=["events:qwen_reply_result"]
        ),
        NavigationClient(
            model=model, lock=lock,
            subed_launch_filter=nav_client_filter,
            # subed_stop_filter=stop_filter,
            pub_chnls=["events:qwen_navigate_result"]
        )
    ]

    # --- 3. 启动服务 ---
    
    try:
        print("正在启动所有客户端线程...")
        for client in clients:
            client.start()
        
        print("\n========================================")
        print("✅ 所有服务已成功启动，正在监听Redis...")
        print("========================================")

        # 播放启动提示音
        broadcast_message(redis_connection, "/data/preaudio/001.wav")

        # 保持主线程运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n检测到退出指令，正在关闭所有服务...")
        for client in clients:
            client.stop()
        print("所有服务已安全关闭。")
    except Exception as e:
        print(f"\n服务启动或运行过程中发生致命错误: {e}")

#######################