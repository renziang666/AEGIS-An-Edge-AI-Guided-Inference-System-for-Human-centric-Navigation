import yaml
import numpy as np
from transformers import AutoTokenizer
from QwenBasic import QwenRerankerNoKV
import time
import redis  
import json   

def softmax2(no_logit, yes_logit):
    m = max(no_logit, yes_logit)
    e_no = np.exp(no_logit - m)
    e_yes = np.exp(yes_logit - m)
    return float(e_yes / (e_no + e_yes))

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def broadcast_message(redis_conn, wav_path):
    """
    广播消息到指定的Redis频道。
    
    Args:
        redis_conn: Redis连接实例。
        channel (str): 目标频道名称。
        message (str): 要广播的消息内容。
    """
    command_data = {
        "action": "play",
        "path": wav_path,
        "key": 1
    }
    message = json.dumps(command_data)
    try:
        # 3. 使用 PUBLISH 命令將 JSON 字串發送到指定的頻道
        redis_conn.publish("audio:playcommand", message)
        print(f"✅ 成功发送指令到頻道 'audio:playcommand'")
        print(f"   訊息內容: {message}")
    except Exception as e:
        print(f"❌ 發送指令時發生錯誤: {e}")

# 2. 新增：将Reranker的核心计算逻辑封装成一个函数
def get_rerank_result(query, candidate_docs, qwen_model, tokenizer, prefix_tokens, suffix_tokens, yes_id, no_id):
    """
    接收查询和文档列表，返回得分最高的文档及其分数。
    """
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    pairs = [format_instruction(task, query, doc) for doc in candidate_docs]

    scores = []
    for p in pairs:
        mid_ids = tokenizer.encode(p, add_special_tokens=False)
        token_ids = prefix_tokens + mid_ids + suffix_tokens
        if len(token_ids) > qwen_model.SEQLEN - 1:
            keep = qwen_model.SEQLEN - 1 - len(prefix_tokens) - len(suffix_tokens)
            mid_ids = mid_ids[:keep]
            token_ids = prefix_tokens + mid_ids + suffix_tokens

        logits = qwen_model.forward_once(token_ids)
        logit_no, logit_yes = float(logits[no_id]), float(logits[yes_id])
        prob_yes = softmax2(logit_no, logit_yes)
        scores.append(prob_yes)

    # 找到得分最高的文档
    if not scores:
        return None, 0.0

    best_score = max(scores)
    best_doc_index = scores.index(best_score)
    best_doc = candidate_docs[best_doc_index]

    print(f"Query: '{query}' 的最佳匹配文档是: '{best_doc}' (得分: {best_score:.4f})")
    
    return best_doc, best_score

if __name__ == "__main__":
    # --- 模型和Tokenizer初始化 (保持不变) ---
    token_path = "/data/RAG_formal/models/qwen3-reranker/token_config_reranker"
    bmodel_path = "/data/RAG_formal/models/qwen3-reranker/qwen3-reranker-0.6b_w4bf16_seq512_bm1684x_1dev_20250818_123132.bmodel"
    dev_ids = "0"
    tokenizer = AutoTokenizer.from_pretrained(token_path, trust_remote_code=True)
    qwen = QwenRerankerNoKV(bmodel_path, dev_ids=dev_ids)
    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, "Tokenizer must map 'yes' and 'no' to single tokens"
    yes_id, no_id = yes_ids[0], no_ids[0]

    # --- 静态知识库 (保持不变) ---
    candidate_docs = [
        "李华医生的接诊时间是每周一、周三的上午9点到12点，地点在门诊大楼三楼的眼科诊室。",
        "关于年度体检的预约，请在工作日下午2点到5点之间，致电8857-1234进行电话办理。",
        "王明医生是心脏科专家，他的门诊时间是每周二全天，和每周五的下午。",
        "前往市图书馆的608路大巴车，每天早上7点从始发站发车，每隔30分钟一班，末班车时间是晚上8点。",
        "注意：由于市政道路施工，从明天起，603路公交车的路线将临时改道，不再经过中心医院站。",
        "我把备用钥匙放在了客厅进门处玄关的第一个抽屉里，挨着一本黄色的笔记本。",
        "上次去超市购买了牛奶、面包和鸡蛋，其中牛奶的保质期是到8月26日。",
        "我的好友小张的电话号码是138-1234-5678，他家住在阳光小区的B栋301。",
        "设置一个提醒：明天下午4点需要去社区服务中心领取新的辅助设备。"
    ]

    # 3. 新增：初始化Redis客户端
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ 成功连接到 Redis 服务器。")
    except redis.exceptions.ConnectionError as e:
        print(f"❌ 无法连接到 Redis，请检查配置和服务器状态: {e}")
        exit() # 连接失败则退出程序

    # 4. 新增：创建Redis订阅对象并进入监听循环
    pubsub = redis_client.pubsub()
    pubsub.subscribe('events:asr2rag')
    print("🚀 RAG服务已启动，正在监听 'events:asr2rag' 频道...")

    for message in pubsub.listen():
        if message['type'] != 'message':
            continue

        try:
            # 解析收到的消息
            data = json.loads(message['data'])
            instruction = data.get('instruction')
            query_text = data.get('text')

            if instruction == 'rag_query' and query_text:
                print(f"\n📩 收到RAG请求，查询内容: '{query_text}'")
                
                # 调用核心处理函数
                best_doc, best_score = get_rerank_result(
                    query=query_text, 
                    candidate_docs=candidate_docs, 
                    qwen_model=qwen,
                    tokenizer=tokenizer,
                    prefix_tokens=prefix_tokens,
                    suffix_tokens=suffix_tokens,
                    yes_id=yes_id,
                    no_id=no_id
                )

                # 准备要发布的新消息
                if best_doc and best_score > 0.1: # 增加一个阈值判断，避免返回不相关的结果
                    # 将问题和检索到的资料拼接，送给Qwen进行最终回答
                    final_qwen_input = f"已知信息：'{best_doc}'。请根据这个信息，回答问题：'{query_text}'"
                    
                    response_message = {
                        "instruction": "chatrag", # 将指令改为"chat"，让Qwen主服务进行处理
                        "text": final_qwen_input
                    }
                else:
                    # 如果没有找到相关信息
                    response_message = {
                        "instruction": "chatrag",
                        "text": "抱歉，关于您的问题，我没有找到相关信息。"
                    }

                # 将处理结果发布到 'events:asr_result'
                response_json = json.dumps(response_message, ensure_ascii=False)
                redis_client.publish('events:asr_result', response_json)
                print(f"📤 已将处理结果发布到 'events:asr_result': {response_json}")

            if instruction == 'rag_input' and query_text:
                candidate_docs.append(query_text)
                print(f"\n🧠 新增记忆: '{query_text}'")
                print(f"   当前记忆库共有 {len(candidate_docs)} 条信息。")
                broadcast_message(redis_client,"/data/preaudio/003.wav")



        except json.JSONDecodeError:
            print(f"⚠️ 无法解析收到的消息: {message['data']}")
        except Exception as e:
            print(f"处理消息时发生错误: {e}")