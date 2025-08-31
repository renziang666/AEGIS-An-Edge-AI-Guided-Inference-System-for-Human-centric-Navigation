import yaml
import numpy as np
from transformers import AutoTokenizer
from QwenBasic import QwenRerankerNoKV
import time

def softmax2(no_logit, yes_logit):
    m = max(no_logit, yes_logit)
    e_no = np.exp(no_logit - m)
    e_yes = np.exp(yes_logit - m)
    return float(e_yes / (e_no + e_yes))

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

if __name__ == "__main__":
    token_path = "/data/RAG_formal/models/qwen3-reranker/token_config_reranker"
    bmodel_path = "/data/RAG_formal/models/qwen3-reranker/qwen3-reranker-0.6b_w4bf16_seq512_bm1684x_1dev_20250818_123132.bmodel"
    dev_ids = "0"

    tokenizer = AutoTokenizer.from_pretrained(
        token_path,
        trust_remote_code=True
    )
    qwen = QwenRerankerNoKV(
        bmodel_path,
        dev_ids=dev_ids
    )

    # prompt prefix/suffix
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    # label token ids
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1, "Tokenizer must map 'yes' and 'no' to single tokens"
    yes_id, no_id = yes_ids[0], no_ids[0]

    # === 测试数据 ===
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    query = "如何联系到小张"

    candidate_docs = [
        # 医疗相关 (包含具体时间，用于精准查询)
        "李华医生的接诊时间是每周一、周三的上午9点到12点，地点在门诊大楼三楼的眼科诊室。",
        "关于年度体检的预约，请在工作日下午2点到5点之间，致电8857-1234进行电话办理。",
        "王明医生是心脏科专家，他的门诊时间是每周二全天，和每周五的下午。",

        # 出行交通相关 (包含具体时刻和路线)
        "前往市图书馆的608路大巴车，每天早上7点从始发站发车，每隔30分钟一班，末班车时间是晚上8点。",
        "注意：由于市政道路施工，从明天起，603路公交车的路线将临时改道，不再经过中心医院站。",
        "地铁4号线的运营时间为早上6点至晚上11点，其中，人民公园站是重要的换乘站。",

        # 日常生活与个人信息
        "我把备用钥匙放在了客厅进门处玄关的第一个抽屉里，挨着一本黄色的笔记本。",
        "上次去超市购买了牛奶、面包和鸡蛋，其中牛奶的保质期是到8月26日。",
        "我的好友小张的电话号码是138-1234-5678，他家住在阳光小区的B栋301。",
        "设置一个提醒：明天下午4点需要去社区服务中心领取新的辅助设备。"
    ]

    pairs = [format_instruction(task, query, doc) for doc in candidate_docs]

    scores = []
    for doc, p in zip(candidate_docs, pairs):
        # 构造 token 序列
        mid_ids = tokenizer.encode(p, add_special_tokens=False)
        token_ids = prefix_tokens + mid_ids + suffix_tokens
        if len(token_ids) > qwen.SEQLEN - 1:
            keep = qwen.SEQLEN - 1 - len(prefix_tokens) - len(suffix_tokens)
            mid_ids = mid_ids[:keep]
            token_ids = prefix_tokens + mid_ids + suffix_tokens

        # 模型前向
        start_time = time.time()
        logits = qwen.forward_once(token_ids)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"nokv代码运行时间: {run_time} 秒")

        logit_no, logit_yes = float(logits[no_id]), float(logits[yes_id])
        prob_yes = softmax2(logit_no, logit_yes)
        scores.append(prob_yes)
        print(f"Doc: {doc}\n  score(prob_yes): {prob_yes:.4f}\n")

    print("All scores:", scores)