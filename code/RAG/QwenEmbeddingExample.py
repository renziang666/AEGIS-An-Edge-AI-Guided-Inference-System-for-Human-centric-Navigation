# QwenEmbedding.py
import numpy as np
from transformers import AutoTokenizer
from QwenBasic import QwenEmbeddingNoKV

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

if __name__ == "__main__":
    # 初始化
    bmodel_path = "models/qwen3-embedding/qwen3-embedding-0.6b_w4bf16_seq512_bm1684x_1dev_20250818_163137.bmodel"
    dev_ids = "0"
    token_path = "models/qwen3-embedding/token_config_embedding"

    tokenizer = AutoTokenizer.from_pretrained(
        token_path,
        padding_side='left',
        trust_remote_code=True
    )
    model = QwenEmbeddingNoKV(
        bmodel_path,
        dev_ids="0"
    )

    # 任务与样例
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity'),
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    # 逐条取向量（单 batch）
    embs = []
    for text in input_texts:
        ids = tokenizer(text, add_special_tokens=True).input_ids
        # 保证不超过图的 SEQLEN（简单右截断）
        max_allowed = model.SEQLEN
        if len(ids) > max_allowed:
            ids = ids[:max_allowed]

        vec = model.forward_once(ids, normalize=True)  # [H]
        embs.append(vec)

    embs = np.stack(embs, axis=0)  # [N, H]
    # 余弦相似度：因为已归一化，可用点积
    q = embs[:2]     # 2 queries
    d = embs[2:]     # 2 docs
    scores = q @ d.T # [2, 2]

    print(scores.tolist())
    # 你应能看到与官方示例相同的模式：对角显著更高，如：
    # [[~0.76, ~0.14],
    #  [~0.13, ~0.60]]
