import numpy as np
from typing import List, Callable, Optional, Tuple
from transformers import AutoTokenizer
from QwenBasic import QwenRerankerNoKV


class QwenRerankerRAG:
    def __init__(
        self,
        token_path: str,
        bmodel_path: str,
        dev_ids: str = "0",
        instruction: Optional[str] = None
    ):
        """
        Qwen reranker 封装，用于 RAG 文本重排序

        Args:
            token_path (str): tokenizer 路径
            bmodel_path (str): bmodel 模型路径
            dev_ids (str): 使用的设备 id
            instruction (Optional[str]): 检索任务的默认指令
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            token_path,
            trust_remote_code=True
        )
        self.qwen = QwenRerankerNoKV(
            bmodel_path,
            dev_ids=dev_ids
        )

        # 默认检索任务指令
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )

        # prompt 模板
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        # yes/no token id
        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1, "Tokenizer must map 'yes' and 'no' to single tokens"
        self.yes_id, self.no_id = yes_ids[0], no_ids[0]

    @staticmethod
    def _softmax2(no_logit: float, yes_logit: float) -> float:
        m = max(no_logit, yes_logit)
        e_no = np.exp(no_logit - m)
        e_yes = np.exp(yes_logit - m)
        return float(e_yes / (e_no + e_yes))

    def _format_instruction(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"

    def __call__(
        self,
        query: str,
        docs: List[str],
        callback: Optional[Callable[[Tuple[int, np.float32]], None]] = None
    ) -> List[np.float32]:
        """
        对一个查询和多个文档进行打分

        Args:
            query (str): 查询
            docs (List[str]): 候选文档
            callback (Callable[[Tuple[int, np.float32]], None], optional): 回调函数，
                每次算出一个分数后会回传 (文档索引, 分数)

        Returns:
            List[np.float32]: 每个文档的相关性分数 (越大越相关)
        """
        scores: List[np.float32] = []

        for i, doc in enumerate(docs):
            # 构造 token 序列
            mid_ids = self.tokenizer.encode(
                self._format_instruction(query, doc),
                add_special_tokens=False
            )
            token_ids = self.prefix_tokens + mid_ids + self.suffix_tokens

            if len(token_ids) > self.qwen.SEQLEN - 1:
                keep = self.qwen.SEQLEN - 1 - len(self.prefix_tokens) - len(self.suffix_tokens)
                mid_ids = mid_ids[:keep]
                token_ids = self.prefix_tokens + mid_ids + self.suffix_tokens

            # 模型前向
            logits = self.qwen.forward_once(token_ids)
            logit_no, logit_yes = float(logits[self.no_id]), float(logits[self.yes_id])
            prob_yes: np.float32 = np.float32(self._softmax2(logit_no, logit_yes))

            scores.append(prob_yes)

            if callback is not None:
                callback((i, prob_yes))

        return scores

class QwenEmbeddingRAG:
    def __init__(
        self,
        token_path: str,
        bmodel_path: str,
        dev_ids: str = "0"
    ):
        """
        Qwen embedding 封装，用于 RAG 向量化

        Args:
            token_path (str): tokenizer 路径
            bmodel_path (str): bmodel 模型路径
            dev_ids (str): 使用的设备 id
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            token_path,
            padding_side="left",
            trust_remote_code=True
        )
        self.model = QwenEmbeddingNoKV(
            bmodel_path,
            dev_ids=dev_ids
        )

    def __call__(
        self,
        input: List[str],
        callback: Optional[Callable[[Tuple[int, np.ndarray]], None]] = None
    ) -> List[np.ndarray]:
        """
        计算输入文本的 embedding 向量

        Args:
            input (List[str]): 待计算的文本列表
            callback (Callable[[Tuple[int, np.ndarray]], None], optional): 回调函数，
                每算出一个向量后会回传 (索引, 向量)

        Returns:
            List[np.ndarray]: 每个文本对应的向量，形状 [H]，float32
        """
        embeddings: List[np.ndarray] = []

        for i, text in enumerate(input):
            ids = self.tokenizer(text, add_special_tokens=True).input_ids
            # 截断，避免超过图的 SEQLEN
            if len(ids) > self.model.SEQLEN:
                ids = ids[:self.model.SEQLEN]

            vec: np.ndarray = self.model.forward_once(ids, normalize=True)  # shape [H]
            vec = vec.astype(np.float32)  # 确保类型一致
            embeddings.append(vec)

            if callback is not None:
                callback((i, vec))

        return embeddings