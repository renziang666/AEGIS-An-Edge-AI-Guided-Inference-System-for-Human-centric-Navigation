# qwen_nokv.py
import os
import time
import numpy as np
import sophon.sail as sail
from transformers import AutoTokenizer

class QwenNoKV:
    """
    通用无-KV 基类：仅包含 embedding(可选) + block_i(+...)+ (可选 lm_head) 的一次前向。
    子类按需复用 forward_hidden()，在其基础上实现具体的输出（logits/embedding）。
    """
    def __init__(self, bmodel_path, dev_ids):

        self.dev_ids = [int(x) for x in str(dev_ids).split(',')]
        self.handles = {dev: sail.Handle(dev) for dev in self.dev_ids}
        self.target = self.handles[self.dev_ids[0]].get_target()

        if self.target in ["BM1688", "CV186AH"]:
            self.model = sail.EngineLLM(bmodel_path, sail.BmrtFlag.BM_RUNTIME_SHARE_MEM, self.dev_ids)
        else:
            self.model = sail.EngineLLM(bmodel_path, self.dev_ids)

        # 收集网络张量描述
        self.tensors = {}
        self.graph_names = self.model.get_graph_names()
        self.io_alone = 0

        for net in self.graph_names:
            self.tensors[net] = {}
            self.tensors[net]["addr_mode"] = self.model.get_addr_mode(net)
            if self.target in ["BM1688", "CV186AH"]:
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.create_max_input_tensors(net)
                    self.tensors[net]['output'] = self.model.create_max_output_tensors(net)
                else:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)
            else:
                if self.tensors[net]["addr_mode"] == 0:
                    self.tensors[net]['input'] = self.model.get_input_tensors_addrmode0(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors_addrmode0(net)
                else:
                    self.io_alone = 1
                    self.tensors[net]['input'] = self.model.get_input_tensors(net)
                    self.tensors[net]['output'] = self.model.get_output_tensors(net)

        # 层与结构
        self.name_blocks = [n for n in self.graph_names if n.startswith("block_") and not n.startswith("block_cache_")]
        self.name_blocks.sort(key=lambda x: int(x.split('_')[-1]))
        self.NUM_LAYERS = len(self.name_blocks)
        assert self.NUM_LAYERS > 0, "No block_i graphs found in bmodel."

        # 动态形状
        self.is_dynamic = self.model.get_is_dynamic(self.name_blocks[0])

        # shape 基础信息
        _, self.SEQLEN, self.HIDDEN_SIZE = self.tensors[self.name_blocks[0]]["input"][0].shape()

        # 可选子图
        self.name_embed = "embedding" if "embedding" in self.graph_names else None
        self.name_lm = "lm_head" if "lm_head" in self.graph_names else None

        # 无 embedding 子图：使用 embedding.bin
        if self.name_embed is None:
            self.embedding_path = os.path.dirname(bmodel_path) + "/embedding.bin"
            self.hidden_bytes = self.HIDDEN_SIZE * np.dtype(np.uint16).itemsize
            with open(self.embedding_path, "rb") as f:
                self.embedding_content = f.read()

        # 预分配
        self.first_hidden_state = {}
        self.first_pid = {}
        self.first_attention_mask = {}
        if self.name_lm is not None:
            self.lm_input = self.model.create_max_input_tensors(self.name_lm)
            self.lm_output = self.model.create_max_output_tensors(self.name_lm)

        if self.name_embed is not None:
            self.first_embed_input = self.model.create_max_input_tensors(self.name_embed)
            self.first_hidden_state_tensors = self.model.create_max_output_tensors(self.name_embed)
        else:
            self.first_hidden_state_tensors = {}

        for i in range(len(self.dev_ids)):
            # 没有 embedding 子图时，分配 hidden buffer 作为 block_0 的输入
            if self.name_embed is None:
                self.first_hidden_state[i] = sail.Tensor(
                    self.handles[self.dev_ids[i]],
                    self.tensors[self.name_blocks[0]]["input"][0].shape(),
                    self.tensors[self.name_blocks[0]]["input"][0].dtype(),
                    False, True
                )
            self.first_pid[i] = sail.Tensor(self.handles[self.dev_ids[i]],
                                            self.tensors[self.name_blocks[0]]["input"][1].shape(),
                                            self.tensors[self.name_blocks[0]]["input"][1].dtype(),
                                            False, True)
            self.first_attention_mask[i] = sail.Tensor(self.handles[self.dev_ids[i]],
                                                       self.tensors[self.name_blocks[0]]["input"][2].shape(),
                                                       self.tensors[self.name_blocks[0]]["input"][2].dtype(),
                                                       False, True)

        # 注意力 mask 基础值
        self.ATTENTION_MASK = -10000.0
        if self.tensors[self.name_blocks[0]]["input"][2].dtype() == sail.Dtype.BM_BFLOAT16:
            self.ATTENTION_MASK = 50716

        self.token_length = 0

    # ---------- 工具函数 ----------
    def type_convert(self, sail_dtype):
        if sail_dtype == sail.Dtype.BM_FLOAT32: return np.float32
        if sail_dtype == sail.Dtype.BM_FLOAT16: return np.float16
        if sail_dtype == sail.Dtype.BM_INT32:   return np.int32
        if sail_dtype == sail.Dtype.BM_BFLOAT16:return np.uint16

    def _bf16_to_fp32(self, x_uint16):
        # x_uint16: np.ndarray dtype=uint16，表示 bfloat16 的高 16 位
        y = x_uint16.astype(np.uint32) << 16
        return y.view(np.float32)

    def _tensor_to_fp32(self, arr, sail_dtype):
        if sail_dtype == sail.Dtype.BM_BFLOAT16:
            return self._bf16_to_fp32(arr)
        elif sail_dtype == sail.Dtype.BM_FLOAT16:
            return arr.astype(np.float16).astype(np.float32)
        elif sail_dtype == sail.Dtype.BM_FLOAT32:
            return arr.astype(np.float32)
        else:
            # 其他类型不应出现；回退 float32
            return arr.astype(np.float32)

    def load_and_infer_embedding(self, tokens):
        size = len(tokens)
        buffer = np.zeros((size, self.HIDDEN_SIZE), dtype=np.uint16)
        limit = min(size, self.token_length)
        for i in range(limit):
            start = tokens[i] * self.hidden_bytes
            data = self.embedding_content[start:start + self.hidden_bytes]
            if len(data) != self.hidden_bytes:
                raise RuntimeError("embedding.bin read failed")
            buffer[i] = np.frombuffer(data, dtype=np.uint16)
        return buffer

    def _build_inputs(self, length, token_ids):
        """
        生成：embedding/hidden 输入 + position_ids + attention_mask（三角因果）
        """
        # hidden/input_ids
        if self.name_embed is not None:
            input_ids = np.zeros(length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype()))
            input_ids[:len(token_ids)] = token_ids
        else:
            input_ids = np.zeros([length, self.HIDDEN_SIZE],
                                 self.type_convert(self.tensors[self.name_blocks[0]]["input"][0].dtype()))
            input_ids[:len(token_ids)] = self.load_and_infer_embedding(token_ids)

        # position_ids
        position_id = np.zeros(length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][1].dtype()))
        for i in range(self.token_length):
            position_id[i] = i

        # causal attention mask
        attention_mask = np.ones(length*length, self.type_convert(self.tensors[self.name_blocks[0]]["input"][2].dtype())) * self.ATTENTION_MASK
        for i in range(len(token_ids)):
            for j in range(length):
                if j <= i:
                    attention_mask[i*length + j] = 0
        return input_ids, position_id, attention_mask

    # ---------- 核心：一次前向得到 hidden ----------
    def forward_hidden(self, token_ids, *, for_next_token: bool):
        """
        返回： (hidden_tensor, length)
          - hidden_tensor: sail.Tensor，形状 [1, length, H]
          - length: 有效长度（embedding: token_len；reranker: token_len+1 以便拿下一 token）
        """
        self.token_length = len(token_ids)
        if self.is_dynamic:
            length = self.token_length + 1 if for_next_token else self.token_length
            length = max(1, length)
        else:
            length = self.SEQLEN

        input_ids, position_id, attention_mask = self._build_inputs(length, token_ids)

        # embedding（可选）
        if self.name_embed is not None:
            for i in range(len(self.dev_ids)):
                self.tensors[self.name_embed]["input"][i]  = sail.Tensor(self.first_embed_input[i], [1, length], 0)
                self.tensors[self.name_embed]["output"][i] = sail.Tensor(self.first_hidden_state_tensors[i], [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[self.name_embed]["input"][i].update_data(input_ids.reshape(self.tensors[self.name_embed]["input"][i].shape()))
            self.model.process(self.name_embed, self.tensors[self.name_embed]["input"], self.tensors[self.name_embed]["output"])
        else:
            for i in range(len(self.dev_ids)):
                self.first_hidden_state[i].update_data(
                    input_ids.reshape(self.tensors[self.name_blocks[0]]["input"][0].shape()).view(np.uint16)
                )

        # 准备 pid / mask
        for dev_idx in range(len(self.dev_ids)):
            self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 1] = sail.Tensor(self.first_pid[dev_idx], [1, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 2] = sail.Tensor(self.first_attention_mask[dev_idx], [1, 1, length, length], 0)
            self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 1].update_data(position_id.reshape(self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 1].shape()))
            self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 2].update_data(attention_mask.reshape(self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 2].shape()).view(np.uint16))

        # 逐层前向（每层仅 1 个输出：hidden）
        for bi, blk in enumerate(self.name_blocks):
            for dev_idx in range(len(self.dev_ids)):
                hidden_buf = self.first_hidden_state_tensors[dev_idx] if self.name_embed is not None else self.first_hidden_state[dev_idx]
                self.tensors[blk]["input"][3*dev_idx + 0]  = sail.Tensor(hidden_buf, [1, length, self.HIDDEN_SIZE], 0)
                self.tensors[blk]["output"][1*dev_idx + 0] = sail.Tensor(hidden_buf, [1, length, self.HIDDEN_SIZE], 0)
                if bi > 0:
                    self.tensors[blk]["input"][3*dev_idx + 1] = self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 1]
                    self.tensors[blk]["input"][3*dev_idx + 2] = self.tensors[self.name_blocks[0]]["input"][3*dev_idx + 2]
            self.model.process(blk, self.tensors[blk]["input"], self.tensors[blk]["output"])

        # 返回最后的 hidden tensor（驻留于 device）
        hidden_tensor = self.first_hidden_state_tensors[0] if self.name_embed is not None else self.first_hidden_state[0]
        return hidden_tensor, length

    # ---------- 子类可用的 lm_head 帮助 ----------
    def lm_head_logits_from_hidden(self, hidden_tensor, last_index):
        """
        将 [1, L, H] 的 hidden 的最后一个位置送入 lm_head，返回 vocab logits。
        """
        assert self.name_lm is not None, "lm_head graph not found in bmodel."
        self.tensors[self.name_lm]["input"][0]  = sail.Tensor(hidden_tensor, [1, 1, self.HIDDEN_SIZE], last_index * self.HIDDEN_SIZE)
        self.tensors[self.name_lm]["output"][0] = self.lm_output[0]
        self.model.process(self.name_lm, self.tensors[self.name_lm]["input"], self.tensors[self.name_lm]["output"])
        logits = np.squeeze(self.tensors[self.name_lm]["output"][0].asnumpy())
        return logits.astype(np.float64)


class QwenRerankerNoKV(QwenNoKV):
    def forward_once(self, token_ids):
        # 对 reranker：需要“下一 token”logits，所以 for_next_token=True，length = token_len + 1
        hidden_tensor, length = self.forward_hidden(token_ids, for_next_token=True)
        last_index = self.token_length - 1  # 取最后有效 token 的 hidden 送入 lm_head
        logits = self.lm_head_logits_from_hidden(hidden_tensor, last_index)
        return logits  # [vocab_size]


class QwenEmbeddingNoKV(QwenNoKV):
    def forward_once(self, token_ids, normalize=True):
        """
        返回文本 embedding：取最后一个 token 的 hidden（未过 lm_head），可选 L2 归一化。
        """
        # 对 embedding：不需要 next-token trick
        hidden_tensor, length = self.forward_hidden(token_ids, for_next_token=False)

        # 取全序列 hidden 到 CPU
        np_hidden = hidden_tensor.asnumpy()  # dtype: float16 / uint16(bf16) / float32
        # 还原 dtype -> fp32
        np_hidden = self._tensor_to_fp32(np_hidden, hidden_tensor.dtype())

        # shape [1, length, H]，池化为最后一个有效 token：index = token_length - 1
        last_index = max(0, self.token_length - 1)
        vec = np_hidden.reshape(1, -1, self.HIDDEN_SIZE)[0, last_index, :]  # [H]

        if normalize:
            norm = np.linalg.norm(vec) + 1e-12
            vec = vec / norm
        return vec  # [H]
