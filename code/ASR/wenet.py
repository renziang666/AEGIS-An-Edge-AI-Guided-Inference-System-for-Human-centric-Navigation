from __future__ import print_function


import json
import argparse
import copy
import logging
import os
import sys
import subprocess  
arch_output = subprocess.check_output(["arch"])  
arch = arch_output.decode().strip()  
sys.dont_write_bytecode = True
sys.path.append(os.getcwd()+"/swig_decoders_"+arch)

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import Dataset
from utils.common import IGNORE_ID
from utils.file_utils import read_symbol_table
from utils.file_utils import read_lists

import multiprocessing
import numpy as np

sys.path.append(os.getcwd()+"/swig_decoders_"+arch)

from swig_decoders import map_batch,ctc_beam_search_decoder_batch,TrieVector, PathTrie
from utils.sophon_inference import SophonInference

import contextlib
import wave
import time

class WeNet:
    def __init__(self,encoder_path, dict_path , result_file, config_path, dev_id=0, decoder_path=None, ctc_mode='ctc_prefix_beam_search', data_type='raw', decoder_len=350):
        #init
        self.batch_size = 1
        self.subsampling = 4
        self.context = 7
        self.decoding_chunk_size = 16
        self.num_decoding_left_chunks = 5
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path 
        self.dict_path = dict_path
        
        self.result_file = result_file
        self.config_path = config_path
        self.dev_id = dev_id
        self.ctc_mode = ctc_mode
        self.data_type = data_type
        self.decoder_len = decoder_len
        self.vocabulary = []

        self.symbol_table = read_symbol_table(dict_path)

        with open(self.config_path, 'r') as fin:
            self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        # list:test_conf
        test_conf = copy.deepcopy(self.configs['dataset_conf'])
        test_conf['filter_conf']['max_length'] = 102400
        test_conf['filter_conf']['min_length'] = 0
        test_conf['filter_conf']['token_max_length'] = 102400
        test_conf['filter_conf']['token_min_length'] = 0
        test_conf['filter_conf']['max_output_input_ratio'] = 102400
        test_conf['filter_conf']['min_output_input_ratio'] = 0
        test_conf['speed_perturb'] = False
        test_conf['spec_aug'] = False
        test_conf['spec_sub'] = False
        test_conf['spec_trim'] = False
        test_conf['shuffle'] = False
        test_conf['sort'] = False
        test_conf['fbank_conf']['dither'] = 0.0
        test_conf['batch_conf']['batch_type'] = "static"
        test_conf['batch_conf']['batch_size'] = self.batch_size
        self.test_conf = test_conf

    def prepare_dataset(self, voice_input):
        self.voice_input = voice_input

        with open(self.voice_input, 'r') as finput:
            lines = finput.readlines()
            test_audio_num = len(lines)
        
        test_dataset = Dataset(self.data_type,
                           self.voice_input,
                           self.symbol_table,
                           self.test_conf,
                           bpe_model=None,
                           partition=False)
        self.test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    def init_model(self):
        # 加载编码器
        self.encoder = SophonInference(model_path=self.encoder_path, device_id=self.dev_id, input_mode=0)
        self.decoder = None
        # 当模式为'attention_rescoring'且提供了decoder路径时，初始化decoder
        if self.ctc_mode == 'attention_rescoring':
            if not self.decoder_path:
                raise ValueError("Decoder model path must be provided for 'attention_rescoring' mode.")
            self.decoder = SophonInference(model_path=self.decoder_path, device_id=self.dev_id, input_mode=0)

        # Load dict 建立字典的映射关系
        
        char_dict = {}
        with open(self.dict_path, 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                char_dict[int(arr[1])] = arr[0]
                self.vocabulary.append(arr[0])
        self.eos = self.sos = len(char_dict) - 1
        
        self.stride = self.subsampling * self.decoding_chunk_size
        self.decoding_window = (self.decoding_chunk_size - 1) * self.subsampling + self.context        
        self.required_cache_size = self.decoding_chunk_size * self.num_decoding_left_chunks
        
    def _recognize_single_audio(self, feats: np.ndarray, feats_lengths: np.ndarray) -> str:
        """
        这个方法负责处理单条音频特征数据并返回识别结果文本。
        【重大修改】这里将整合所有解码逻辑，包括 Attention Rescoring。
        """
        # 1. Encoder 推理 (流式和非流式共用)
        #    这部分逻辑与你原来的代码类似，但我们需要获取 encoder_out 和 ctc 解码的完整输出
        
        encoder_out = None
        beam_log_probs = None
        beam_log_probs_idx = None
        encoder_out_lens = None

        # 非流式处理
        if len(self.encoder.inputs_shapes) == 2:
            if self.encoder.inputs_shapes[0][1] - feats.shape[1] < 0:
                logging.warning(f"Skipping audio, input length > model input shape: {feats.shape[1]} > {self.encoder.inputs_shapes[0][1]}")
                return ""
            
            speech = np.pad(feats, [(0, 0), (0, self.encoder.inputs_shapes[0][1] - feats.shape[1]), (0, 0)], mode='constant', constant_values=0)
            encoder_input = {"speech": speech, "speech_lengths": feats_lengths}
            out_dict_ = self.encoder.infer_numpy_dict(encoder_input)
            out_dict = {key[:-len("_f32")] if "_f32" in key else key: value for key, value in out_dict_.items()}
            
            encoder_out = out_dict['encoder_out_LayerNormalization']
            encoder_out_lens = out_dict['/ReduceSum_output_0_ReduceSum'].astype(np.int32)
            beam_log_probs = out_dict['beam_log_probs_TopK']
            beam_log_probs_idx = out_dict['beam_log_probs_idx_TopK'].astype(np.int32)
        
        # 流式处理略
        
        # 2. 根据解码模式进行处理
        
        # 如果不是 attention_rescoring，直接用 CTC prefix beam search 解码并返回
        if self.ctc_mode != 'attention_rescoring':
            results, _ = self.ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, self.vocabulary, mode='ctc_prefix_beam_search')
            return results[0]

        # 【关键修改】 Attention Rescoring 流程
        if self.ctc_mode == 'attention_rescoring':
            if self.decoder is None:
                raise RuntimeError("Decoder is not initialized, cannot perform attention rescoring.")

            # (1) 使用 CTC beam search 获取候选列表 (hyps) 和带分数的候选列表 (score_hyps)
            #     注意，这里的模式要传 'attention_rescoring'，这样 ctc_decoding 才会返回 score_hyps
            _, score_hyps = self.ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, self.vocabulary, self.ctc_mode)
            
            # (2) 准备 Decoder 输入 (这部分逻辑直接从你的原始脚本迁移)
            beam_size = beam_log_probs.shape[-1]
            batch_size = beam_log_probs.shape[0]

            ctc_score, all_hyps = [], []
            max_len = 0
            for hyps_list in score_hyps:
                cur_len = len(hyps_list)
                if cur_len < beam_size:
                    hyps_list += (beam_size - cur_len) * [(-float("INF"), (0,))]
                
                cur_ctc_score = []
                for hyp in hyps_list:
                    cur_ctc_score.append(hyp[0])
                    all_hyps.append(list(hyp[1]))
                    if len(hyp[1]) > max_len:
                        max_len = len(hyp[1])
                ctc_score.append(cur_ctc_score)
            ctc_score = np.array(ctc_score, dtype=np.float32)

            # 调整长度以匹配 decoder 的输入要求
            hyps_pad_sos_eos = np.ones((batch_size, beam_size, self.decoder_len), dtype=np.int64) * IGNORE_ID
            r_hyps_pad_sos_eos = np.ones((batch_size, beam_size, self.decoder_len), dtype=np.int64) * IGNORE_ID
            hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
            
            k = 0
            for i in range(batch_size):
                for j in range(beam_size):
                    cand = all_hyps[k]
                    l = len(cand) + 2 # sos + cand + eos
                    if l > self.decoder_len:
                        # 如果候选序列太长，截断它
                        cand = cand[:self.decoder_len - 2]
                        l = self.decoder_len
                    
                    hyps_pad_sos_eos[i, j, 0:l] = [self.sos] + cand + [self.eos]
                    r_hyps_pad_sos_eos[i, j, 0:l] = [self.sos] + cand[::-1] + [self.eos]
                    hyps_lens_sos[i, j] = len(cand) + 1
                    k += 1

            # 填充 encoder_out 以匹配 decoder 的输入长度
            encoder_out = np.pad(encoder_out, [(0, 0), (0, self.decoder_len - encoder_out.shape[1]), (0, 0)], mode='constant', constant_values=0)
            
            hyps_pad_sos_eos = hyps_pad_sos_eos.astype(np.int32)
            r_hyps_pad_sos_eos = r_hyps_pad_sos_eos.astype(np.int32)
            encoder_out_lens_for_decoder = np.full(batch_size, fill_value=encoder_out.shape[1], dtype=np.int32)

            # (3) 调用 Decoder 进行推理
            decoder_input = [encoder_out, encoder_out_lens_for_decoder, hyps_pad_sos_eos, hyps_lens_sos, r_hyps_pad_sos_eos, ctc_score]
            out_dict_ = self.decoder.infer_numpy(decoder_input)
            out_dict = {key[:-len("_f32")] if "_f32" in key else key: value for key, value in out_dict_.items()}

            # (4) 从 Decoder 输出中找到最佳结果
            best_index = out_dict["best_index_ArgMax"].astype(np.int32)
            
            best_sents = []
            k = 0
            for idx in best_index:
                cur_best_sent = all_hyps[k: k + beam_size][idx]
                best_sents.append(cur_best_sent)
                k += beam_size
            
            num_processes = min(multiprocessing.cpu_count(), batch_size)
            final_hyps = map_batch(best_sents, self.vocabulary, num_processes)
            
            return final_hyps[0] # 返回最佳结果

    def speech_recognition(self):
        """
        【改造后的原始函数】
        这个函数现在用于批处理数据集，但它内部会调用新的核心方法。
        它现在除了写入文件，还会返回一个包含所有结果的JSON字符串。
        """
        all_results_data = [] # 新增：用于收集所有结果的列表

        with torch.no_grad(), open(self.result_file, 'a') as fout:
            # 循环遍历数据集保持不变
            for _, batch in enumerate(self.test_data_loader):
                keys, feats, _, feats_lengths, _ = batch
                
                # 注意：这里假设你的dataloader的batch_size=1
                # 如果大于1，你需要在这里加一个循环来处理batch里的每一条数据
                feats_numpy = feats.numpy()
                feats_lengths_numpy = feats_lengths.numpy()
                
                # --- 调用新的核心识别方法 ---
                content = self._recognize_single_audio(feats_numpy, feats_lengths_numpy)
                
                key = keys[0] # 同样假设batch_size=1
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))

                # --- 新增：将结果存入列表 ---
                result_item = {"audio_key": key, "recognized_text": content}
                all_results_data.append(result_item)
        return content

        # --- 新增：在函数末尾，将列表转换为JSON字符串并返回 ---
        # return json.dumps(all_results_data, indent=4, ensure_ascii=False)
   

    """def speech_recognition(self):#暂时
        encoder_inference_time = 0.0
        encoder_infenence_count = 0
        decoder_inference_time = 0.0
        postprocess_time = 0.0

        #recognize config
        output_size = self.configs["encoder_conf"]["output_size"]
        num_layers = self.configs["encoder_conf"]["num_blocks"]
        cnn_module_kernel = self.configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
        head = self.configs["encoder_conf"]["attention_heads"]
        d_k = self.configs["encoder_conf"]["output_size"] // head

        with torch.no_grad(), open(self.result_file, 'a') as fout:# 无需反向传播
            for _, batch in enumerate(self.test_data_loader):# 逐batch批次遍历数据集
                keys, feats, _, feats_lengths, _ = batch
                feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
                if len(self.encoder.inputs_shapes) == 2: # non streaming
                    if self.encoder.inputs_shapes[0][1] - feats.shape[1] < 0:
                        print("Skipping this audio, input feat length exceed bmodel's input shape: feat_length {} > bmodel_input_shape {}".format(feats.shape[1], encoder.inputs_shapes[0][1]))
                        continue
                    # 构建输入数据到pt文件 再输出
                    speech = np.pad(feats, [(0, 0),(0, self.encoder.inputs_shapes[0][1] - feats.shape[1]), (0, 0)], mode='constant', constant_values=0)
                    encoder_input = {"speech": speech, "speech_lengths": feats_lengths}
                    out_dict_ = self.encoder.infer_numpy_dict(encoder_input)#处理数据
                    encoder_infenence_count += 1
                    out_dict = {  
                        key[:-len("_f32")] if "_f32" in key else key: value for key, value in out_dict_.items()  
                    }

                    encoder_out_lens = out_dict['/ReduceSum_output_0_ReduceSum'].astype(np.int32)
                    encoder_out = out_dict['encoder_out_LayerNormalization']
                    beam_log_probs = out_dict['beam_log_probs_TopK']
                    beam_log_probs_idx = out_dict['beam_log_probs_idx_TopK'].astype(np.int32)

                    # ctc decode
                    
                    results, _ = ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, self.vocabulary)
                    result = results[0]
                    
                else:#streaming
                    supplemental_batch_size = self.batch_size - feats.shape[0]
                    
                    att_cache = np.zeros((self.batch_size, num_layers, head, self.required_cache_size, d_k * 2), dtype=np.float32)
                    cnn_cache = np.zeros((self.batch_size, num_layers, output_size, cnn_module_kernel), dtype=np.float32)
                    cache_mask = np.zeros((self.batch_size, 1, self.required_cache_size), dtype=np.float32)
                    offset = np.zeros((self.batch_size, 1), dtype=np.int32)
                    
                    encoder_out = []
                    beam_log_probs = []
                    beam_log_probs_idx = []
                    
                    num_frames = feats.shape[1]
                    result = ""
                    
                    for cur in range(0, num_frames - self.context + 1, self.stride):
                        
                        end = min(cur + self.decoding_window, num_frames)
                        chunk_xs = feats[:, cur:end, :]
                        if chunk_xs.shape[1] < self.decoding_window:
                            chunk_xs = self.adjust_feature_length(chunk_xs, self.decoding_window, padding_value=0)
                            chunk_xs = chunk_xs.astype(np.float32)
                        chunk_lens = np.full(self.batch_size, fill_value=chunk_xs.shape[1], dtype=np.int32)

                        encoder_input = {"chunk_lens": chunk_lens, "att_cache": att_cache, "cnn_cache": cnn_cache, 
                                        "chunk_xs": chunk_xs, "cache_mask": cache_mask, "offset": offset}
                        
                        
                        # 送入pt文件运算
                        out_dict_ = self.encoder.infer_numpy_dict(encoder_input)
                        
                        encoder_infenence_count += 1
                        out_dict = {  
                            key[:-len("_f32")] if "_f32" in key else key: value for key, value in out_dict_.items()  
                        }
                        
                        chunk_log_probs = out_dict["log_probs_TopK"]
                        chunk_log_probs_idx = out_dict["log_probs_idx_TopK"].astype(np.int32)
                        chunk_out = out_dict["chunk_out_LayerNormalization"]
                        chunk_out_lens = out_dict['/Div_output_0_Div_floor'].astype(np.int32)
                        offset = out_dict['r_offset_Unsqueeze'].astype(np.int32)
                        att_cache = out_dict['r_att_cache_Concat']
                        cnn_cache = out_dict['r_cnn_cache_Concat']
                        cache_mask = out_dict['r_cache_mask_Slice']

                        encoder_out.append(chunk_out)
                        beam_log_probs.append(chunk_log_probs)
                        beam_log_probs_idx.append(chunk_log_probs_idx)
                        
                        # ctc decode
                        
                        chunk_hyps, _ = self.ctc_decoding(chunk_log_probs, chunk_log_probs_idx, chunk_out_lens, self.vocabulary)
                        
                        # print(chunk_hyps)
                        result += chunk_hyps[0]
                
                    encoder_out = np.concatenate(encoder_out, axis=1)
                    encoder_out_lens = np.full(self.batch_size, fill_value=encoder_out.shape[1], dtype=np.int32)
                    beam_log_probs = np.concatenate(beam_log_probs, axis=1)
                    beam_log_probs_idx = np.concatenate(beam_log_probs_idx, axis=1)

                for i, key in enumerate(keys):
                    content = None
                    if self.ctc_mode == 'attention_rescoring':
                        content = hyps[i]
                    else:
                        content = result
                    logging.info('{} {}'.format(key, content))
                    fout.write('{} {}\n'.format(key, content))
                
"""

    # tools
    def ctc_decoding(self,beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, mode='ctc_prefix_beam_search'):
        """
        输入​​
        beam_log_probs: 模型输出的CTC概率对数（Tensor，形状为 [batch_size, seq_len, beam_size]）
        beam_log_probs_idx: CTC概率对应的索引（Tensor，形状同 beam_log_probs）
        encoder_out_lens: 编码器输出序列的实际长度（Tensor，形状为 [batch_size]）
        vocabulary: 词汇表（列表，用于将索引映射为字符）
        mode: 解码模式（可选 ctc_greedy_search, ctc_prefix_beam_search, attention_rescoring）

        ​​输出​​
        hyps: 解码后的文本结果（列表，每个元素为字符串）
        score_hyps: 解码结果的得分（列表，包含得分和路径信息）
        """
        beam_size = beam_log_probs.shape[-1]
        batch_size = beam_log_probs.shape[0]
        num_processes = min(multiprocessing.cpu_count(), batch_size)
        hyps = []
        score_hyps = []
        
        if mode == 'ctc_greedy_search':
            if beam_size != 1:
                log_probs_idx = beam_log_probs_idx[:, :, 0]
            batch_sents = []
            for idx, seq in enumerate(log_probs_idx):
                batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
            hyps = map_batch(batch_sents, vocabulary, num_processes,
                            True, 0)
        elif mode in ('ctc_prefix_beam_search', "attention_rescoring"):
            batch_log_probs_seq_list = beam_log_probs.tolist()
            batch_log_probs_idx_list = beam_log_probs_idx.tolist()
            batch_len_list = encoder_out_lens.tolist()
            batch_log_probs_seq = []
            batch_log_probs_ids = []
            batch_start = []  # only effective in streaming deployment
            batch_root = TrieVector()
            root_dict = {}
            for i in range(len(batch_len_list)):
                num_sent = batch_len_list[i]
                batch_log_probs_seq.append(
                    batch_log_probs_seq_list[i][0:num_sent])
                batch_log_probs_ids.append(
                    batch_log_probs_idx_list[i][0:num_sent])
                root_dict[i] = PathTrie()
                batch_root.append(root_dict[i])
                batch_start.append(True)
            score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                    batch_log_probs_ids,
                                                    batch_root,
                                                    batch_start,
                                                    beam_size,
                                                    num_processes,
                                                    0, -2, 0.99999)
            if mode == 'ctc_prefix_beam_search': 
                for cand_hyps in score_hyps:
                    hyps.append(cand_hyps[0][1])
                hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
        return hyps, score_hyps
    
    def adjust_feature_length(self, feats, length, padding_value=0):
        """
        ​输入​​
        feats: 输入特征（NumPy数组，形状为 [batch_size, time_steps, feature_dim]）
        length: 目标时间步长度（整数）
        padding_value: 填充值（默认0）

        输出：
        调整后的特征（NumPy数组，形状为 [batch_size, length, feature_dim]）

        ​（填充/截断）
        """
        if feats.shape[1] < length:
            B, T, L = feats.shape
            tmp = np.full([B, length-T, L], padding_value)
            feats = np.concatenate((feats, tmp), axis=1)
        elif feats.shape[1] > length:
            feats = feats[:, :length, :]
        return feats

    def calculate_total_time(self, data_list):# 后期可以去掉
        """
        ​输入​​
        data_list: 数据列表文件路径（字符串，包含音频文件路径及元数据）
        ​​输出​​
        total_time: 所有音频文件的总时长（浮点数，单位：秒）
        ​​功能​​
        ​​计算音频总时长​​：遍历数据列表中的音频文件，读取帧数和采样率，累加所有音频的时长
        
        """
        lists = read_lists(data_list)
        total_time = 0
        for _, list in enumerate(lists):
            list = eval(list)
            wav_file_path = list["wav"]
            with contextlib.closing(wave.open(wav_file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                total_time += frames / float(rate)
        return total_time


# test if the class is work
if __name__ == "__main__":
    encoder_path="/media/admin/MyUSB/renziang_model/WeNet/model/wenet_encoder_streaming_fp32.bmodel"
    config_path="/home/admin/voice/WeNet_v2/config/train_u2++_conformer.yaml"
    dict_path="/home/admin/voice/WeNet_v2/config/lang_char.txt"
    voice_input="/media/admin/MyUSB/renziang_model/WeNet/dataset/aishell_S0764/aishell_S0764.list"
    decoder_path = "/data/models/wenet_decoder_fp32.bmodel" #! 请替换为你的真实路径
    result_file="./result.txt"
    #start_loaddata = time.time()
    recognizer = WeNet(
        encoder_path=encoder_path,
        decoder_path=decoder_path,  # 传入 decoder 路径
        dict_path=dict_path,
        result_file=result_file,
        config_path=config_path,
        ctc_mode='attention_rescoring' # 设置模式
    )

    
    recognizer.prepare_dataset(voice_input=voice_input)#0.049s
    #start_loader = time.time()
    recognizer.init_model()#0.36s
    #end_loader = time.time()
    #start_recognition = time.time()
    recognizer.speech_recognition()#0.23s/each
    #end_recognition = time.time()
    #print("load_data:",start_loader-start_loaddata)
    #print("load_model:",end_loader-start_loader)
    #print("recognition",end_recognition-start_recognition)

