import soundfile
import os
#import speaker
# import argparse  # 删掉 argparse 库
import numpy as np
import math
import sophon.sail as sail
from text import cleaned_text_to_sequence, pinyin_dict
from typing import Iterator, Optional
import time
import logging
import platform
if platform.machine() != 'aarch64':
    from tn.chinese.normalizer import Normalizer
from pypinyin import lazy_pinyin, Style
from pypinyin.core import load_phrases_dict
from bert import TTSProsody
logging.basicConfig(level=logging.INFO)


class VITS:
    def __init__(
        self,
        args,
    ):
        self.net = sail.Engine(args.vits_model, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.vits_model))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.max_length = self.input_shape[1]

        self.tts_front = VITS_PinYin(args.bert_model, args.dev_id, hasBert=True)
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
        self.sample_rate = 16000
        self.stage_factor = 900.0

    def estimate_silence_threshold(self, audio, sample_rate, duration=0.1):
        """
        Estimate the threshold of silence in an audio signal by calculating
        the average energy of the first 'duration' seconds of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            duration: duration (in seconds) of the initial audio to consider for silence.

        Returns:
            The estimated silence threshold.
        """
        # Calculate the number of samples to consider
        num_samples = int(sample_rate * duration)

        # Calculate the energy of the initial segment of the audio
        initial_energy = np.mean(np.abs(audio[-num_samples:]))

        # Return the average energy as the threshold
        return initial_energy


    def remove_silence_from_end(self, audio, sample_rate, threshold=0.005, frame_length=512):
        """
        Removes silence from the end of an audio signal using a specified energy threshold.
        If no threshold is provided, it estimates one based on the initial part of the audio.

        Args:
            audio: numpy array of audio data.
            sample_rate: the sample rate of the audio data.
            threshold: amplitude threshold to consider as silence. If None, will be estimated.
            frame_length: number of samples to consider in each frame.

        Returns:
            The audio signal with end silence removed.
        """
        if threshold is None:
            threshold = self.estimate_silence_threshold(audio, sample_rate)

        # Calculate the energy of audio by frame
        energies = [np.mean(np.abs(audio[i:i+frame_length])) for i in range(0, len(audio), frame_length)]

        # Find the last frame with energy above the threshold
        for i, energy in enumerate(reversed(energies)):
            if energy > threshold:
                last_non_silent_frame = len(energies) - i - 1
                break
        else:
            # In case the whole audio is below the threshold
            return np.array([])

        # Calculate the end index of the last non-silent frame
        end_index = (last_non_silent_frame + 1) * frame_length

        # Return the trimmed audio
        return audio[:end_index]


    def split_text_near_punctuation(self, text, max_length):
        # Define punctuation marks where the text can be split
        punctuation = "。！？，、；：,."
        # Initialize a list to hold the split text segments
        split_texts = []

        # Continue splitting the text until the remaining text is shorter than max_length
        while len(text) > max_length:
            # Assume we need to split at the max_length, then search left for the nearest punctuation
            split_pos = max_length
            # Search left for the nearest punctuation
            while split_pos > 0 and text[split_pos] not in punctuation:
                split_pos -= 1

            # If no punctuation is found to the left, split at the original max_length
            if split_pos == 0:
                split_pos = max_length

            # Split the text and add to the list
            split_texts.append(text[:split_pos + 1])
            # Update the remaining text
            text = text[split_pos + 1:].lstrip()

        # Add the remaining text segment
        split_texts.append(text)
        return split_texts

    def preprocess(self, split_item:list):
        logging.info(split_item)
        phonemes, char_embeds = self.tts_front.chinese_to_phonemes(split_item)
        input_ids = cleaned_text_to_sequence(phonemes)
        char_embeds = np.expand_dims(char_embeds, 0)
        x = np.array(input_ids, dtype=np.int32)
        x = np.expand_dims(x, axis=0) if x.ndim == 1 else x
        if x.shape[1] < self.max_length:
            padding_size = self.max_length - x.shape[1]
            x = np.pad(x, [(0, 0), (0, padding_size)], mode='constant', constant_values=0)
        return x,char_embeds

    def postprocess(self,output_data:dict,outputs:list):
        y_max, y_segment = output_data.values()

        y_segment = y_segment[:math.ceil(y_max[0] / self.stage_factor * len(y_segment) + 1)]
        y_segment = self.remove_silence_from_end(y_segment, self.sample_rate)

        # Collect the output
        outputs.append(y_segment)

        # Concatenate all output segments along the sequence dimension
        y = np.concatenate(outputs, axis=-1)
        return y
    
    def inference(self, x: np.ndarray, char_embeds: np.ndarray):
        # Initialize an empty list to collect output tensors
        outputs = []

        # Extract a sequence of length `self.max_length` from x
        start_time = time.time()
        input_data = {self.input_names[0]: x, self.input_names[1]: char_embeds}
        output_data = self.net.process(self.graph_name, input_data)
        self.inference_time += time.time() - start_time

        start_time = time.time()
        y = self.postprocess(output_data, outputs)
        self.postprocess_time += time.time() - start_time
        return y

    def __call__(self, split_item:list):
        start_time = time.time()
        x,char_embeds = self.preprocess(split_item)
        self.preprocess_time += time.time() - start_time
        y = self.inference(x, char_embeds)
        return y



def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def clean_chinese(text: str):
    text = text.strip()
    text_clean = []
    for char in text:
        if (is_chinese(char)):
            text_clean.append(char)
        else:
            if len(text_clean) > 1 and is_chinese(text_clean[-1]):
                text_clean.append(',')
    text_clean = ''.join(text_clean).strip(',')
    return text_clean


def load_pinyin_dict():
    my_dict={}
    # 确保这个相对路径是正确的
    with open("./python/text/pinyin-local.txt", "r", encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            cuts = line.strip().split()
            hanzi = cuts[0]
            phone = cuts[1:]
            tmp = []
            for p in phone:
                tmp.append([p])
            my_dict[hanzi] = tmp
    load_phrases_dict(my_dict)


class VITS_PinYin:
    def __init__(self, bert_model, dev_id, hasBert=True):
        load_pinyin_dict()
        self.hasBert = hasBert
        if self.hasBert:
            self.prosody = TTSProsody(bert_model, dev_id)
        if platform.machine() != 'aarch64':
            self.normalizer = Normalizer()

    def get_phoneme4pinyin(self, pinyins):
        result = []
        count_phone = []
        for pinyin in pinyins:
            if pinyin[:-1] in pinyin_dict:
                tone = pinyin[-1]
                a = pinyin[:-1]
                a1, a2 = pinyin_dict[a]
                result += [a1, a2 + tone]
                count_phone.append(2)
        return result, count_phone

    def chinese_to_phonemes(self, text):
        if platform.machine() != 'aarch64':
            text = self.normalizer.normalize(text)
        text = clean_chinese(text)
        phonemes = ["sil"]
        chars = ['[PAD]']
        count_phone = []
        count_phone.append(1)
        for subtext in text.split(","):
            if (len(subtext) == 0):
                continue
            pinyins = self.correct_pinyin_tone3(subtext)
            sub_p, sub_c = self.get_phoneme4pinyin(pinyins)
            phonemes.extend(sub_p)
            phonemes.append("sp")
            count_phone.extend(sub_c)
            count_phone.append(1)
            chars.append(subtext)
            chars.append(',')
        phonemes.append("sil")
        count_phone.append(1)
        chars.append('[PAD]')
        chars = "".join(chars)
        char_embeds = None

        if self.hasBert:
            char_embeds = self.prosody.get_char_embeds(chars)
            char_embeds = self.prosody.expand_for_phone(char_embeds, count_phone)
        return " ".join(phonemes), char_embeds

    def correct_pinyin_tone3(self, text):
        pinyin_list = lazy_pinyin(text,
                                    style=Style.TONE3,
                                    strict=False,
                                    neutral_tone_with_five=True,
                                    tone_sandhi=True)
        # , tone_sandhi=True -> 33变调
        return pinyin_list

class Args:
        vits_model = '/data/tts_models/BM1684X/vits_chinese_f16.bmodel'
        bert_model = '/data/tts_models/BM1684X/bert_f16_1core.bmodel'
        dev_id = 0

class TTS:
    def __init__(self):
        # 2. 实例化这个类
        self.arg = Args()
        # ---- 改动结束 ----
        start = time.time()
        self.vits = VITS(self.arg)
        end = time.time()
        self.results_path = "/data/RECOsys_data_cache/TTS_wav/"
        os.makedirs(self.results_path, exist_ok=True)
        # 启动时清理一次旧文件
        self._cleanup_output_folder()

    def _cleanup_output_folder(self):
        """清理输出文件夹内的旧音频文件。"""
        print(f"🧹 (TTS) 正在清理旧的音频文件于: {self.results_path}")
        for filename in os.listdir(self.results_path):
            file_path = os.path.join(self.results_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'❌ (TTS) 无法删除 {file_path}. 原因: {e}')
        print(f"✨ (TTS) 音频输出目录已清空。")

    # 用下面的新方法替换掉旧的 process_text
    def process_text(self, text_to_convert: str) -> Optional[Iterator[str]]:
        """
        [生成器版本]
        接收一个文本字符串，按句号分割，流式地为每句话生成音频，
        并逐个产出(yield)生成的文件路径。
        
        Args:
            text_to_convert (str): 可能包含句号的完整文本。

        Yields:
            str: 每成功生成一个.wav文件，就产出其绝对路径。
        """
        if not text_to_convert:
            print("⚠️ (TTS) 收到空文本，跳过处理。")
            return None # 对于生成器，可以用 return 来表示提前结束

        print(f"🎤 (TTS) 开始流式处理长文本: '{text_to_convert}'")
        
        sentences = text_to_convert.split('。')
        
        # 我们不再需要一个列表来存储所有路径
        # generated_audio_paths = [] 

        for i, sentence in enumerate(sentences):
            cleaned_sentence = sentence.strip()
            if not cleaned_sentence:
                continue

            print(f"   正在处理分句 {i+1}/{len(sentences)}: '{cleaned_sentence}'")
            
            unique_id = f"{int(time.time() * 1000)}_{i}"
            audio_path = os.path.join(self.results_path, f"tts_output_{unique_id}.wav")

            try:
                # VITS核心处理逻辑不变
                split_items = self.vits.split_text_near_punctuation(cleaned_sentence, int(self.vits.max_length / 2 - 5))
                output_audio_segments = []
                for split_item in split_items:
                    output_audio_segments.append(self.vits(split_item))
                final_audio = np.concatenate(output_audio_segments, axis=-1)
                soundfile.write(audio_path, final_audio, self.vits.sample_rate)
                
                print(f"   🎧 分句音频已生成: {audio_path}")

                # =======================================================
                # --- ✨ 核心改动：使用 yield 产出结果 ---
                # =======================================================
                # yield会立刻将 audio_path 返回给调用者，然后暂停在这里
                # 等待调用者请求下一个结果时，再从这里继续执行
                yield audio_path
                # =======================================================

            except Exception as e:
                print(f"❌ (TTS) 处理分句 '{cleaned_sentence}' 时发生错误: {e}")
                continue
        
        print(f"✅ (TTS) 所有分句处理完毕。")
        # 生成器函数在执行完毕后会自动结束，不需要显式 return

# 这是最优雅的地方：因为Python的 for 循环天生就可以处理生成器（迭代器），所以你的消费者代码几乎不需要改动！它会自动地、流式地处理 yield 出来的每一个结果。