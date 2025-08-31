from wenet import WeNet
import os
import json
import time


class StreamingRecoginzer:
    def __init__(self):
        encoder_path="../models/wenet_encoder_non_streaming_fp32.bmodel"
        decoder_path = "../models/wenet_decoder_fp32.bmodel"
        config_path="./config/train_u2++_conformer.yaml"
        dict_path="./config/lang_char.txt"
        result_file="./result.txt"
        self.recognizer = WeNet(
            encoder_path=encoder_path,
            decoder_path=decoder_path,  # 传入 decoder 路径
            dict_path=dict_path,
            result_file=result_file,
            config_path=config_path,
            ctc_mode='attention_rescoring' # 设置模式
        )
        self.voice_list="./voice_wav.list"

    def Model_Loader(self):
        self.recognizer.init_model()#0.36s
    
    def recognizing(self, key, wav_path):
        self.add_entry(key=key, wav_path=wav_path, list_file=self.voice_list)
        self.recognizer.prepare_dataset(voice_input=self.voice_list)#0.049s
        content = self.recognizer.speech_recognition()
        return content

    #tools
    def add_entry(self, key: str, wav_path: str, list_file: str) -> None:
        """
        向 .list 文件追加一行 JSON 格式的条目
        
        :param key: 条目ID（例如 "001"）
        :param wav_path: wav文件的绝对路径（例如 "/path/to/file.wav"）
        :param list_file: 目标 .list 文件路径（例如 "audio.list"）
        """
        entry = {
            "key": key,
            "wav": os.path.abspath(wav_path),  # 确保路径标准化
            "txt": " "
        }
        
        # 检查文件是否存在并追加写入（不存在则创建）
        with open(list_file, 'w', encoding='utf-8') as f:#自动清空之前list内容
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__=="__main__":

    wav_file = "/data/Zoo2/Radxa-Model-Zoo/sample/WeNet/datasets/aishell_S0764/BAC009S0764W0123.wav"
    recognizer = StreamingRecoginzer()

    # --- 测量模型加载 ---
    start_load = time.perf_counter()
    recognizer.Model_Loader()
    end_load = time.perf_counter()
    load_time = (end_load - start_load) * 1000  # 转换为毫秒

    # --- 测量推理 ---
    start_recognize = time.perf_counter()
    asr_content = recognizer.recognizing(key="001", wav_path=wav_file)
    end_recognize = time.perf_counter()
    recognize_time = (end_recognize - start_recognize) * 1000  # 转换为毫秒

    print(f"模型加载耗时: {load_time:.2f} ms")
    print(f"语音识别耗时: {recognize_time:.2f} ms")
    print(f"识别结果: {asr_content}")

# if __name__ == "__main__":
#     # --- 配置你的测试数据 ---
#     # 你可以添加更多的音频文件和对应的期望文本来增加测试的全面性
#     # 为了准确率测试，请确保期望文本中没有标点和空格
#     test_data = [
#         {"key": "001", "wav_path": "/data/RECOsys_data_cache/Microphone/_(2).wav", "expected_text": "尽管人们普遍认为人工智能将在未来取代大量人工劳动力但也有观点认为它更应该被视为一种强大的辅助工具能够显著提升工作效率和创造力从而使人类能够专注于更具创造性和战略性的任务而不是重复性的机械劳动"},
#         # {"key": "002", "wav_path": "/path/to/your_second_audio.wav", "expected_text": "这是第二段测试文本"},
#         # {"key": "003", "wav_path": "/path/to/your_third_audio.wav", "expected_text": "第三段文本包含一些数字和英文名称例如WeNet和Google"},
#     ]

#     recognizer = StreamingRecoginzer()

#     # --- 测量模型加载耗时 ---
#     print("开始加载模型...")
#     start_load = time.perf_counter()
#     recognizer.Model_Loader()
#     end_load = time.perf_counter()
#     load_time = (end_load - start_load) * 1000
#     print(f"模型加载耗时: {load_time:.2f} ms\n")

#     # --- 循环进行语音识别测试 ---
#     total_recognize_time = 0
#     correct_count = 0
#     total_count = len(test_data)

#     print("--- 开始批量识别测试 ---\n")
#     for data in test_data:
#         key = data["key"]
#         wav_file = data["wav_path"]
#         expected_text = data["expected_text"]

#         if not os.path.exists(wav_file):
#             print(f"警告：文件 {wav_file} 不存在，跳过该测试。")
#             continue
        
#         print(f"--- 正在识别文件: {wav_file} ---")
#         start_recognize = time.perf_counter()
#         try:
#             asr_content = recognizer.recognizing(key=key, wav_path=wav_file)
#         except Exception as e:
#             print(f"识别 {wav_file} 时发生错误：{e}")
#             continue
            
#         end_recognize = time.perf_counter()
#         recognize_time = (end_recognize - start_recognize) * 1000
#         total_recognize_time += recognize_time

#         print(f"语音识别耗时: {recognize_time:.2f} ms")
#         print(f"期望结果: {expected_text}")
#         print(f"识别结果: {asr_content}")

#         # 检查准确率（忽略标点和空格）
#         # 你的识别结果可能也包含标点，所以这里对两者都进行处理
#         cleaned_asr_content = asr_content.replace("，", "").replace("。", "").replace("、", "").replace("！", "").replace("？", "").replace(" ", "").replace("…", "")
#         if cleaned_asr_content == expected_text:
#             print("识别准确性: 准确 ✓")
#             correct_count += 1
#         else:
#             print("识别准确性: 错误 ✗")

#         print("-" * 20)

#     # --- 打印汇总报告 ---
#     print("\n" + "="*40)
#     print("             测试报告")
#     print("="*40)
#     if total_count > 0:
#         average_time = total_recognize_time / total_count
#         accuracy = (correct_count / total_count) * 100
#         print(f"总测试文件数: {total_count}")
#         print(f"平均识别耗时: {average_time:.2f} ms")
#         print(f"识别准确率: {accuracy:.2f}% ({correct_count}/{total_count})")
#     else:
#         print("没有可测试的文件。")
#     print("="*40)

