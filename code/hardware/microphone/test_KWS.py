import tensorflow as tf
import numpy as np
import wave
import pathlib
import time

# --- 配置区域 ---
MODEL_PATH = '/home/linaro/smart_cane_project/hardware/microphone/tensorflow/model2/saved'
TEST_WAV_FILE = '/home/linaro/smart_cane_project/hardware/microphone/yes.wav' # 使用你确认可用的文件

# --- 主逻辑 ---

def load_model(model_path):
    """加载 TensorFlow SavedModel。"""
    print(f"🚀 [1/4] 正在从 '{model_path}' 加载模型...")
    if not pathlib.Path(model_path).exists():
        print(f"❌ 错误：模型路径不存在！")
        return None
    try:
        start_time = time.time()
        loaded_model = tf.saved_model.load(model_path)
        end_time = time.time()
        print(f"✅ 模型加载成功！耗时: {end_time - start_time:.2f} 秒。")
        return loaded_model
    except Exception as e:
        print(f"❌ 加载模型时发生错误: {e}")
        return None

def load_wav_to_bytes(wav_path):
    """从WAV文件加载原始音频数据到内存字节串。"""
    print(f"🎧 [2/4] 正在从 '{wav_path}' 加载音频数据到内存...")
    wav_path_obj = pathlib.Path(wav_path)
    if not wav_path_obj.exists():
        print(f"❌ 错误：测试WAV文件不存在！")
        return None, None
    try:
        with wave.open(str(wav_path_obj), 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            params = wf.getparams()
            print(f"✅ 音频加载成功。格式: {params.nchannels}声道, {params.framerate}Hz, {params.sampwidth*8}-bit")
            if params.nchannels != 1 or params.framerate != 16000:
                print("⚠️ 警告：音频文件不是16kHz单声道，预测结果可能不准确！")
            return frames, params
    except Exception as e:
        print(f"❌ 读取WAV文件时发生错误: {e}")
        return None, None

def predict_from_memory(model, audio_bytes):
    """【核心测试函数】直接从内存中的音频字节进行预测。"""
    print("🧠 [3/4] 正在内存中准备数据并执行预测...")
    label_names = ['backward', 'down', 'follow', 'forward', 'go', 'left', 'no', 'noise', 'off', 'on', 'right', 'stop', 'up', 'yes']

    if not audio_bytes or not model:
        print("❌ 无法预测：模型或音频数据为空。")
        return

    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_np.astype(np.float32) / 32768.0

        target_len = 16000
        if len(audio_float) > target_len:
            audio_float = audio_float[:target_len]
        elif len(audio_float) < target_len:
            audio_float = np.pad(audio_float, (0, target_len - len(audio_float)), 'constant')
        
        audio_tensor = tf.constant(audio_float, dtype=tf.float32)

        # --- 核心修复：增加一个“批次”维度 (batch dimension) ---
        # 将 (16000,) 变为 (1, 16000)
        audio_tensor = tf.expand_dims(audio_tensor, 0) 
        # ----------------------------------------------------

        print("    [Info] 数据已转换为Tensor。Shape:", audio_tensor.shape)
        print("    [Info] 正在调用模型进行推理...")
        predictions = model(audio_tensor)
        print("    [Info] 模型推理完成。")

        predicted_logits = predictions['predictions']
        probabilities = tf.nn.softmax(predicted_logits).numpy().flatten()
        predicted_class_id = np.argmax(probabilities)
        predicted_word = label_names[predicted_class_id]
        confidence = probabilities[predicted_class_id]

        print("\n" + "="*40)
        print(f"🎯 [4/4] 预测结果")
        print(f"    - 识别出的词: '{predicted_word}'")
        print(f"    - 置信度: {confidence:.2%}")
        print("="*40 + "\n")
        print("✅ 测试成功：模型可以接受内存中的音频数据！")

    except Exception as e:
        print("\n" + "!"*40)
        print(f"❌ 在内存预测过程中发生严重错误: {e}")
        print("!"*40 + "\n")
        print("🛑 测试失败。")

if __name__ == "__main__":
    print("--- KWS内存直传预测测试 ---")
    
    kws_model = load_model(MODEL_PATH)
    if not kws_model:
        exit()

    wav_bytes, wav_params = load_wav_to_bytes(TEST_WAV_FILE)
    if not wav_bytes:
        exit()

    predict_from_memory(kws_model, wav_bytes)