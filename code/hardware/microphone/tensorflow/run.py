import tensorflow as tf
import numpy as np
import pathlib
import time
# 定义你的模型保存路径
saved_model_path = '/home/linaro/smart_cane_project/hardware/microphone/tensorflow/model2/saved'

# 加载保存的模型
try:
    loaded_model = tf.saved_model.load(saved_model_path)
    print("模型加载成功！")
except Exception as e:
    print(f"加载模型失败: {e}")
    exit()

# 确保 label_names 在当前环境中可用，如果 ExportModel 依赖它
# 假设你在导出模型时 used_labels 是你的所有类别名列表
label_names = ['backward' ,'down' ,'follow' ,'forward' ,'go' ,'left' ,'no' ,'noise' ,'off' ,'on' ,'right' ,'stop' ,'up' ,'yes']


# 假设你的数据目录
#data_dir = pathlib.Path('/home/rrrtd/asr_tensorflow/TensorFlow/dataset/data') # 假设解压后的数据在 ./data 或你的实际路径

# 选择一个要预测的 WAV 文件路径
test_wav_path1 = pathlib.Path('/home/linaro/smart_cane_project/hardware/microphone/tensorflow/nono.wav') # 预测 'no'


# test_wav_path = data_dir/'yes/0a7c2a8d_nohash_0.wav' # 预测 'yes'

if not test_wav_path1.exists():
    print(f"错误: 文件 {test_wav_path} 不存在。请确保数据已下载并解压到正确路径。")
else:
    print(f"\n正在使用文件路径进行预测: {test_wav_path1}")
    # 调用加载的模型，传入文件路径
    start_time = time.time()
    predictions = loaded_model(str(test_wav_path1))
    end_time = time.time()

    # 获取结果
    predicted_logits = predictions['predictions']
    predicted_class_ids = predictions['class_ids'].numpy() # 转换为 NumPy
    predicted_class_names = predictions['class_names'].numpy() # 转换为 NumPy

    # 打印结果
    print(f"原始 logits: {predicted_logits.numpy()}")
    print(f"预测的类别 ID: {predicted_class_ids}")
    print(f"预测的关键词名称: {predicted_class_names}")

    # 转换为概率
    probabilities = tf.nn.softmax(predicted_logits).numpy()
    print(f"各标签的概率: {probabilities}")

    # 找到最高概率的标签
    top_prob_index = np.argmax(probabilities)
    print(f"最高概率的关键词: {label_names[top_prob_index]} (概率: {probabilities[0][top_prob_index]:.4f})")

    print(f"耗时:{end_time - start_time}")