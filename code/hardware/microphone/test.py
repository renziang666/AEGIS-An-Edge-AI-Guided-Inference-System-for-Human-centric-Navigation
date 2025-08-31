import pyaudio

p = pyaudio.PyAudio()

print("==========================================================")
print("           PyAudio 音频设备信息查询工具           ")
print("==========================================================")

# 获取设备总数
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

print(f"\n查询到 {numdevices} 个音频设备：\n")

# 遍历所有设备
for i in range(0, numdevices):
    device_info = p.get_device_info_by_host_api_device_index(0, i)

    # 我们只关心输入设备 (麦克风)
    if device_info.get('maxInputChannels') > 0:
        print(f"--- 输入设备索引 (Index): {i} ---")
        print(f"  名称 (Name): {device_info.get('name')}")
        print(f"  最大输入声道数 (Max Input Channels): {device_info.get('maxInputChannels')}")
        print(f"  默认采样率 (Default Sample Rate): {device_info.get('defaultSampleRate')}")
        print("-" * 40)

print("\n查询完毕。\n")
p.terminate()