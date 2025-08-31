from streaming_microphone import Microphone


recorder = Microphone(buffer_seconds=6)
print("麦克风模块准备开始，将启动录音...")
recorder.start_listening()

