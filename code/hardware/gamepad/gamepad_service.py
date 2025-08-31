from gamepad import GamepadPublisher
import time
import redis
import json
import sys


TARGET_GAMEPAD_NAME = "SHANWAN Android Gamepad" # 请确保这个名称和你的设备完全一致
    
# 1. 创建服务实例
gamepad_service = GamepadPublisher(device_name=TARGET_GAMEPAD_NAME)
# 2. 启动服务
if not gamepad_service.start():
    print("Failed to start gamepad service. Exiting.")
    sys.exit(1)

print("\nService is running. Press buttons and move sticks on your gamepad.")
print("Press Ctrl+C to stop the service.")

try:
    # 主线程现在可以什么都不做，或者做其他事情
    # 我们让它在这里一直等待，直到用户按下Ctrl+C
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # 用户按下Ctrl+C，优雅地停止服务
    gamepad_service.stop()