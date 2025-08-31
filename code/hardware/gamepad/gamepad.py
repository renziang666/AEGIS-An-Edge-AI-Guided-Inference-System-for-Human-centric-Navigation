# gamepad_service.py
import evdev
import threading
import time
import redis
import json
import sys

def find_device_by_name(device_name):
    """根据设备名称查找并返回其 /dev/input/eventX 路径。"""
    try:
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for device in devices:
            if device.name == device_name:
                print(f"✅ Found device '{device.name}' at {device.path}")
                return device.path
    except Exception as e:
        print(f"❌ Error listing devices: {e}. Do you have permission?")
        print("   Try running with 'sudo' or add your user to the 'input' group: 'sudo usermod -aG input $USER'")
        sys.exit(1)
        
    print(f"❌ Error: Could not find a device named '{device_name}'.")
    print("   Available devices:")
    for device in devices:
        print(f"     - {device.path}: {device.name}")
    return None

# --- 按钮和轴的映射表 (与之前相同) ---
BUTTON_CODES = {
    'A': 304, 'B': 305, 'X': 308, 'Y': 307,
    'L1': 310, 'R1': 311, 'L2': 312, 'R2': 313,
    'Select': 306, 'Start': 309, 'Home': 316,
}

AXIS_CODES = {
    'LeftX': 0, 'LeftY': 1, 'RightX': 2, 'RightY': 5,
    'Gas': 9, 'Brake': 10, 'DPadX': 16, 'DPadY': 17,
}

# --- 服务类 ---
class GamepadPublisher:
    def __init__(self, device_name, redis_host='localhost', redis_port=6379):
        """
        初始化手柄发布者服务。

        :param device_name: 手柄的设备名称。
        :param redis_host: Redis服务器地址。
        :param redis_port: Redis服务器端口。
        """
        self.device_path = find_device_by_name(device_name)
        self.dev = None
        self.redis_conn = redis.Redis(host=redis_host, port=redis_port)
        self.redis_channel = "gamepad:events" # 定义要发布到的频道
        
        # 反向映射，方便从code查找name
        self._button_map = {code: name for name, code in BUTTON_CODES.items()}
        self._axis_map = {code: name for name, code in AXIS_CODES.items()}
        
        self._read_thread = None
        self._running = False

    def start(self):
        """尝试连接手柄并启动读取和发布线程。"""
        if not self.device_path:
            return False
            
        try:
            self.dev = evdev.InputDevice(self.device_path)
            self._running = True
            self._read_thread = threading.Thread(target=self._read_and_publish, daemon=True)
            self._read_thread.start()
            print(f"✅ Gamepad service started. Publishing events to Redis channel '{self.redis_channel}'.")
            return True
        except Exception as e:
            print(f"❌ Error connecting to gamepad: {e}")
            return False

    def stop(self):
        """停止服务。"""
        print("\nStopping gamepad service...")
        self._running = False
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=1)
        print("✅ Service stopped.")

    def _read_and_publish(self):
        """
        【核心方法】
        在后台线程中运行，读取手柄事件并直接发布到Redis。
        """
        try:
            for event in self.dev.read_loop():
                if not self._running:
                    break # 如果服务被要求停止，则退出循环

                # 将原始事件处理成一个结构化的字典
                message = self._process_event(event)
                
                # 如果事件是我们关心的类型，则发布它
                if message:
                    # 将字典转换为JSON字符串
                    json_message = json.dumps(message)
                    # 发布到Redis
                    self.redis_conn.publish(self.redis_channel, json_message)
                    print(f"Published: {json_message}") # 用于调试，可以注释掉

        except Exception as e:
            if self._running:
                print(f"❌ An error occurred in the reading thread: {e}")
            self._running = False

    def _process_event(self, event):
        """将evdev事件转换为一个结构化的字典消息。"""
        # --- 按钮事件 ---
        if event.type == evdev.ecodes.EV_KEY:
            button_name = self._button_map.get(event.code)
            if button_name:
                return {
                    "type": "button",
                    "name": button_name,
                    "value": event.value, # 1=按下, 0=松开, 2=长按
                    "state": "pressed" if event.value == 1 else "released",
                    "timestamp": time.time()
                }
        
        # --- 轴事件 (摇杆/扳机) ---
        elif event.type == evdev.ecodes.EV_ABS:
            axis_name = self._axis_map.get(event.code)
            if axis_name:
                return {
                    "type": "axis",
                    "name": axis_name,
                    "value": event.value, # 原始的整数值
                    "timestamp": time.time()
                }
        
        # 其他类型的事件（如EV_SYN）我们忽略，返回None
        return None

# --- 使用示例 ---
if __name__ == "__main__":
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
        # 主线程:用于响应你的 Ctrl+C 关停指令
        # 我们让它在这里一直等待，直到用户按下Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # 用户按下Ctrl+C，优雅地停止服务
        gamepad_service.stop()

