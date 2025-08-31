import simpleaudio as sa
import threading
import time
import os

class Speaker:
    """
    一个可以在独立线程中播放WAV音频的扬声器类。
    支持启动播放和中途停止播放。
    """
    def __init__(self):
        self.playback_thread = None  # 用于存放播放线程的引用
        self._stop_event = threading.Event()  # 用于通知线程停止的事件标志

    def _play_wav_threaded(self, wav_path: str):
        """
        这个方法在独立的线程中运行，负责实际的音频播放。
        [这是一个私有方法，不应该在类的外部被直接调用] retry机制
        """
        if not os.path.exists(wav_path):
            print(f"[Thread] Error: WAV file not found at {wav_path}")
            return

        self._stop_event.clear()
        
        # --- 新的重试逻辑 ---
        max_retries = 5  # 最多重试5次
        retry_delay = 0.25 # 每次重试前等待0.25秒
        play_obj = None

        for attempt in range(max_retries):
            try:
                # 尝试获取并播放音频
                wave_obj = sa.WaveObject.from_wave_file(wav_path)
                if attempt == 0:
                    print(f"[Thread] Starting playback of: {os.path.basename(wav_path)}")
                else:
                    print(f"[Thread] Retry #{attempt} successful. Starting playback of: {os.path.basename(wav_path)}")
                
                play_obj = wave_obj.play()
                
                # 如果成功开始播放，就跳出重试循环
                break 

            except Exception as e:
                # 只在我们关心的“设备忙”错误时才重试
                error_str = str(e)
                if 'Device or resource busy' in error_str or '-16' in error_str:
                    print(f"[Thread] Attempt {attempt + 1}/{max_retries} failed: Device is busy. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue # 继续下一次循环尝试
                else:
                    # 如果是其他错误（如文件损坏），则直接报错并退出
                    print(f"[Thread] An unrecoverable error occurred during playback startup: {e}")
                    return # 结束这个函数
        
        # 如果经过多次重试后仍然失败
        if not play_obj:
            print(f"[Thread] FATAL: Could not start playback for {os.path.basename(wav_path)} after {max_retries} attempts.")
            return

        # --- 成功播放后的监控循环 (这部分不变) ---
        try:
            while play_obj.is_playing():
                if self._stop_event.is_set():
                    print("[Thread] Stop event received, stopping playback.")
                    play_obj.stop()
                    break
                time.sleep(0.1)
        except Exception as e:
            # 这个try/except是为了捕捉播放过程中的意外错误
            print(f"[Thread] An error occurred while monitoring playback: {e}")
        finally:
            print("[Thread] Playback finished or was stopped.")


    def play_voice(self, wav_path: str):
        """
        公开方法：启动一个新的线程来播放指定的WAV文件。
        如果已有音频在播放，会先停止它，再播放新的。
        """
        # 如果当前有音频正在播放
        if self.is_playing():
            print("❌ Speaker is busy. New play request rejected.")
            return False

        # 创建并启动新的播放线程
        print(f"Requesting to play {os.path.basename(wav_path)} in a new thread.")
        self.playback_thread = threading.Thread(
            target=self._play_wav_threaded, 
            args=(wav_path,)
        )
        self.playback_thread.daemon = True  # 设置为守护线程，主程序退出时线程也会退出
        self.playback_thread.start()

        return True

    def stop_playback(self):
        """
        公开方法：停止当前正在播放的音频。
        这就是你用来“kill”播放线程的方式。
        """
        if self.playback_thread and self.playback_thread.is_alive():
            print("Sending stop signal to playback thread...")
            self._stop_event.set()  # 设置事件，通知线程内部循环停止
            self.playback_thread.join()  # 等待线程真正执行完毕并退出
            print("Playback thread has been stopped.")
        else:
            print("No audio is currently playing.")
    def is_playing(self) -> bool:
        """
        检查当前是否有音频正在播放。
        返回:
            bool: 如果正在播放则为 True，否则为 False。
        """
        return self.playback_thread and self.playback_thread.is_alive()


    

# --- 主程序入口，用于演示如何使用 Speaker 类 ---
if __name__ == "__main__":
    # 请确保你有一个名为 test.wav 的文件，或者修改下面的路径
    # 注意：请使用你自己的WAV文件路径
    wav_file_path = "/home/linaro/renziang_space/voice/test_data/record (1).wav" 

    if not os.path.exists(wav_file_path):
        print("="*50)
        print(f"演示需要一个WAV文件，请将你的文件放在以下路径或修改代码：\n{wav_file_path}")
        print("="*50)
    else:
        speaker = Speaker()

        # 示例1: 正常播放，主线程做其他事情
        print("\n--- 示例1: 启动播放，主线程不阻塞 ---")
        speaker.play_voice(wav_file_path)
        
        print("Main thread: Playback started. I can do other things now.")
        time.sleep(2) # 模拟主线程在忙其他工作
        print("Main thread: Still doing other things...")
        time.sleep(2)

        # 示例2: 播放一个音频，并在中途停止它
        print("\n--- 示例2: 播放一个音频并在3秒后强行停止 ---")
        speaker.play_voice(wav_file_path)
        
        print("Main thread: New playback started.")
        for i in range(1, 4):
            print(f"Main thread: Waiting... {i}s")
            time.sleep(1)

        speaker.stop_playback() # “杀死”播放线程
        print("Main thread: Playback has been successfully stopped by command.")

        print("\n--- Demo finished ---")