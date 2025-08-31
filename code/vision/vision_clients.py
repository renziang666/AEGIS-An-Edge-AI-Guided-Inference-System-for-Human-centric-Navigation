# vision_clients.py

import redis
import threading
import cv2
import json
import logging
from abc import abstractmethod
from basic_detects import VisualDetect, VisualRecognize, StairDetect, YoloSegDetect
from text_generate import generate_warning, find_number, generate_locations, generate_stair_warning, generate_pavement_location
import time
from collections import Counter
import uuid
from datetime import datetime
import os

# Redis客户端封装类
class RedisClient():
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379
    ):
        self.host = host
        self.port = port
        self.client = redis.Redis(host=self.host, port=self.port, decode_responses=True)

        self._running = False #在client类里，请用此方法控制循环。在client类外，请显式调用start()和stop()方法管理线程。
        self.thread = None #线程被注册存在标志，并非线程存活标志。用keep_alive()查询存活。
    
    @abstractmethod
    def main_loop(self):
        # 需要手动实现方法
        pass

    @abstractmethod
    def process(self):
        # 需要手动实现方法
        pass

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("请避免重复启用推理循环。")
        self._running = True
        self.thread = threading.Thread(target=self.main_loop, daemon=True)
        self.thread.start()
        
    def stop(self, timeout: int = 5):        
        self._running = False
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            raise TimeoutError(f"在尝试终止线程{self.thread}时超时。")
        self.thread = None

# 多线程视觉协同类
class FrameGrabber:
    def __init__(
        self,
        camara_id: int = 0,
        width: int = 1920,
        height: int = 1080
    ):
        self.camara_id = camara_id
        self.capture = cv2.VideoCapture(self.camara_id)
        if not self.capture.isOpened():
            raise RuntimeError(f"摄像头{camara_id}没有成功打开。")
        
        self.width = width
        self.height = height
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.width)

        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"尝试设置分辨率为 {width}x{height}，实际分辨率为 {actual_width}x{actual_height}")
        
        self.frame = None
        self._running = True
        self.lock = threading.Lock() #线程锁
        self.thread = threading.Thread(target=self.update_frame, daemon=True)
        self.thread.start()#直接开启
    
    def update_frame(self):
        while self._running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock: #阻塞当前线程，直至线程锁被持有
                    self.frame = frame
            else: logging.info(f"无法从摄像头{self.camara_id}中读取和更新图像帧。")
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self._running = False
        self.thread.join()
        self.capture.release()

# 继承RedisClient，形成OCR类，为文字推理类。 线路号
class OCRClient(RedisClient):
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        host: str = "localhost",
        port: int = 6379,
    ):
        super().__init__(host=host, port=port)

        # 请在此修改各项参数  result = 111
        self.publish = {"bus:number:detect": {"eventType": "bus_number_detected"}}
        self.subscribe = {"gamepad:events": {"name": "L2", "state": "pressed"}}

        self.pubsub = self.client.pubsub()
        self.pubsub.subscribe(*list(self.subscribe.keys()))
        self.listen = self.pubsub.listen #迭代器

        self.frame_grabber = frame_grabber
        self.visual_detect = VisualDetect(
            bmodel_path="/data/sophon-demo/sample/YOLOv8_plus_det/models/test/yolov8s_fp16_1b_number_new.bmodel",
            dev_id=0,
            use_resize_padding=True
        )
        self.visual_recognize = VisualRecognize(
            bmodel_rec="/data/sophon-demo/sample/PP-OCR/models/BM1684X/ch_PP-OCRv4_rec_fp16.bmodel",
            char_dict_path="/data/sophon-demo/sample/PP-OCR/datasets/ppocr_keys_v1.txt",
            dev_id=0
        )

    def main_loop(self):
        print("OCRClient: 线程已启动，等待游戏手柄触发...")
        while self._running:
            message = self.pubsub.get_message(timeout=1)
            if message is None:
                continue

            if message["type"] != "message":
                continue

            channel = message["channel"]
            data = message["data"]
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue

            # 遍历订阅字典，查看消息是否命中消息
            for subscribed_channel, subscribed_filter in self.subscribe.items():
                if channel != subscribed_channel:
                    continue

                if all(payload.get(k) == v for k, v in subscribed_filter.items()):
                    print("✅ L2键按下，开始连续识别5次...")
                    recognition_results = []
                    for i in range(5):
                        single_result_dict = self.process()
                        if single_result_dict and single_result_dict.get("result"):
                            # 如果有效，将识别出的数字字符串存入列表
                            recognized_number = single_result_dict["result"]
                            recognition_results.append(recognized_number)
                            print(f"  [第 {i+1}/5 次] 成功: {recognized_number}")
                        else:
                            print(f"  [第 {i+1}/5 次] 失败或未识别到有效号码。")

                        time.sleep(0.1)
                    if not recognition_results:
                        print("❌ 5次识别均未成功，本次任务结束。")
                        continue # 跳过发布环节，继续等待下一次按键
                    final_result_str = Counter(recognition_results).most_common(1)[0][0]
                    print(f"🗳️ 投票完成！最终识别结果为: {final_result_str}")

                    # 6. 使用最终结果构建要发布的消息
                    content_str = f"目前来的是{final_result_str}路公交车"
                    final_result_dict = {
                        "result": final_result_str,
                        "content": content_str
                    }
                    

                    for channel_topublish, filter_topublish in self.publish.items():
                        to_publish = {**filter_topublish, **final_result_dict}
                        self.client.publish(channel_topublish, json.dumps(to_publish))
                        print(f"🚀 已将最终结果发布到频道: {channel_topublish}")

            
        self.pubsub.unsubscribe()
        self.pubsub.close()
    
    def process(self):
        def restore_bbox(box, transform_info): #裁减图像
            ratio_x, ratio_y = transform_info["ratio"]
            tx1, ty1 = transform_info["txy"]
            org_w, org_h = transform_info["org_size"]

            x1 = (box[0] - tx1) / ratio_x
            y1 = (box[1] - ty1) / ratio_y
            x2 = (box[2] - tx1) / ratio_x
            y2 = (box[3] - ty1) / ratio_y

            x1 = max(0, min(x1, org_w - 1))
            x2 = max(0, min(x2, org_w - 1))
            y1 = max(0, min(y1, org_h - 1))
            y2 = max(0, min(y2, org_h - 1))

            return [x1, y1, x2, y2]

        frame = self.frame_grabber.get_frame()
        if frame is None:
            logging.info(f"无法获取图像帧。")
            return None

        results_det = self.visual_detect([frame])
        number_boxes = results_det[0][1]
        transform_info = results_det[0][4]
        results_rec = []
        for box in number_boxes:
            cor = restore_bbox(box, transform_info)
            crop_img = frame[int(cor[1]):int(cor[3]), int(cor[0]):int(cor[2]), :]
            result_rec = self.visual_recognize([crop_img])
            results_rec.append(result_rec)
        
        if len(results_rec) == 0:
            return {"result": "", "content": "未识别到公交线路"}
        else: 
            result_str = find_number(results_rec[0][0][0])
            if result_str is None:
                return {"result": "", "content": "未识别到有效公交线路"}
            else: 
                content_str = f"目前来的是{result_str}路公交车"
                
                # 返回一个包含 result 和 content 的新字典
                result_rec_dict = {
                    "result": result_str,
                    "content": content_str
                }
                return result_rec_dict

# 继承RedisClient，形成DET类，为光学识别类。 发障碍物文本
class DETClient(RedisClient):
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        host: str = "localhost",
        port: int = 6379,
    ):
        super().__init__(host=host, port=port)

        # 请在此修改各项参数
        self.publish = {"channel:vision_to_tts": {"eventType": "obstacle_detected"}}
        # 不订阅
        self.last_broadcast_time = 0

        self.frame_grabber = frame_grabber
        self.visual_detect = VisualDetect(
            bmodel_path="/data/sophon-demo/sample/YOLOv8_plus_det/models/test/yolov8s_fp16_1b.bmodel",
            dev_id=0,
            use_resize_padding=True
        )

    def main_loop(self):
        time.sleep(2)

        while self._running: # 请用此标志控制循环
            obstacle_text = self.process()
            # 2. 检查结果是否有效（不是None也不是空字符串）
            if not obstacle_text:
                # 如果没有检测到有效障碍物，就进入下一次循环
                time.sleep(0.2) # 短暂休眠，避免CPU占用过高
                continue
            # 3. 检查是否在3秒的不应期（冷却时间）内
            current_time = time.time()
            if (current_time - self.last_broadcast_time) < 7:
                # 如果距离上次广播还不到3秒，也跳过本次广播
                print(f"--- 处于不应期，忽略障碍物: {obstacle_text} ---")
                time.sleep(0.2)
                continue
            # 4. 如果通过了所有检查，就执行广播
            print(f"✅✅✅ 发现有效障碍物并准备广播: {obstacle_text}")
            
            # 构建要发布的消息体
            results_det_dict = {"content": obstacle_text}

            for channel_topublish, filter_topublish in self.publish.items():
                to_publish = {**filter_topublish, **results_det_dict}
                self.client.publish(channel_topublish, json.dumps(to_publish))
            
            # 5. 【关键】广播后，立刻更新上次广播的时间戳
            self.last_broadcast_time = current_time
            print(f"--- 广播完成，进入3秒不应期 ---")

            time.sleep(0.2) # 短暂休眠
            
    def process(self):
        frame = self.frame_grabber.get_frame()
        if frame is None:
            logging.info(f"无法获取图像帧。")
            return None

        results_det = self.visual_detect([frame])
        result_str = generate_warning(results_det[0])
        
        return result_str

# 继承RedisClient，形成DES类，为识别描述类。 发障碍物json
class DESClient(RedisClient):
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        host: str = "localhost",
        port: int = 6379,
    ):
        super().__init__(host=host, port=port)

        self.publish = {"event:vision:detection": {"eventType": "obstacle_described"}}
        self.subscribe = {"gamepad:events": {"name": "R1", "state": "pressed"}}

        self.pubsub = self.client.pubsub()
        self.pubsub.subscribe(*list(self.subscribe.keys()))
        self.listen = self.pubsub.listen #迭代器

        self.frame_grabber = frame_grabber
        self.visual_detect = VisualDetect(
            bmodel_path="/data/sophon-demo/sample/YOLOv8_plus_det/models/test/yolov8s_fp16_1b.bmodel",
            dev_id=0,
            use_resize_padding=True
        )
    
    def main_loop(self):
        while self._running:
            message = self.pubsub.get_message(timeout=1)
            if message is None:
                continue

            if message["type"] != "message":
                continue

            channel = message["channel"]
            data = message["data"]
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue

            # 遍历订阅字典，查看消息是否命中消息
            for subscribed_channel, subscribed_filter in self.subscribe.items():
                if channel != subscribed_channel:
                    continue

                if all(payload.get(k) == v for k, v in subscribed_filter.items()):
                    result_des_dict = self.process()
                    if result_des_dict is None:
                        continue

                    # 遍历发布字典，发布到应当发布的所有频道
                    for channel_topublish, filter_topublish in self.publish.items():
                        to_publish = {**filter_topublish, **result_des_dict}

                        self.client.publish(channel_topublish, json.dumps(to_publish))
            
        self.pubsub.unsubscribe()
        self.pubsub.close()
    
    def process(self):
        frame = self.frame_grabber.get_frame()
        if frame is None:
            logging.info(f"无法获取图像帧。")
            return None

        results_det = self.visual_detect([frame])
        result_str = generate_locations(results_det[0])
        result_des_dict = {"result": result_str}
        return result_des_dict


# 继承RedisClient， 台阶检测Step Detection Client.
class StepDetectorClient(RedisClient):
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        host: str = "localhost",
        port: int = 6379,
    ):
        super().__init__(host=host, port=port)

        # 请在此修改各项参数
        self.publish = {"channel:vision_to_tts": {"eventType": "stair_detected"}}
        # 不订阅
        self.last_broadcast_time = 0

        self.frame_grabber = frame_grabber
        self.visual_detect = StairDetect(
            bmodel_path="/data/Zoo2/Radxa-Model-Zoo/sample/YOLOv8_plus_det/yolov8s_fp16_1b.bmodel",
            dev_id=0,
            conf_thresh=0.8,
            nms_thresh=0.5
        )

    def main_loop(self):
        time.sleep(2)

        while self._running: # 请用此标志控制循环
            obstacle_text = self.process()
            # 2. 检查结果是否有效（不是None也不是空字符串）
            if not obstacle_text:
                # 如果没有检测到有效障碍物，就进入下一次循环
                time.sleep(0.2) # 短暂休眠，避免CPU占用过高
                continue
            # 3. 检查是否在3秒的不应期（冷却时间）内
            current_time = time.time()
            if (current_time - self.last_broadcast_time) < 7:
                # 如果距离上次广播还不到3秒，也跳过本次广播
                print(f"--- 处于不应期，忽略障碍物: {obstacle_text} ---")
                time.sleep(0.2)
                continue
            # 4. 如果通过了所有检查，就执行广播
            print(f"✅✅✅ 发现有效障碍物并准备广播: {obstacle_text}")
            
            # 构建要发布的消息体
            results_det_dict = {"content": obstacle_text}

            for channel_topublish, filter_topublish in self.publish.items():
                to_publish = {**filter_topublish, **results_det_dict}
                self.client.publish(channel_topublish, json.dumps(to_publish))
            
            # 5. 【关键】广播后，立刻更新上次广播的时间戳
            self.last_broadcast_time = current_time
            print(f"--- 广播完成，进入3秒不应期 ---")

            time.sleep(0.2) # 短暂休眠
            
    def process(self):
        frame = self.frame_grabber.get_frame()
        if frame is None:
            logging.info(f"无法获取图像帧。")
            return None

        results_det = self.visual_detect([frame])
        result_str = generate_stair_warning(results_det[0])
        
        return result_str
    # vision_clients.py -> class StepDetectorClient

    # def process(self):
    #     # 使用一张同时包含人和台阶的图片进行测试
    #     frame = cv2.imread("/data/Zoo2/Radxa-Model-Zoo/sample/YOLOv8_plus_det/test/5bc790dcc4e8ee05_jpg.rf.3c658ce07a7967a23cf0d200417ac08a.jpg") # <--- ！！！修改为你的图片路径
    #     if frame is None:
    #         print("错误：无法读取测试图片，请检查路径。")
    #         return None

    #     results_det = self.visual_detect([frame])
        
    #     print("----------- 模型原始输出 -----------")
    #     detected_something = False
    #     for class_id, detected_boxes in enumerate(results_det[0][:-1]):
    #         if len(detected_boxes) > 0:
    #             print(f"✅ 发现物体！类别ID: {class_id}, 数量: {len(detected_boxes)}")
    #             detected_something = True
        
    #     if not detected_something:
    #         print("❌ 在这张图片中没有检测到任何已知物体。")
    #     print("------------------------------------")

    #     # 调试时先返回None
    #     return None

cls_map_pavement = {
    0: "直线盲道",
    1: "提示盲道",
    2: "路缘石",
    3: "坑洼",
    4: "积水"
}
class PavementDetectorClient(RedisClient):
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        host: str = "localhost",
        port: int = 6379,
    ):
        super().__init__(host=host, port=port)

        self.publish = {"event:vision:detection": {"eventType": "pavement_detected"}}
        self.subscribe = {"gamepad:events": {"name": "R2", "state": "pressed"}}

        self.pubsub = self.client.pubsub()
        self.pubsub.subscribe(*list(self.subscribe.keys()))
        self.listen = self.pubsub.listen #迭代器

        self.frame_grabber = frame_grabber
        self.visual_detect = YoloSegDetect(
            bmodel_path="/data/sophon-demo/sample/YOLOv8_plus_seg/models/BM1684X/yolov8s_fp16_1b.bmodel",
            dev_id=0,
            conf_thresh=0.7, # 可根据需要调整
            nms_thresh=0.7
        )
    
    def main_loop(self):
        while self._running:
            message = self.pubsub.get_message(timeout=1)
            if message is None:
                continue

            if message["type"] != "message":
                continue

            channel = message["channel"]
            data = message["data"]
            try:
                payload = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                continue

            # 遍历订阅字典，查看消息是否命中消息
            for subscribed_channel, subscribed_filter in self.subscribe.items():
                if channel != subscribed_channel:
                    continue

                if all(payload.get(k) == v for k, v in subscribed_filter.items()):
                    result_des_dict = self.process()
                    if result_des_dict is None:
                        continue

                    # 遍历发布字典，发布到应当发布的所有频道
                    for channel_topublish, filter_topublish in self.publish.items():
                        to_publish = {**filter_topublish, **result_des_dict}

                        self.client.publish(channel_topublish, json.dumps(to_publish))
            
        self.pubsub.unsubscribe()
        self.pubsub.close()
    # def main_loop(self):
    #     """
    #     一个持续运行的调试循环，完全绕过Redis手柄指令。
    #     它会每隔2秒钟自动调用一次 process 方法。
    #     """
    #     print("PavementDetectorClient: 已进入持续调试模式...")
    #     time.sleep(2) # 暂停2秒，等待其他服务完全启动

    #     while self._running:
    #         # 1. 直接调用 process 方法进行图像处理
    #         #    (您当前的 process 方法是读取静态图片并打印结果)
    #         self.process()
            
    #         # 2. 暂停2秒，避免CPU占用过高，也方便您观察打印结果
    #         time.sleep(2) 
    
    def process(self):
        frame = self.frame_grabber.get_frame()
        results_det = self.visual_detect([frame])
        if not results_det or len(results_det[0][0]) == 0:
            # print("❌ 在这张图片中没有检测到任何已知物体。")
            result_str = "您附近没有盲道"
        else:
            boxes = results_det[0][0]
            
            # 4. 调用新的位置生成函数，现在它能收到完整的(N, 6)数据了
            result_str = generate_pavement_location(boxes, cls_map_pavement)
        print(f"[盲道识别程序]{result_str}")

        result_des_dict = {"content": result_str}
        return result_des_dict


class QWenPhotoClient(RedisClient):
    """
    一个专用的Redis客户端，用于在接收到特定手柄信号时捕获并发布照片路径。

    功能:
    1. 监听 'gamepad:events' 频道中的 L1 按键信号。
    2. 收到信号后，从摄像头抓取一帧图像。
    3. 将图像以唯一的名称 (时间戳_uuid.jpg) 保存到指定目录。
    4. 将包含图片绝对路径的JSON消息发布到 'event:vision:photo' 频道。
    """
    def __init__(
        self,
        frame_grabber: FrameGrabber,
        save_dir: str = "/data/RECOsys_data_cache/TTS_wav", # 图片保存目录
        host: str = "localhost",
        port: int = 6379,
    ):
        """
        初始化客户端。

        Args:
            frame_grabber (FrameGrabber): 用于从摄像头获取图像帧的实例。
            save_dir (str): 捕获的图片将被保存到的文件夹路径。
            host (str): Redis服务器主机。
            port (int): Redis服务器端口。
        """
        super().__init__(host=host, port=port)
        self.frame_grabber = frame_grabber
        self.save_dir = save_dir

        # 启动时确保图片保存目录存在
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"图片将保存至: {self.save_dir}")
        except OSError as e:
            print(f"错误：无法创建图片保存目录 {self.save_dir}: {e}")
            raise  # 无法创建目录则直接抛出异常，服务无法正常运行

        # 订阅手柄L1按键事件
        self.pubsub = self.client.pubsub()
        self.pubsub.subscribe("gamepad:events")
        
        self._running = True

    def _generate_unique_filepath(self) -> str:
        """生成一个基于时间戳和UUID的、独一无二的文件路径。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]  # 使用UUID前8位增加唯一性
        filename = f"capture_{timestamp}_{unique_id}.jpg"
        return os.path.join(self.save_dir, filename)

    def main_loop(self):
        """主循环，监听Redis消息并处理L1信号。"""
        print("Photo Capture Client 已启动，等待L1信号...")
        for message in self.pubsub.listen():
            if not self._running:
                break
            
            if message["type"] != "message":
                continue

            try:
                payload = json.loads(message["data"])
                
                # 检查是否是我们关心的L1按键按下的信号
                if payload.get("name") == "L1" and payload.get("state") == "pressed":
                    print("接收到L1信号，正在捕获图像...")
                    
                    # 1. 从摄像头获取一帧
                    frame = self.frame_grabber.get_frame()
                    if frame is None:
                        print("错误：未能从摄像头获取到图像帧。")
                        continue

                    # 2. 生成唯一的保存路径
                    filepath = self._generate_unique_filepath()

                    # 3. 保存图像到指定位置
                    success = cv2.imwrite(filepath, frame)
                    if not success:
                        print(f"错误：保存图像到 {filepath} 失败。")
                        continue
                    
                    print(f"成功保存图像: {filepath}")

                    # 4. 在 'event:vision:photo' 频道发布路径
                    publish_payload = {
                        "eventType": "image_capture",
                        "path": filepath
                    }
                    self.client.publish("event:vision:photo", json.dumps(publish_payload))
                    print(f"已在 'event:vision:photo' 频道发布路径。")

            except (json.JSONDecodeError, ValueError) as e:
                # 忽略无法解析的消息
                print(f"忽略无效的JSON消息: {e}")
                continue
            except Exception as e:
                print(f"处理消息时发生未知错误: {e}")
    
    def stop(self):
        """停止监听循环。"""
        self._running = False
        self.pubsub.unsubscribe()
        self.pubsub.close()
        print("Photo Capture Client 已停止。")






import time
import logging


def main():
    """
    主函数，负责初始化、启动和管理所有视觉客户端服务。
    """
    # 配置日志记录，方便在终端看到所有服务的输出信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化所有对象变量为 None，方便在 finally 中进行安全检查
    frame_grabber = None
    ocr_client = None
    det_client = None
    des_client = None
    step_detector_client = None
    qwen_photo_client = None

    try:
        # --- 1. 初始化共享的摄像头抓取器 ---
        print("🚀 [主程序] 正在初始化摄像头 FrameGrabber...")
        # 将摄像头初始化放在 try...except 块中，如果失败则直接退出
        try:
            frame_grabber = FrameGrabber(camara_id=0)
            print("✅ [主程序] 摄像头 FrameGrabber 初始化成功。")
        except RuntimeError as e:
            print(f"❌ [主程序] 致命错误：摄像头初始化失败！ {e}")
            print("   请检查摄像头是否连接正确，以及是否有权限访问（例如，用户是否在 'video' 组）。")
            return # 直接退出程序

        # --- 2. 初始化三个独立的视觉客户端 ---
        # 它们都使用同一个 frame_grabber 实例，以避免资源冲突
        print("🚀 [主程序] 正在初始化三个视觉客户端...")
        ocr_client = OCRClient(frame_grabber=frame_grabber)
        det_client = DETClient(frame_grabber=frame_grabber)
        # des_client = DESClient(frame_grabber=frame_grabber)
        step_detector_client = StepDetectorClient(frame_grabber=frame_grabber)  # 台阶检测客户端
        pavement_detector_client = PavementDetectorClient(frame_grabber=frame_grabber)
        qwen_photo_client = QWenPhotoClient(frame_grabber=frame_grabber)
        print("✅ [主程序] 六个视觉客户端初始化成功。")

        # --- 3. 在各自的线程中启动三个客户端 ---
        print("🚀 [主程序] 正在启动所有客户端线程...")
        ocr_client.start()  # 启动 OCR (L2键触发)
        det_client.start()  # 启动 DET (持续检测)
        # des_client.start()  # 启动 DES (R2键触发)
        step_detector_client.start()  # 启动 StepDetector (持续检测台阶)
        pavement_detector_client.start()  # 启动 PavementDetector (R1键触发)
        qwen_photo_client.start()

        print("✅ [主程序] 所有客户端线程已启动，系统现在运行中...")
        print("   - DETClient 正在持续检测障碍物...")
        print("   - OCRClient 正在等待 'L2' 键按下...")
        print("   - DESClient 正在等待 'R2' 键按下...")
        print("   - StepDetectorClient 正在持续检测台阶...")
        print("   - PavementDetectorClient 正在等待 'R1' 键按下...")
        print("   - QWenPhotoClient 正在等待 'L1' 键按下...")
        print("\n--- 按下 Ctrl+C 可以安全停止所有服务 ---")

        # --- 4. 主线程进入守护循环 ---
        # 这个循环让主线程保持存活，以便子线程可以持续运行
        # 同时它也在这里等待用户的 Ctrl+C 信号
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        # 当用户按下 Ctrl+C 时，会触发这个异常
        print("\n🛑 [主程序] 检测到退出信号 (Ctrl+C)，正在准备关闭所有服务...")

    except Exception as e:
        # 捕捉其他可能的意外错误
        print(f"❌ [主程序] 发生未预料的致命错误: {e}")

    finally:
        # --- 5. 优雅地停止所有服务 ---
        # finally 块确保无论程序是正常退出还是异常退出，都会执行清理工作
        print("🧹 [主程序] 正在停止所有服务，请稍候...")
        
        # 依次停止每个客户端的线程
        if ocr_client:
            print("   - 正在停止 OCR 客户端...")
            ocr_client.stop()
        
        if det_client:
            print("   - 正在停止 DET 客户端...")
            det_client.stop()

        # if des_client:
        #     print("   - 正在停止 DES 客户端...")
        #     des_client.stop()
        
        if step_detector_client:
            print("   - 正在停止 StepDetector 客户端...")
            step_detector_client.stop()
            
        # 最后停止摄像头抓取器线程
        if frame_grabber:
            print("   - 正在停止摄像头 FrameGrabber...")
            frame_grabber.stop()

        if pavement_detector_client:
            print("   - 正在停止 PavementDetector 客户端...")
            pavement_detector_client.stop()

        if qwen_photo_client:
            print("   - 正在停止 QWenPhotoClient 客户端...")
            qwen_photo_client.stop()
            
        print("✅ [主程序] 所有服务已安全停止。程序退出。")


if __name__ == "__main__":
    main()