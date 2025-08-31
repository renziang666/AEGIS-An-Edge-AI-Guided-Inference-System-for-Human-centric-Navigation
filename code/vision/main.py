# main.py

from vision_clients import FrameGrabber, OCRClient, DETClient

import time
import logging

logging.basicConfig(level=logging.INFO)

# 启动头
def main():
    frame_grabber = FrameGrabber()
    ocr_client = OCRClient(frame_grabber)
    ocr_client.start()
    det_client = DETClient(frame_grabber)
    det_client.start()
    logging.info("视觉部分子线程已完全启动。")

    time.sleep(10)
    ocr_client.stop()
    det_client.stop()
    frame_grabber.stop()
    logging.info("视觉部分子线程已完全终止。")

if __name__ == "__main__":
    main()