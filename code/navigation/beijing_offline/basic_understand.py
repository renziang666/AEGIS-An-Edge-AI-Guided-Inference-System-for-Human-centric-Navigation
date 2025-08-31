import re
import os


class Capture_Destination:
    def __init__(self,result_path):
        self.result_path = result_path
        self.texts = []
        self.destinations = []
    
    def read_txt(self):
        with open(self.result_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
            # 现在 lines 是一个字符串列表，每个元素是一行内容
        print("原结果：",self.texts)

    def capture(self):
        self.read_txt()
        # 模式1：匹配 "导航到" 或 "我要去" 后面的目的地（连续非空字符）
        pattern1 = r"(?:导航到|我要去)(\S+)"
        # 模式2：匹配前面跟着"怎么走"的目的地
        pattern2 = r"(\S+)(?=怎么走)"

        texts = self.texts

        for text in texts:
            m = re.search(pattern1, text)
            if m:
                dest = m.group(1)
            else:
                m = re.search(pattern2, text)
                if m:
                    dest = m.group(1)
                else:
                    dest = None
            if dest:
                self.destinations.append(dest)

        unique_destinations = list(set(self.destinations))# 去重后输出目的地列表

        return unique_destinations

if __name__ == "__main__":
    result_path = "/home/admin/renziang_program/voice/WeNet_v2/result.txt"
    capture_destination = Capture_Destination(result_path)
    print("提取的目的地：", capture_destination.capture())
