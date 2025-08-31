# text_generate.py

import re
from typing import List, Union
from collections import Counter
import numpy as np

cls_dict = [
    "停车标志", "人", "自行车", "公交车",
    "卡车", "汽车", "摩托车", "反光锥",
    "垃圾桶", "警示柱", "球形路障", "电线杆",
    "狗", "三轮车", "消防栓"
]

def generate_warning(input: List[Union[List, dict]]) -> str:
    if len(input) != len(cls_dict) + 1:
        raise ValueError("类别字典长度与输入数组长度不匹配。")

    pre = "注意，前方有"
    mid = ""
    for idx, sublist in enumerate(input[:-1]):
        if len(sublist) == 0 or idx == 0:
            continue
        else: mid = mid + str(len(sublist)) + "个" + cls_dict[idx] + "，"
    post = "请注意安全。"
    output = pre + mid + post

    if mid == "":
        return None
    else: return output

def find_number(input: str) -> str:
    match = re.search(r"\d+", input)
    if match:
        return int(match.group(0))
    return None

def generate_locations(input: List[Union[List, dict]]) -> str:
    boxes = input[:-1]
    cls_num = len(boxes)
    cls_list = range(cls_num)
    left, middle, right = Counter(), Counter(), Counter()

    for idx, cls_boxes in enumerate(boxes):
        for cls_box in cls_boxes:
            x1, y1, x2, y2, *res = cls_box
            xc = (x1 + x2)/2
            yc = (y1 + y2)/2
            if xc < 640 / 3:
                left.update([idx])
            elif xc < 640 * 2 / 3:
                middle.update([idx])
            else:
                right.update([idx])
    
    def describe(counter, position):
        parts = []
        for idx in cls_list:
            if counter[idx] > 0:
                parts.append(f"{counter[idx]}个{cls_dict[idx]}")
        if parts:
            return f"{position}有" + "、".join(parts)
        else:
            return ""

    descriptions = list(filter(None, [
        describe(left, "左侧"),
        describe(middle, "中间"),
        describe(right, "右侧")
    ]))

    result = "，".join(descriptions) + "。" if descriptions else "未检测到障碍物。"

    return result

STAIR_CLASS_ID = 0 
def generate_stair_warning(input: List[Union[List, dict]]) -> str:
    """
    一个专门为台阶检测设计的警告生成函数。
    它会优先检查是否存在台阶，如果存在，则返回特定的台阶警告。
    """
    # 检查台阶ID对应的检测框列表是否为空
    stair_detections = input[STAIR_CLASS_ID]
    
    if len(stair_detections) > 0:
        # 检测到台阶，立即返回独特的、高优先级的提示
        return f"请注意，前方发现{len(stair_detections)}处台阶，请慢行并注意脚下安全。"
    
    # 如果没有检测到台阶，则返回 None，表示没有需要播报的紧急情况
    return None


cls_map_pavement = {
    0: "直线盲道",
    1: "提示盲道",
    2: "路缘石",
    3: "坑洼",
    4: "积水"
}

# 2. 编写全新的、智能的位置生成函数
def generate_pavement_location(boxes: np.ndarray, cls_map: dict) -> str:
    """
    专为多类别路面分割模型设计的函数。
    它将多个相关的ID（如各种盲道）概括为同一类别（“盲道”），并描述其位置。
    :param boxes: 模型直接输出的 boxes 数组，形状为 (N, 6)，列为 x1,y1,x2,y2,score,id
    :param cls_map: 类别ID到名称的映射字典
    """
    left_items = set()
    middle_items = set()
    right_items = set()

    # 将多个ID归纳为几个关键类别
    tactile_ids = {0, 1, 2, 3, 4}       # 假设ID 0和1都代表盲道
    
    for det in boxes:
        x1, _, x2, _, _, class_id_float = det
        class_id = int(class_id_float)
        xc = (x1 + x2) / 2
        
        # 确定归纳后的类别名称
        category = None
        if class_id in tactile_ids:
            category = "盲道"
    
        if category is None:
            continue

        # 将类别名称添加到对应位置的集合中（set可以自动去重）
        if xc < 640 / 3:
            left_items.add(category)
        elif xc < 640 * 2 / 3:
            middle_items.add(category)
        else:
            right_items.add(category)

    def describe(position, item_set):
        if not item_set:
            return ""
        # 使用"、"连接一个位置发现的所有类别
        description = "、".join(sorted(list(item_set)))
        return f"{position}有{description}"

    descriptions = list(filter(None, [
        describe("左侧", left_items),
        describe("中间", middle_items),
        describe("右侧", right_items)
    ]))

    if not descriptions:
        return "前方路面平整。"
        
    return "注意，" + "，".join(descriptions) + "。"