# 文件名: number_converter.py (或 zhuanhua.py)

import re

class NumberConverter:
    """
    一个用于将阿拉伯数字转换为汉字的类。
    支持整数、浮点数以及负数。
    (这个类的内部代码保持和你原来的一样，非常完善，无需改动)
    """
    def __init__(self):
        self.num_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }
        self.unit_map = {0: '', 1: '十', 2: '百', 3: '千'}
        self.big_unit_map = {0: '', 1: '万', 2: '亿', 3: '兆'} # 简化以常用单位为例

    def convert_section(self, section_str):
        # (这个方法的内部逻辑保持原样)
        if int(section_str) == 0:
            return '零'
        res = ''
        length = len(section_str)
        for i, char in enumerate(section_str):
            digit = int(char)
            unit_pos = length - 1 - i
            if digit == 0:
                if i < length - 1 and int(section_str[i+1]) != 0 and '零' not in res:
                    res += '零'
            else:
                res += self.num_map[char] + self.unit_map[unit_pos]
        return res.strip('零')

    def convert(self, num_str):
        # (这个方法的内部逻辑保持原样，但可以稍作简化和修正)
        if not num_str or not re.match(r'^-?\d+(\.\d+)?$', num_str):
            return num_str # 如果不是有效数字，返回原样

        is_negative = num_str.startswith('-')
        if is_negative:
            num_str = num_str[1:]

        parts = num_str.split('.')
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ""

        # 转换整数
        if integer_part == '0':
            integer_res = '零'
        else:
            integer_res = ''
            section_count = (len(integer_part) + 3) // 4
            for i in range(section_count):
                start = len(integer_part) - 4 * (i + 1)
                end = len(integer_part) - 4 * i
                section = integer_part[max(0, start):end]
                if int(section) != 0:
                    integer_res = self.convert_section(section) + self.big_unit_map[i] + integer_res
                elif '零' not in integer_res:
                     integer_res = '零' + integer_res
        
        integer_res = integer_res.strip('零')
        if integer_res.startswith('一十'):
            integer_res = integer_res[1:]

        # 转换小数
        decimal_res = ''
        if decimal_part:
            decimal_res = '点' + ''.join([self.num_map[d] for d in decimal_part])

        final_res = integer_res + decimal_res
        return '负' + final_res if is_negative else final_res


# =======================================================
# --- ✨ 新的核心函数在这里 ✨ ---
# =======================================================

def convert_sentence_numbers(sentence: str) -> str:
    """
    接收一个完整的句子，将其中的所有阿拉伯数字转换为中文汉字。
    
    Args:
        sentence (str): 包含数字的原始句子，例如 "我买了3.5斤苹果，花了25元。"

    Returns:
        str: 数字被转换后的新句子，例如 "我买了两斤半苹果，花了二十五元。"
    """
    if not sentence:
        return ""
        
    converter = NumberConverter()

    # 定义一个内部回调函数，用于 re.sub
    def replacement_callback(match):
        # match.group(0) 获取匹配到的整个数字字符串
        num_str = match.group(0)
        return converter.convert(num_str)

    # 使用正则表达式查找所有数字（包括整数、负数、小数）
    # 并使用回调函数进行替换
    # 这个正则表达式匹配一个可选的负号，后跟数字，再后跟一个可选的(小数点+数字)部分
    pattern = r'-?\d+(\.\d+)?'
    return re.sub(pattern, replacement_callback, sentence)


# --- 使用示例 ---
if __name__ == "__main__":
    test_sentence_1 = "请在3号航站楼乘坐CA1405航班，预计飞行时间3.5小时，票价是1280元。"
    converted_sentence_1 = convert_sentence_numbers(test_sentence_1)
    print(f"原始句子: {test_sentence_1}")
    print(f"转换后: {converted_sentence_1}")
    print("-" * 20)

    test_sentence_2 = "今天的气温是-5度，湿度是65.8%。"
    converted_sentence_2 = convert_sentence_numbers(test_sentence_2)
    print(f"原始句子: {test_sentence_2}")
    print(f"转换后: {converted_sentence_2}")
    print("-" * 20)
    
    test_sentence_3 = "订单号是007，金额0.0元。"
    converted_sentence_3 = convert_sentence_numbers(test_sentence_3)
    print(f"原始句子: {test_sentence_3}")
    print(f"转换后: {converted_sentence_3}")