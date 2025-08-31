from difflib import get_close_matches
from pypinyin import lazy_pinyin
import Levenshtein


class Correction_Station:
    def __init__(self):
        self.stdlist_path = "/home/linaro/smart_cane_project/navigation/beijing_offline/data/subway_map.txt"
        self.station_list = []
        

        self.standard_stations = self.load_standard_stations()

    def load_standard_stations(self):
        with open(self.stdlist_path, 'r', encoding='utf-8') as f:
            # 直接读取时过滤空行和&开头的行
            standard_stations = [
                line.strip() for line in f
                if line.strip() and not line.strip().startswith('&')
            ]
        return standard_stations

    def correct_station_names_pinyin_first(self,station_list, 
                                        pinyin_threshold=0.9, str_threshold=0.5):
        """
        优先用拼音匹配（高阈值），失败后再用字符串匹配
        
        :param station_list: 待修正的站名列表
        :param standard_stations: 标准站名列表
        :param pinyin_threshold: 拼音相似度阈值（更高更严格）
        :param str_threshold: 字符串相似度阈值 (基于 SequenceMatcher 的相似度计算)
        :return: 修正后的站名列表
        """
        corrected_list = []
        # 预计算标准站名的拼音（提升性能）
        std_pinyin = {s: ''.join(lazy_pinyin(s)) for s in self.standard_stations}
        
        for station in station_list:
            station_pinyin = ''.join(lazy_pinyin(station))
            best_match = None
            max_sim = 0
            
            # 1. 优先拼音匹配（高阈值）
            for s, s_pinyin in std_pinyin.items():
                # 计算拼音相似度（Levenshtein比率）
                pinyin_sim = Levenshtein.ratio(station_pinyin, s_pinyin)
                if pinyin_sim >= pinyin_threshold and pinyin_sim > max_sim:
                    best_match = s
                    max_sim = pinyin_sim
            
            # 2. 拼音无匹配时，再用字符串匹配
            if best_match is None:
                str_matches = get_close_matches(station, self.standard_stations, n=1, cutoff=str_threshold)
                best_match = str_matches[0] if str_matches else "error_station_name"
            
            corrected_list.append(best_match)
        
        return corrected_list


if __name__=="__main__":

    correction_station = Correction_Station()

    list1 = ["苹果园", "大望录", "希值们", "吴克松", "天通园北","错误站名", "良乡大答城北"]

    corrected = correction_station.correct_station_names_pinyin_first(list1)
    print("原结果:",list1)
    print("修正结果:", corrected)
