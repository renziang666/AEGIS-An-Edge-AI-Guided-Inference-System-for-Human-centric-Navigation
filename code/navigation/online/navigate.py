import requests
import json
from urllib.request import urlopen
from urllib.parse import quote
import os
import re # 导入re模块用于清理HTML标签

# 你的百度地图AK，保持不变
BAIDU_AK = "" # 请务必替换为你的实际AK

def address_to_coord(address: str):
    """将地址转换为百度地图坐标系经纬度"""
    root_url = "http://api.map.baidu.com/geocoding/v3/"
    output = 'json'
    add = quote(address)
    url = f"{root_url}?address={add}&output={output}&ak={BAIDU_AK}"
    try:
        req = urlopen(url)
        res = req.read().decode()
        temp = json.loads(res)
        if temp['status'] == 0:
            lng = temp['result']['location']['lng']
            lat = temp['result']['location']['lat']
            return lng, lat
        else:
            print(f"地址解析失败: {address}, 错误: {temp.get('message', '未知错误')}")
            return None, None
    except Exception as e:
        print(f"请求地址解析API时发生错误: {e}")
        return None, None

def get_navigation_route2(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float):
    """
    从百度地图API获取所有公共交通和出租车数据。
    """
    root_url2 = "https://api.map.baidu.com/directionlite/v1/transit"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "ak": BAIDU_AK,
    }

    try:
        response = requests.get(root_url2, params, timeout=10)
        response.raise_for_status() # 检查HTTP请求是否成功
        data = response.json()
        
        all_parsed_info = {
            "subway_route": None,
            "bus_route": None,
            "taxi_info": None,
            "status": data.get("status"),
            "message": data.get("message", "")
        }

        if data.get("status") == 0:
            result = data.get("result", {})
            
            # --- 解析并分类所有公共交通路线 ---
            if "routes" in result and result["routes"]:
                all_subway_candidate_routes = []
                all_bus_candidate_routes = []
                
                step_type_map = {
                    3: "公交", 1: "地铁、轻轨", 5: "步行",
                    "vehicle_type_1": "地铁、轻轨", "vehicle_type_0": "公交"
                }

                for route in result["routes"]:
                    route_has_subway = False
                    route_has_bus = False
                    processed_steps = []

                    for step_group in route.get("steps", []):
                        for step in step_group:
                            step_info = {
                                "type": step_type_map.get(step.get('type'), "其他"),
                                "instruction": step.get('instruction'),
                                "distance_meters": step.get('distance'),
                                "duration_seconds": step.get('duration')
                            }
                            if 'vehicle' in step and step['vehicle']:
                                vehicle_info = step.get('vehicle', {})
                                v_type = vehicle_info.get("type")
                                if v_type == 1:
                                    step_info["type"] = "地铁、轻轨"
                                    route_has_subway = True
                                elif v_type == 0 or v_type == 6: # 普通公交或夜班车
                                    step_info["type"] = "公交"
                                    route_has_bus = True
                                
                                step_info.update({
                                    "vehicle_name": vehicle_info.get("name"),
                                    "start_station": vehicle_info.get("start_name"),
                                    "end_station": vehicle_info.get("end_name"),
                                    "stop_num": vehicle_info.get("stop_num")
                                })
                            processed_steps.append(step_info)
                    
                    current_route_parsed = {
                        "total_distance_meters": route.get('distance'),
                        "total_duration_seconds": route.get('duration'),
                        "total_price_yuan": route.get('price'),
                        "steps": processed_steps
                    }

                    if route_has_subway:
                        all_subway_candidate_routes.append(current_route_parsed)
                    elif route_has_bus:
                        all_bus_candidate_routes.append(current_route_parsed)
                
                # 选择最佳（第一个）方案
                if all_subway_candidate_routes:
                    all_parsed_info["subway_route"] = all_subway_candidate_routes[0]
                if all_bus_candidate_routes:
                    all_parsed_info["bus_route"] = all_bus_candidate_routes[0]
            
            # --- 解析出租车信息 ---
            if "taxi" in result and result["taxi"]:
                taxi_info = result["taxi"]
                all_parsed_info["taxi_info"] = {
                    "distance_meters": taxi_info.get('distance'),
                    "duration_seconds": taxi_info.get('duration'),
                    "remark": taxi_info.get("remark"),
                    "prices": [{
                        "description": detail.get("desc"),
                        "total_price_yuan": detail.get('total_price')
                    } for detail in taxi_info.get("detail", [])]
                }
        
        return all_parsed_info

    except requests.exceptions.RequestException as e:
        return {"status": -1, "message": f"请求失败: {e}"}
    except json.JSONDecodeError as e:
        return {"status": -1, "message": f"JSON解析错误: {e}"}


# --- 新增的步行路线获取函数 ---
def get_walking_route_info(origin_lng: float, origin_lat: float, dest_lng: float, dest_lat: float):
    
    url = "https://api.map.baidu.com/direction/v2/walking"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "ak": BAIDU_AK, # 使用全局的BAIDU_AK
    }

    try:
        response = requests.get(url=url, params=params, timeout=10) 
        response.raise_for_status() 

        data = response.json()
        
        if data.get("status") == 0:
            result_info = {
                "status": 0,
                "message": "成功",
                "total_distance": "N/A",
                "total_duration": "N/A",
                "steps": []
            }

            if data["result"] and data["result"]["routes"]:
                route = data["result"]["routes"][0]
                
                result_info["total_distance"] = route.get('distance', 'N/A')
                result_info["total_duration"] = route.get('duration', 'N/A')

                for step in route["steps"]:
                    instructions = step.get("instructions", "无具体指令")
                    name = step.get("name", "").strip() 
                    
                    # 使用正则表达式移除HTML标签
                    clean_instructions = re.sub(r'<[^>]+>', '', instructions)
                    
                    result_info["steps"].append({
                        "instructions": clean_instructions,
                        "road_name": name
                    })
            else:
                result_info["status"] = 2001 
                result_info["message"] = "未找到可用的步行路线方案。"
            
            return result_info

        else:
            return {
                "status": data.get("status"),
                "message": data.get('message', 'API返回未知错误。')
            }

    except requests.exceptions.HTTPError as http_err:
        return {
            "status": response.status_code if 'response' in locals() else 'N/A',
            "message": f"HTTP错误：{http_err}"
        }
    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": -1, 
            "message": f"连接错误：请检查网络或API地址。{conn_err}"
        }
    except requests.exceptions.Timeout as timeout_err:
        return {
            "status": -2, 
            "message": f"请求超时：{timeout_err}"
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": -3, 
            "message": f"请求异常：{req_err}"
        }
    except Exception as e:
        return {
            "status": -4, 
            "message": f"发生未知错误：{e}"
        }


def get_routes_as_json(start_address: str, end_address: str, travel_modes: list) -> str:
    """
    根据起点、终点和指定的出行方式列表，获取所有路线信息并返回一个JSON字符串。

    参数:
        start_address (str): 起点地址。
        end_address (str): 终点地址。
        travel_modes (list): 包含所需出行方式的列表, e.g., ['walking', 'subway', 'bus', 'taxi']。

    返回:
        str: 一个包含所有请求的出行方案的JSON格式字符串。
    """
    # 最终要返回的JSON对象
    final_json_output = {
        "query": {
            "start_address": start_address,
            "end_address": end_address,
            "requested_modes": travel_modes
        },
        "results": {}
    }

    # 1. 地址解析为坐标
    print(f"正在解析起点地址: {start_address}")
    origin_lng, origin_lat = address_to_coord(start_address)
    print(f"正在解析终点地址: {end_address}")
    dest_lng, dest_lat = address_to_coord(end_address)

    if not all([origin_lng, origin_lat, dest_lng, dest_lat]):
        final_json_output["error"] = "地址解析失败，无法进行路径规划。请检查地址是否准确。"
        return json.dumps(final_json_output, indent=4, ensure_ascii=False)

    # 2. 根据请求的 travel_modes 调用相应的API并填充 results
    if 'walking' in travel_modes:
        print("正在获取步行路线...")
        walking_info = get_walking_route_info(origin_lng, origin_lat, dest_lng, dest_lat)
        if walking_info and walking_info["status"] == 0:
            final_json_output["results"]["walking"] = walking_info
        else:
            final_json_output["results"]["walking"] = {"error": walking_info.get("message", "获取步行路线失败。")}
            
    if any(mode in travel_modes for mode in ['subway', 'bus', 'taxi']):
        print("正在获取公共交通和出租车路线...")
        all_transit_taxi_info = get_navigation_route2(origin_lng, origin_lat, dest_lng, dest_lat)

        if all_transit_taxi_info["status"] != 0:
            # 如果公共交通API调用失败，统一记录错误
            error_msg = all_transit_taxi_info.get('message', '获取公共交通/出租车信息失败。')
            if 'subway' in travel_modes:
                final_json_output["results"]["subway"] = {"error": error_msg}
            if 'bus' in travel_modes:
                final_json_output["results"]["bus"] = {"error": error_msg}
            if 'taxi' in travel_modes:
                final_json_output["results"]["taxi"] = {"error": error_msg}
        else:
            # 根据请求的模式填充数据
            if 'subway' in travel_modes:
                if all_transit_taxi_info["subway_route"]:
                    final_json_output["results"]["subway"] = all_transit_taxi_info["subway_route"]
                else:
                    final_json_output["results"]["subway"] = {"error": "未找到包含地铁的路线方案。"}
            
            if 'bus' in travel_modes:
                if all_transit_taxi_info["bus_route"]:
                    final_json_output["results"]["bus"] = all_transit_taxi_info["bus_route"]
                else:
                    final_json_output["results"]["bus"] = {"error": "未找到纯公交路线方案。"}

            if 'taxi' in travel_modes:
                if all_transit_taxi_info["taxi_info"]:
                    final_json_output["results"]["taxi"] = all_transit_taxi_info["taxi_info"]
                else:
                    final_json_output["results"]["taxi"] = {"error": "未找到出租车信息。"}
    
    # 3. 将字典转换为JSON字符串
    return json.dumps(final_json_output, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # --- 示例用法 ---
    
    # 1. 定义起点、终点和想要的出行方式
    start_address = "北京西站"
    end_address = "良乡大学城" 
    
    # 你可以自由组合想要的模式：'walking', 'subway', 'bus', 'taxi'
    modes_to_get = ['walking', 'subway', 'bus', 'taxi'] 

    # 2. 调用新函数获取JSON结果
    print(f"正在为 '{start_address}' 到 '{end_address}' 获取路线...")
    travel_plan_json = get_routes_as_json(start_address, end_address, modes_to_get)

    # 3. 打印JSON结果到控制台
    print("\n--- 生成的JSON结果 ---")
    print(travel_plan_json)

    # 4. 将JSON字符串保存到文件
    output_directory = "./output" # 定义输出目录
    output_filename = "travel_plan_full.json"
    output_filepath = os.path.join(output_directory, output_filename)

    # 创建目录（如果不存在）
    os.makedirs(output_directory, exist_ok=True)

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(travel_plan_json)
        print(f"\n✅ 出行方案已成功保存到: {output_filepath}")
    except IOError as e:
        print(f"\n❌ 保存文件时出错: {e}")

    print("\n" + "="*30 + "\n")

    # 示例2: 只获取步行路线
    start_address_walk = "天安门广场"
    end_address_walk = "故宫博物院"
    modes_only_walk = ['walking']
    print(f"正在为 '{start_address_walk}' 到 '{end_address_walk}' 获取步行路线...")
    walking_plan_json = get_routes_as_json(start_address_walk, end_address_walk, modes_only_walk)
    print("\n--- 生成的步行路线JSON结果 ---")
    print(walking_plan_json)
    output_filepath_walk = os.path.join(output_directory, "travel_plan_walking_only.json")
    try:
        with open(output_filepath_walk, "w", encoding="utf-8") as f:
            f.write(walking_plan_json)
        print(f"\n✅ 步行方案已成功保存到: {output_filepath_walk}")
    except IOError as e:
        print(f"\n❌ 保存文件时出错: {e}")