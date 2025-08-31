import requests
import json
from urllib.parse import quote
from urllib.request import urlopen
import re

# --- 配置 ---
# 这是你的百度API密钥。请务必替换为你的实际AK！
BAIDU_AK = "vPcIiXRLrTNjzCPLg0u95sNybfCp64DT" # 请替换为你的真实AK！

def summarize_transit_route(json_data):
    """
    Summarizes a transit route from a JSON object into a concise sentence,
    including detailed steps for each leg of the journey.

    Args:
        json_data (dict): A Python dictionary containing the route information.

    Returns:
        str: A single-sentence summary of the transit route, with step details.
    """
    if json_data["status"] == "success":
        origin = json_data["origin_address"]
        destination = json_data["destination_address"]
        route = json_data["route"]
        
        total_distance_km = route["total_distance_meters"] / 1000
        total_duration_minutes = route["total_duration_minutes"]

        summary_parts = []
        summary_parts.append(f"从{origin}乘坐公共交通到{destination}，全程{total_distance_km:.1f}公里，预计耗时{total_duration_minutes:.0f}分钟。")
        summary_parts.append("详细换乘信息如下：")

        detailed_transit_instructions = []
        if route.get("steps"):
            for step in route["steps"]:
                step_type = step.get("type", "未知类型")
                instructions = step.get("instructions", "无具体指令")
                
                if "line_name" in step:
                    # Specific formatting for bus/subway lines
                    on_station = step.get('on_station', '起点')
                    off_station = step.get('off_station', '终点')
                    detailed_transit_instructions.append(f"乘坐{step.get('line_name')} ，从{on_station}站上车，到{off_station}站下车")
                elif instructions:
                    # General instructions for walking parts of transit
                    detailed_transit_instructions.append(f"{instructions} ")
                else:
                    detailed_transit_instructions.append(f"{step_type} - 无详细信息")
        
        summary_parts.append("。".join(detailed_transit_instructions) + "。") # Join all detailed transit instructions

        return "\n".join(summary_parts)
    else:
        return f"获取公交路线信息失败：{json_data.get('message', '未知错误')}"



# --- 将地址转换为百度地图坐标系经纬度的函数 ---
def address_to_coord(address):
    """将地址转换为百度地图坐标系经纬度"""
    root_url = "http://api.map.baidu.com/geocoding/v3/"
    output = 'json'
    add = quote(address) # 对地址进行URL编码，以支持中文
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
            # 返回错误信息
            return None, {"error": f"地址解析失败: {address}", "message": temp.get('message', '未知错误')}
    except Exception as e:
        return None, {"error": f"请求地址解析API时发生错误", "message": str(e)}

# --- 获取公交路线的主函数 ---
def get_transit_routes(origin_address, destination_address):
    result_data = {
        "status": "fail",
        "message": "",
        "origin_address": origin_address,
        "destination_address": destination_address,
        "route": None
    }

    # 解析起点地址
    origin_lng, origin_lat_or_error = address_to_coord(origin_address)
    if origin_lng is None:
        result_data["message"] = origin_lat_or_error.get("error", "无法获取起点坐标") + f" - {origin_lat_or_error.get('message', '')}"
        return result_data
    origin_lat = origin_lat_or_error # 这里 origin_lat_or_error 实际上是纬度了

    # 解析终点地址
    destination_lng, destination_lat_or_error = address_to_coord(destination_address)
    if destination_lng is None:
        result_data["message"] = destination_lat_or_error.get("error", "无法获取终点坐标") + f" - {destination_lat_or_error.get('message', '')}"
        return result_data
    destination_lat = destination_lat_or_error # 这里 destination_lat_or_error 实际上是纬度了

    result_data["origin_coordinates"] = {"lng": origin_lng, "lat": origin_lat}
    result_data["destination_coordinates"] = {"lng": destination_lng, "lat": destination_lat}

    # 公交路线查询API的URL
    transit_url = "https://api.map.baidu.com/direction/v2/transit"

    params = {
        "origin": f"{origin_lat},{origin_lng}",  # 百度API要求格式为 纬度,经度
        "destination": f"{destination_lat},{destination_lng}", # 百度API要求格式为 纬度,经度
        "ak": BAIDU_AK,
        "ret_coordtype": "bd09ll" # 确保返回的坐标系是百度经纬度坐标
    }

    try:
        response = requests.get(url=transit_url, params=params)
        response.raise_for_status() # 检查HTTP响应状态码，如果不是200会抛出异常
        data = response.json()

        if data.get("status") == 0:
            result = data.get("result")
            if result and result.get("routes"):
                route = result["routes"][0] # 获取第一条路线
                
                route_info = {
                    "total_distance_meters": route.get('distance'),
                    "total_duration_minutes": round(route.get('duration') / 60, 2),
                    "steps": []
                }

                if route.get("steps"):
                    for j, step_schemes_list in enumerate(route["steps"]):
                        step_data = {}
                        step_data["step_number"] = j + 1

                        if isinstance(step_schemes_list, list) and step_schemes_list:
                            scheme_info = step_schemes_list[0] # 获取第一个方案

                            step_data["distance_meters"] = scheme_info.get('distance')
                            step_data["duration_minutes"] = round(scheme_info.get('duration') / 60, 2)
                            if scheme_info.get('instructions'):
                                step_data["instructions"] = scheme_info.get('instructions')

                            if scheme_info.get("vehicle_info") and scheme_info["vehicle_info"].get("type") == 3:
                                bus_detail = scheme_info["vehicle_info"].get("detail")
                                if bus_detail:
                                    bus_type = bus_detail.get("type")
                                    if bus_type == 0:
                                        step_data["type"] = "普通公交"
                                        step_data["line_name"] = bus_detail.get('name')
                                        step_data["on_station"] = bus_detail.get('on_station')
                                        step_data["off_station"] = bus_detail.get('off_station')
                                        # step_data["first_time"] = bus_detail.get('first_time') # 如果需要可以取消注释
                                        # step_data["last_time"] = bus_detail.get('last_time')  # 如果需要可以取消注释
                                    elif bus_type == 1:
                                        step_data["type"] = "地铁/轻轨"
                                        step_data["line_name"] = bus_detail.get('name')
                                        step_data["on_station"] = bus_detail.get('on_station')
                                        step_data["off_station"] = bus_detail.get('off_station')
                                        # step_data["first_time"] = bus_detail.get('first_time')
                                        # step_data["last_time"] = bus_detail.get('last_time')
                                    else:
                                        step_data["type"] = f"其他公交类型 ({bus_type})"
                                else:
                                    step_data["type"] = "公交详情缺失"
                            elif scheme_info.get("vehicle_info"):
                                vehicle_type_map = {
                                    1: "火车", 2: "飞机", 4: "驾车", 5: "步行", 6: "大巴"
                                }
                                step_data["type"] = vehicle_type_map.get(scheme_info["vehicle_info"]["type"], "未知交通工具")
                            else:
                                step_data["type"] = "未知交通方式"
                        else:
                            step_data["type"] = "数据解析异常"
                            step_data["raw_data"] = step_schemes_list # 记录异常数据方便调试
                        
                        route_info["steps"].append(step_data)
                
                result_data["status"] = "success"
                result_data["message"] = "路线获取成功"
                result_data["route"] = route_info
            else:
                result_data["message"] = "未找到路线信息。"
        elif data.get("status") == 1001:
            result_data["message"] = "API错误：没有公交方案。"
        elif data.get("status") == 1002:
            result_data["message"] = "API错误：不支持跨域。"
        else:
            result_data["message"] = f"API请求失败，状态码: {data.get('status')}, 消息: {data.get('message')}"
    except requests.exceptions.RequestException as e:
        result_data["message"] = f"HTTP请求失败: {e}"
    except json.JSONDecodeError:
        result_data["message"] = "API返回了无效的JSON。"
    except Exception as e:
        result_data["message"] = f"处理API响应时发生未知错误: {e}"
    
    return result_data

# --- 新增函数：获取出租车路线信息 ---
def navigate_taxi(origin_address: str, destination_address: str):

    result_data = {
        "status": "fail",
        "message": "",
        "origin_address": origin_address,
        "destination_address": destination_address,
        "taxi_info": None
    }

    origin_lng, origin_lat_or_error = address_to_coord(origin_address)
    if origin_lng is None:
        result_data["message"] = f"无法获取起点坐标: {origin_lat_or_error.get('error', '')} - {origin_lat_or_error.get('message', '')}"
        return result_data
    origin_lat = origin_lat_or_error

    dest_lng, dest_lat_or_error = address_to_coord(destination_address)
    if dest_lng is None:
        result_data["message"] = f"无法获取终点坐标: {dest_lat_or_error.get('error', '')} - {dest_lat_or_error.get('message', '')}"
        return result_data
    dest_lat = dest_lat_or_error

    result_data["origin_coordinates"] = {"lng": origin_lng, "lat": origin_lat}
    result_data["destination_coordinates"] = {"lng": dest_lng, "lat": dest_lat}

    url = "https://api.map.baidu.com/directionlite/v1/transit"
    ak = BAIDU_AK

    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "ak": ak,
    }

    try:
        response = requests.get(url=url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == 0:
            taxi_raw_info = data.get("result", {}).get("taxi")
            if taxi_raw_info:
                taxi_info_formatted = {
                    "total_distance_meters": taxi_raw_info.get('distance'),
                    "total_duration_seconds": taxi_raw_info.get('duration'),
                    "remark": taxi_raw_info.get('remark'),
                    "price_details": []
                }
                if 'detail' in taxi_raw_info and isinstance(taxi_raw_info['detail'], list):
                    for detail in taxi_raw_info['detail']:
                        taxi_info_formatted["price_details"].append({
                            "description": detail.get('desc'),
                            "km_price": detail.get('km_price'),
                            "start_price": detail.get('start_price'),
                            "total_price": detail.get('total_price')
                        })
                result_data["status"] = "success"
                result_data["message"] = "出租车路线获取成功"
                result_data["taxi_info"] = taxi_info_formatted
            else:
                result_data["message"] = "抱歉，未找到出租车信息。"
        else:
            result_data["message"] = f"API请求失败，状态码: {data.get('status')}，信息: {data.get('message')}"
    except requests.exceptions.RequestException as e:
        result_data["message"] = f"网络请求错误或HTTP状态码异常: {e}"
    except ValueError as e:
        result_data["message"] = f"JSON解析错误: {e}"
    except Exception as e:
        result_data["message"] = f"发生未知错误: {e}"
    
    return result_data

# --- 新增函数：获取步行路线信息 ---
def navigate_walking(origin_address: str, destination_address: str):

    result_data = {
        "status": "fail",
        "message": "",
        "origin_address": origin_address,
        "destination_address": destination_address,
        "route": None
    }

    origin_lng, origin_lat_or_error = address_to_coord(origin_address)
    if origin_lng is None:
        result_data["message"] = f"无法获取起点坐标: {origin_lat_or_error.get('error', '')} - {origin_lat_or_error.get('message', '')}"
        return result_data
    origin_lat = origin_lat_or_error

    dest_lng, dest_lat_or_error = address_to_coord(destination_address)
    if dest_lng is None:
        result_data["message"] = f"无法获取终点坐标: {dest_lat_or_error.get('error', '')} - {dest_lat_or_error.get('message', '')}"
        return result_data
    dest_lat = dest_lat_or_error

    result_data["origin_coordinates"] = {"lng": origin_lng, "lat": origin_lat}
    result_data["destination_coordinates"] = {"lng": dest_lng, "lat": dest_lat}

    ak = BAIDU_AK
    
    url = "https://api.map.baidu.com/direction/v2/walking"
    params = {
        "origin": f"{origin_lat},{origin_lng}",
        "destination": f"{dest_lat},{dest_lng}",
        "ak": ak,
    }

    try:
        response = requests.get(url=url, params=params, timeout=10) 
        response.raise_for_status() 

        data = response.json()
        
        if data.get("status") == 0:
            if data["result"] and data["result"]["routes"]:
                route_raw = data["result"]["routes"][0]
                
                route_info_formatted = {
                    "total_distance_meters": route_raw.get('distance', 'N/A'),
                    "total_duration_seconds": route_raw.get('duration', 'N/A'),
                    "steps": []
                }
                
                for i, step in enumerate(route_raw["steps"]):
                    instructions = step.get("instructions", "无具体指令")
                    name = step.get("name", "").strip() 
                    
                    clean_instructions = re.sub(r'<[^>]+>', '', instructions)
                    
                    step_detail = {
                        "step_number": i + 1,
                        "instructions": clean_instructions
                    }
                    if name:
                        step_detail["road_name"] = name
                    
                    route_info_formatted["steps"].append(step_detail)
                
                result_data["status"] = "success"
                result_data["message"] = "步行路线获取成功"
                result_data["route"] = route_info_formatted
            else:
                result_data["message"] = "未找到可用的步行路线。请检查起点和终点是否有效或距离过远。"
        else:
            result_data["message"] = f"API请求失败，状态码：{data.get('status')}，错误信息：{data.get('message', '未知错误。')}"
            
    except requests.exceptions.HTTPError as http_err:
        result_data["message"] = f"网络请求HTTP错误：{http_err} (状态码: {response.status_code})"
    except requests.exceptions.ConnectionError as conn_err:
        result_data["message"] = f"网络连接错误：请检查您的网络连接或API地址是否正确。{conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        result_data["message"] = f"请求超时：在规定时间内未能从服务器获取响应。{timeout_err}"
    except requests.exceptions.RequestException as req_err:
        result_data["message"] = f"请求发生异常：{req_err}"
    except json.JSONDecodeError as json_err:
        result_data["message"] = f"JSON解析错误：API返回的数据无法解析。{json_err}"
    except Exception as e:
        result_data["message"] = f"发生未知错误：{e}"
    
    return result_data


    # --- 示例使用 ---
if __name__ == "__main__":
    import time # 引入时间模块

    origin_address = "天安门"
    destination_address = "北京南站"

    # ==========================================================
    # 1. 测试公交路线查询 (get_transit_routes) 的耗时
    # ==========================================================
    print(f"正在查询从 '{origin_address}' 到 '{destination_address}' 的公交路线...")
    
    start_time_transit = time.perf_counter() # <--- 开始计时
    transit_output = get_transit_routes(origin_address, destination_address)
    end_time_transit = time.perf_counter() # <--- 结束计时
    
    elapsed_time_transit_ms = (end_time_transit - start_time_transit) * 1000 # <--- 计算耗时（毫秒）

    print("--- 公交路线结果 ---")
    print(json.dumps(transit_output, indent=2, ensure_ascii=False))
    print(f"\n[性能测试] get_transit_routes 函数总耗时: {elapsed_time_transit_ms:.2f} ms\n")


    # ==========================================================
    # 2. 测试出租车路线查询 (navigate_taxi) 的耗时
    # ==========================================================
    print(f"正在查询从 '{origin_address}' 到 '{destination_address}' 的出租车路线...")

    start_time_taxi = time.perf_counter() # <--- 开始计时
    taxi_output = navigate_taxi(origin_address, destination_address)
    end_time_taxi = time.perf_counter() # <--- 结束计时

    elapsed_time_taxi_ms = (end_time_taxi - start_time_taxi) * 1000 # <--- 计算耗时（毫秒）

    print("--- 出租车路线结果 ---")
    print(json.dumps(taxi_output, indent=2, ensure_ascii=False))
    print(f"\n[性能测试] navigate_taxi 函数总耗时: {elapsed_time_taxi_ms:.2f} ms\n")


    # ==========================================================
    # 3. 测试步行路线查询 (navigate_walking) 的耗时
    # ==========================================================
    print(f"正在查询从 '{origin_address}' 到 '{destination_address}' 的步行路线...")
    
    start_time_walking = time.perf_counter() # <--- 开始计时
    walking_output = navigate_walking(origin_address, destination_address)
    end_time_walking = time.perf_counter() # <--- 结束计时

    elapsed_time_walking_ms = (end_time_walking - start_time_walking) * 1000 # <--- 计算耗时（毫秒）

    print("--- 步行路线结果 ---")
    print(json.dumps(walking_output, indent=2, ensure_ascii=False))
    print(f"\n[性能测试] navigate_walking 函数总耗时: {elapsed_time_walking_ms:.2f} ms\n")