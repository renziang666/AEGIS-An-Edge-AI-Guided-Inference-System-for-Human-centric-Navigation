import json
import sys
import os

# Add the directory containing navigate_new.py to the Python path
# This assumes navigate_new.py is in the same directory as json_text.py
sys.path.append(os.path.dirname(__file__))

import navigate_new

# --- Helper function: Summarizes JSON walking route data with detailed steps ---
def summarize_walking_route(json_data):
    """
    Summarizes a walking route from a JSON object into a single sentence,
    including detailed steps.

    Args:
        json_data (dict): A Python dictionary containing the route information.

    Returns:
        str: A single-sentence summary of the route, with step details.
    """
    if json_data["status"] == "success":
        origin = json_data["origin_address"]
        destination = json_data["destination_address"]
        total_distance_meters = json_data["route"]["total_distance_meters"]
        total_duration_seconds = json_data["route"]["total_duration_seconds"]
        steps = json_data["route"]["steps"]

        # Convert total_distance_meters to kilometers
        total_distance_km = total_distance_meters / 1000

        # Convert total_duration_seconds to hours and minutes
        hours = total_duration_seconds // 3600
        minutes = (total_duration_seconds % 3600) // 60

        # Format the duration string
        duration_str = ""
        if hours > 0:
            duration_str += f"{hours}小时"
        if minutes > 0:
            duration_str += f"{minutes}分钟"
        if not duration_str:  # Handle cases where duration is less than a minute
            duration_str = f"{total_duration_seconds}秒"

        summary_parts = []
        summary_parts.append(f"从{origin}步行到{destination}，全程{total_distance_km:.1f}公里，预计耗时{duration_str}。")
        
        # Build the detailed instructions string without newlines between steps
        detailed_instructions_list = []
        for step in steps:
            instructions = step.get("instructions", "无具体指令")
            road_name = step.get("road_name", "").strip()

            if road_name:
                detailed_instructions_list.append(f"{instructions} (路名: {road_name})")
            else:
                detailed_instructions_list.append(f"{instructions}")
        
        # Join all detailed instructions with a period and a space, and add the "详细路线如下：" prefix
        summary_parts.append("详细路线如下：" + "。".join(detailed_instructions_list) + "。") # Added final period

        return "\n".join(summary_parts)
    else:
        return f"获取步行路线信息失败：{json_data.get('message', '未知错误')}"


# --- Helper function: Summarizes JSON public transit route data with detailed steps ---
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
                    detailed_transit_instructions.append(f"乘坐{step.get('line_name')} ({step_type})，从{on_station}站上车，到{off_station}站下车")
                elif instructions:
                    # General instructions for walking parts of transit
                    detailed_transit_instructions.append(f"{instructions} ({step_type})")
                else:
                    detailed_transit_instructions.append(f"{step_type} - 无详细信息")
        
        summary_parts.append("。".join(detailed_transit_instructions) + "。") # Join all detailed transit instructions

        return "\n".join(summary_parts)
    else:
        return f"获取公交路线信息失败：{json_data.get('message', '未知错误')}"

# --- Helper function: Summarizes JSON taxi route data into a single sentence ---
def summarize_taxi_route(json_data):
    """
    Summarizes a taxi route from a JSON object into a single sentence.

    Args:
        json_data (dict): A Python dictionary containing the taxi information.

    Returns:
        str: A single-sentence summary of the taxi route.
    """
    if json_data["status"] == "success":
        origin = json_data["origin_address"]
        destination = json_data["destination_address"]
        taxi_info = json_data["taxi_info"]

        total_distance_km = taxi_info["total_distance_meters"] / 1000
        total_duration_minutes = round(taxi_info["total_duration_seconds"] / 60)
        remark = taxi_info.get("remark", "")

        price_details = []
        if taxi_info.get("price_details"):
            for detail in taxi_info["price_details"]:
                price_details.append(f"{detail.get('description', '')}，预计费用：{detail.get('total_price', 'N/A')}元")

        price_str = "；".join(price_details) if price_details else "无详细费用信息。"
        
        summary = (
            f"从{origin}乘坐出租车到{destination}，全程{total_distance_km:.1f}公里，"
            f"预计耗时{total_duration_minutes}分钟。{remark} {price_str}"
        )
        return summary
    else:
        return f"获取出租车路线信息失败：{json_data.get('message', '未知错误')}"


# --- Example Usage ---
if __name__ == "__main__":
    origin = "天安门"
    destination = "北京南站"

    # --- Get and summarize walking route ---
    # print(f"正在查询并总结从 '{origin}' 到 '{destination}' 的步行路线...\n")
    # walking_output = navigate_new.navigate_walking(origin, destination)
    # walking_summary = summarize_walking_route(walking_output)
    # print("--- 步行路线总结 ---")
    # print(walking_summary)
    # print("\n" + "-"*30 + "\n")


    # --- Get and summarize public transit route ---
    # print(f"正在查询并总结从 '{origin}' 到 '{destination}' 的公交路线...\n")
    # transit_output = navigate_new.get_transit_routes(origin, destination)
    # transit_summary = summarize_transit_route(transit_output)
    # print("--- 公交路线总结 ---")
    # print(transit_summary)
    # print("\n" + "-"*30 + "\n")


    # You can uncomment this section if you want to test taxi route
    # --- Get and summarize taxi route ---
    print(f"正在查询并总结从 '{origin}' 到 '{destination}' 的出租车路线...\n")
    taxi_output = navigate_new.navigate_taxi(origin, destination)
    taxi_summary = summarize_taxi_route(taxi_output)
    print("--- 出租车路线总结 ---")
    print(taxi_summary)