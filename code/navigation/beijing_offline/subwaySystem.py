# subwaySystem.py
# -*- coding: utf-8 -*-
# @Author: MarkJiYuan
# @Date:   2019-05-23 10:39:42
# @Last Modified by:   MarkJiYuan
# @email: zhengjiy16@163.com
# @Last Modified time: 2019-05-31 23:56:47
# @Abstract: 生成station对象

import sys
import math
import time
import json 
from beijing_offline.Correction_station import Correction_Station # 导入纠错类

# --- Helper Class Definitions ---

class Station:
    """Represents a subway station."""
    def __init__(self, name):
        self.name = name
        self.line = [] # List of lines it belongs to
        self.next_station = {} # Adjacent stations and time costs {station_name: time_cost}
        self.location = None # Geographic location (latitude, longitude)

    def add_line(self, line_name):
        if line_name not in self.line:
            self.line.append(line_name)

    def add_next_station(self, next_station_name, time_cost=0):
        self.next_station[next_station_name] = time_cost

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Station('{self.name}')"

class Node:
    """Node class for A* algorithm, containing station info and path cost."""
    def __init__(self, station, subway_map):
        self.station = station  # self.station is a Station object
        self.subway_map = subway_map # Reference to the subway map
        self.parent = None # Parent node
        self.basecost = 0 # Actual cost from start to this node (time)

    def Parent(self, parent_node):
        """Sets the parent node and updates the cost from start to this node."""
        self.parent = parent_node
        # Default time cost if not specified in data
        time_cost = parent_node.station.next_station.get(self.station.name, 2) 
        self.basecost = self.parent.basecost + time_cost

    def IsRoot(self):
        """Checks if the node is the root node (start)."""
        return self.parent is None

    def GetParent(self):
        """Gets the parent node."""
        return self.parent

    def IsStation(self, other_node):
        """
        Compares if this node's station is the same as another node's (or station's) station.
        `other_node` can be another Node object or Station object.
        """
        if isinstance(other_node, Node):
            return self.station.name == other_node.station.name
        elif isinstance(other_node, Station):
            return self.station.name == other_node.name
        return False # If types don't match, return False

    def Info(self):
        """Prints node info (for debugging)."""
        print('大家好。我目前在' + self.station.name)
        if self.parent:
            print('我来自' + self.parent.station.name)
        print('我已经经过了' + str(self.basecost) + '分钟')
        print('***********************************')

    def __str__(self):
        return f"{self.station.name} {self.basecost:.2f}min"

class AStar:
    """Implements the A* algorithm to find subway routes."""

    SUBWAY_SPEED = 0.58 # km/min

    def __init__(self, subway_map, start_station_name, end_station_name):
        self.subway_map = subway_map
        # Check if start and end stations exist in the map
        if start_station_name not in self.subway_map:
            raise ValueError(f"Start station '{start_station_name}' not found in subway map.")
        if end_station_name not in self.subway_map:
            raise ValueError(f"End station '{end_station_name}' not found in subway map.")

        self.start_station = self.subway_map[start_station_name]
        self.end_station = self.subway_map[end_station_name]
        
        start_node = Node(self.start_station, self.subway_map)
        self.open_set = [start_node] # List of nodes to explore
        self.close_set = [] # List of explored nodes
        # Record the lowest cost for visited stations for optimization
        self.stations_have_been = {start_node.station.name: start_node.basecost}

    def GetDistance(self, station1, station2) -> int:
        """
        Calculates the straight-line distance (meters) between two stations 
        using the Haversine formula.
        """
        location1 = station1.location
        location2 = station2.location
        if not location1 or not location2:
            return 0 # If geographic location data is missing, return 0

        R = 6371  # Earth radius in kilometers
        lat1, lon1 = math.radians(location1[0]), math.radians(location1[1])
        lat2, lon2 = math.radians(location2[0]), math.radians(location2[1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance_km = R * c
        return int(distance_km * 1000) # Return in meters

    def HeuristicCost(self, node) -> float:
        """
        Calculates the heuristic cost (estimated time) from the current node to the end.
        """
        station_now = node.station
        distance_meters = self.GetDistance(station_now, self.end_station)
        distance_km = distance_meters / 1000
        return distance_km / self.SUBWAY_SPEED

    def TotalCost(self, node) -> float:
        """
        Calculates the total cost F = G + H, where G is the actual cost and H is the heuristic cost.
        """
        return node.basecost + self.HeuristicCost(node)

    def OnSameLine(self, station1_obj, station2_obj) -> str:
        """
        Checks if there's a common line between two stations and returns one common line name.
        If there are multiple common lines, returns the first one. Returns None if no common line.
        """
        common_lines = list(set(station1_obj.line) & set(station2_obj.line))
        return common_lines[0] if common_lines else None


    def ExpandNode(self, node) -> list:
        """
        Expands the given node, generating all its adjacent child nodes.
        Also calculates transfer penalties.
        """
        child_node_list = []
        for next_station_name in node.station.next_station:
            next_station = self.subway_map[next_station_name]
            child_node = Node(next_station, self.subway_map)
            child_node.Parent(node) # This sets basecost based on inter-station travel time

            transfer_penalty = 0
            if node.parent: # Only consider transfers for non-start nodes
                # Get the line arriving at the current station
                line_to_current = self.OnSameLine(node.parent.station, node.station)
                # Get the line departing from the current station to the next
                line_from_current = self.OnSameLine(node.station, child_node.station)

                # If arriving and departing lines are different, it's a transfer
                if line_to_current and line_from_current and line_to_current != line_from_current:
                    transfer_penalty = 8 # Transfer penalty, e.g., 8 minutes
                # If there's no common line, also consider it a transfer (defensive programming)
                elif not line_from_current:
                    transfer_penalty = 8

            child_node.basecost += transfer_penalty
            child_node_list.append(child_node)
        return child_node_list

    def TraceRoute(self, node):
        """
        Traces back from the end node to the start to construct the path list.
        """
        node_path = []
        while not node.IsRoot():
            node_path.insert(0, node)
            node = node.parent
        node_path.insert(0, node) # Add the start node

        station_name_path = []
        for n in node_path:
            station_name_path.append(n.station.name)

        return station_name_path

    def InsertNodeInOpenSet(self, node):
        """
        Inserts a node into the open_set, keeping it sorted by TotalCost (ascending).
        """
        this_node_cost = self.TotalCost(node)
        for i, open_node in enumerate(self.open_set):
            if this_node_cost < self.TotalCost(open_node):
                self.open_set.insert(i, node)
                return
        self.open_set.append(node) # If greater than all, append to end

    def Run(self):
        """
        Executes the A* algorithm to find the path from start to end.
        """
        while self.open_set:
            # Pop the node with the lowest total cost from open_set
            current_node = self.open_set.pop(0)

            # If the current node is the destination, path found
            if current_node.station.name == self.end_station.name: 
                return self.TraceRoute(current_node)
            
            # Add current node to close_set
            self.close_set.append(current_node)

            # Expand current node, generate its children
            new_node_list = self.ExpandNode(current_node)
            
            for new_node in new_node_list:
                # Optimization 1: If new node is already in close_set and its path cost
                # is not better than the existing path, skip
                skip_node = False
                for closed_node in self.close_set:
                    if new_node.IsStation(closed_node) and new_node.basecost >= closed_node.basecost:
                        skip_node = True
                        break
                if skip_node:
                    continue

                # Optimization 2: If new node is already in open_set, but the new path cost is lower, update it
                found_in_open_set = False
                for i, open_node in enumerate(self.open_set):
                    if new_node.IsStation(open_node):
                        found_in_open_set = True
                        if new_node.basecost < open_node.basecost:
                            self.open_set.pop(i) # Remove old entry
                            self.InsertNodeInOpenSet(new_node) # Insert updated node at correct position
                        break
                
                if not found_in_open_set:
                    # If it's a new node, or though in open_set but new path is better and not updated yet, insert
                    self.InsertNodeInOpenSet(new_node)
        return None # If open_set is empty and no path found, return None

class Path:
    """Processes and formats subway path information."""
    def __init__(self, path_stations, line_map, subway_map):
        self.path_stations = path_stations
        self.line_map = line_map
        self.subway_map = subway_map

    def get_route_details(self) -> str:
        """
        Generates a complete route description string, including transfer info and station counts per segment.
        """
        route_messages = []
        total_stations_in_path = len(self.path_stations)
        
        if total_stations_in_path == 0:
            return "未找到路线。"
        if total_stations_in_path == 1:
            return f"您已在目的地站：{self.path_stations[0]}。"

        # Get the first and second station in the A* path to determine the initial line
        start_station_obj = self.subway_map[self.path_stations[0]]
        next_station_after_start_obj = self.subway_map[self.path_stations[1]]
        
        # Find the common line between the start and the next station as the initial riding line
        current_segment_line = None
        common_lines_initial = list(set(start_station_obj.line) & set(next_station_after_start_obj.line))
        if common_lines_initial:
            current_segment_line = common_lines_initial[0]
        else: # If start and second station have no common line, implies an issue or incomplete data
            # This should rarely happen, but for robustness, try to pick any line from the start station
            if start_station_obj.line:
                current_segment_line = start_station_obj.line[0]

        # Add the departure message with a period at the end
        route_messages.append(f"您的出发地为{current_segment_line if current_segment_line else '未知线路'}的{self.path_stations[0]}站。")

        segment_start_index = 0 # Index of the starting station for the current segment

        for i in range(total_stations_in_path - 1):
            current_station_name = self.path_stations[i]
            next_station_name = self.path_stations[i+1]
            
            current_station_obj = self.subway_map[current_station_name]
            next_station_obj = self.subway_map[next_station_name]

            # Find the common line between the current and next station
            common_line_between_current_and_next = self._get_common_line(current_station_obj, next_station_obj)

            # Determine if a transfer is needed
            # Transfer condition:
            # 1. No common line between current and next station (A* should ideally avoid this, but as error prevention)
            # 2. The common line between current and next station is different from the current riding line
            is_transfer_point = False
            if common_line_between_current_and_next is None:
                is_transfer_point = True # Should ideally not happen, but if it does, treat as transfer
            elif current_segment_line and common_line_between_current_and_next != current_segment_line:
                is_transfer_point = True
            
            # If it's a transfer point
            if is_transfer_point:
                # Record the current segment's travel info
                segment_stations_count = i - segment_start_index + 1 # Includes start and current station
                route_messages.append(f"乘坐{current_segment_line}线{segment_stations_count}站后，在{current_station_name}从换乘到{common_line_between_current_and_next if common_line_between_current_and_next else '未知线路'}。")
                
                # Update for the new segment's start info
                segment_start_index = i + 1 # New segment starts from the next station
                current_segment_line = common_line_between_current_and_next
            
        # Add the destination station description
        # The last segment's description needs to be handled separately as it doesn't trigger a transfer
        final_segment_stations_count = total_stations_in_path - segment_start_index
        route_messages.append(f"乘坐{current_segment_line}线{final_segment_stations_count}站后，您将会到达位于{current_segment_line}线的{self.path_stations[-1]}站。")
        route_messages.append(f"全长{total_stations_in_path}站。")

        return "".join(route_messages) # Join all messages without any separators

    def _get_common_line(self, station1_obj, station2_obj) -> str:
        """
        Helper method: Gets the common line between two stations.
        """
        common_lines = list(set(station1_obj.line) & set(station2_obj.line))
        return common_lines[0] if common_lines else None


class Subway_Navigator:
    """Main class for Beijing Subway Navigation."""
    def __init__(self):
        # Define data file paths
        # Assuming subwaySystem.py and Correction_station.py are in Beijing_Subway_System/
        # and data files are in Beijing_Subway_System/data/
        subway_system_base_path = "/data/Zoo2/Beijing_Subway_System" 
        self.subway_map_path = f"{subway_system_base_path}/subway_map.txt" 
        self.station_geo_path = f'{subway_system_base_path}/data/station_geo.txt'
        self.time_cost_path = f'{subway_system_base_path}/data/timecost_list.txt'
        self.price_path = f'{subway_system_base_path}/data/price.txt'
        
        self.subway_map = {} # Stores all station objects
        self._load_subway_map(self.subway_map_path)
        self.ExceptionAirportLine() # Handle special Airport Line connections
        self._load_line_map(self.subway_map_path)
        self._load_station_geo(self.station_geo_path)
        self._load_time_costs(self.time_cost_path)
        self._load_price_map(self.price_path)

        # Initialize station correction tool
        self.correction_tool = Correction_Station() 
        # Ensure the correction tool uses the correct subway map file path
        self.correction_tool.stdlist_path = self.subway_map_path
        self.correction_tool.standard_stations = self.correction_tool.load_standard_stations()


    def _load_subway_map(self, subway_map_path):
        """Loads subway map and creates station objects from file."""
        with open(subway_map_path, 'r', encoding='utf-8') as f:
            texts = f.read().split()

        length = len(texts)
        current_line_name = ""
        for i in range(length):
            station_name = texts[i]
            if station_name.startswith('&'):
                current_line_name = station_name[1:]
            else:
                if station_name not in self.subway_map:
                    station = Station(station_name)
                    self.subway_map[station_name] = station
                station = self.subway_map[station_name]
                station.add_line(current_line_name)
                
                # Add connection to the next station
                if i < length - 1:
                    next_item = texts[i+1]
                    if not next_item.startswith('&'):
                        station.add_next_station(next_item)
                
                # Add connection to the previous station (ensure bidirectional connections)
                if i > 0:
                    prev_item = texts[i-1]
                    if not prev_item.startswith('&'):
                        if prev_item not in self.subway_map:
                            self.subway_map[prev_item] = Station(prev_item)
                        # Ensure previous station also connects back to current station
                        self.subway_map[prev_item].add_next_station(station_name)


    def _load_line_map(self, subway_map_path):
        """Loads line-to-station mapping from file."""
        self.line_map = {}
        with open(subway_map_path, 'r', encoding='utf-8') as f:
            current_line_name = ""
            for line_content in f.readlines():
                line_content = line_content.strip()
                if line_content:
                    if line_content.startswith('&'):
                        current_line_name = line_content[1:]
                        self.line_map[current_line_name] = []
                    else:
                        station_name = line_content
                        if current_line_name in self.line_map:
                            self.line_map[current_line_name].append(station_name)

    def _load_station_geo(self, station_geo_path):
        """Loads station geographic locations from file."""
        with open(station_geo_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                s = line.split()
                if len(s) == 2:
                    station_name = s[0]
                    location = s[1].split(',')
                    if station_name in self.subway_map:
                        self.subway_map[station_name].location = (float(location[0]), float(location[1]))

    def _load_time_costs(self, time_cost_path):
        """Loads travel times between stations from file."""
        with open(time_cost_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                message = line.strip().split()
                if len(message) == 3:
                    start, end, timeused = message
                    if start in self.subway_map and end in self.subway_map:
                        self.subway_map[start].next_station[end] = int(timeused)
                        # Ensure bidirectional time costs
                        if start not in self.subway_map[end].next_station:
                            self.subway_map[end].next_station[start] = int(timeused)

    def _load_price_map(self, price_path):
        """Loads prices between stations from file."""
        self.price_map = {}
        with open(price_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                s = line.split()
                if len(s) == 3:
                    start_station_name, end_station_name, price = s
                    combine_forward = start_station_name + '-' + end_station_name
                    combine_backward = end_station_name + '-' + start_station_name
                    self.price_map[combine_forward] = price
                    self.price_map[combine_backward] = price

    def find_nearest_station(self, lat, lon):
        """
        Finds the nearest subway station to given latitude and longitude.
        """
        best, best_d2 = None, float('inf')
        for name, station in self.subway_map.items():
            if station.location:
                x, y = station.location
                d2 = (x - lat)**2 + (y - lon)**2
                if d2 < best_d2:
                    best_d2, best = d2, name
        return best

    def ExceptionAirportLine(self):
        """
        Handles special connection logic for Airport Line.
        """
        if all(station_name in self.subway_map for station_name in ['2号航站楼', '3号航站楼', '三元桥']):
            station2 = self.subway_map['2号航站楼']
            if '3号航站楼' in station2.next_station:
                del station2.next_station['3号航站楼']
            station2.add_next_station('三元桥')

            station3 = self.subway_map['3号航站楼']
            if '三元桥' in station3.next_station:
                del station3.next_station['三元桥']
            
    def GetPrice(self, start_station_name, end_station_name):
        """
        Gets the price between two stations.
        """
        combine = start_station_name + '-' + end_station_name
        price = self.price_map.get(combine)
        if price is None:
            combine = end_station_name + '-' + start_station_name
            price = self.price_map.get(combine)
        return price if price else "N/A"

    def Route_Planning(self, start_station_input, end_station_input):
        """
        Plans subway route, including station correction and route generation.
        """
        # 1. Correct user-entered station names
        corrected_inputs = self.correction_tool.correct_station_names_pinyin_first(
            [start_station_input, end_station_input]
        )
        corrected_start_station = corrected_inputs[0]
        corrected_end_station = corrected_inputs[1]

        # Check correction results
        if corrected_start_station == "error_station_name":
            return {"content": "我没有听清出发站名，请重新试试。"}
        if corrected_end_station == "error_station_name":
            return {"content": "我没有听清到达站名，请重新试试。"}

        try:
            # 2. Execute A* algorithm to find path
            a = AStar(self.subway_map, corrected_start_station, corrected_end_station)
            path_station_names = a.Run()

            result = {}
            if path_station_names:
                # 3. Format route information as a string
                p = Path(path_station_names, self.line_map, self.subway_map)
                route_string = p.get_route_details()
                
                result["content"] = route_string
                
                # You can add other information here as needed, e.g., price, total time
                # price = self.GetPrice(corrected_start_station, corrected_end_station)
                # result["price"] = price
                # result["total_estimated_time_minutes"] = round(a.TotalCost(a.open_set[0]), 2) if a.open_set else 0 
                # result["transfer_count"] = route_string.count("换乘") 

            else:
                result = {"content": f"未找到从 '{corrected_start_station}' 到 '{corrected_end_station}' 的路线。"}
            
            return result

        except ValueError as e:
            return {"content": f"我没有听清，请重新输入"}


# --- Main Program Entry ---

if __name__=='__main__':
    # Example: Set correct sys.path to ensure Correction_station can be imported
    # If subwaySystem.py and Correction_station.py are in the same directory, typically not needed
    # If Correction_station.py is in a parent directory, you might need something like:
    # sys.path.append('/data/Zoo2/Beijing_Subway_System/') # Adjust based on your actual path

    navigator = Subway_Navigator()
    
    start_station_name = input("请输入出发站：")
    end_station_name = input("请输入目的地站：")
    
    # Call route planning and get JSON result
    route_output = navigator.Route_Planning(start_station_name, end_station_name)
    
    # Print JSON result, ensuring Chinese characters are displayed correctly and formatted
    print(json.dumps(route_output, ensure_ascii=False, indent=4))