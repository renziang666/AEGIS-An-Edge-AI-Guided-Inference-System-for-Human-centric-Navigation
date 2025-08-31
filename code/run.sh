#!/bin/bash

# ===================================================================================
#
# Copyright (c) 2025 [Ziang Ren]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ===================================================================================
#
# Smart Cane Project - Parallel Service Launcher
# Version: 3.0 (GitHub Ready)
#
# Description:
# This script launches all the necessary Python microservices for the Smart Cane
# project in parallel. It handles:
# 1. Running each service as a background process.
# 2. Piping the output of each service to both the console and a dedicated log file.
# 3. Sending an alert via Redis if a critical service fails.
# 4. Gracefully shutting down all child processes on exit (Ctrl+C).
#
# Usage:
# 1. IMPORTANT: Activate your Python virtual environment first.
#    e.g., source /path/to/your/venv/bin/activate
# 2. Grant execution permissions to the script:
#    chmod +x start_services.sh
# 3. Run the script from the project's root directory:
#    ./start_services.sh
# 4. Log files will be created in the 'logs' directory.
#
# ===================================================================================

echo "🚀 Starting all services..."
echo "The output of all services will be displayed on the current screen."
echo "Individual logs for each service will be saved in the 'logs' directory."
echo "Press Ctrl+C to terminate this script and attempt to shut down all background services."
echo "--------------------------------------------------------"


function send_error_alert() {
    # $1 是传入函数的第一个参数，即服务名称
    SERVICE_NAME=$1
    echo "🚨 CRITICAL: 服务 '$SERVICE_NAME' 已意外退出！正在发送语音警报..."
    
    # 定义要发送的JSON消息
    ERROR_MESSAGE='{
        "action": "play",
        "path": "/data/preaudio/004.wav",
        "key": 1
    }'

    # 使用 redis-cli 发送消息
    # 添加 -h 和 -p 参数可以确保连接到正确的Redis实例
    redis-cli -h 127.0.0.1 -p 6379 PUBLISH "audio:playcommand" "$ERROR_MESSAGE"
}

# --- 配置区域: 请根据你的实际路径修改 ---

# 新增: 日志文件存储目录 (Log file storage directory)
LOG_DIR="./logs"

# 服务A: 麦克风与KWS服务 (Microphone Service)
VENV_Microphone="./env/rza_voice_env"
PATH_Microphone="./hardware/microphone"
SCRIPT_Microphone="./hardware/microphone/microphone_service.py"

# 服务B: 手柄服务 (Gamepad Service)
VENV_Gamepad="./env/hardware_env"
PATH_Gamepad="./hardware/gamepad"
SCRIPT_Gamepad="./hardware/gamepad/gamepad_service.py"

# 服务C: 扬声器服务 (Speaker Service)
VENV_Speaker="./env/hardware_env"
PATH_Speaker="./hardware/speaker"
SCRIPT_Speaker="./hardware/speaker/speaker_service.py"

# 服务D: 语音转文字服务 (ASR Service)
VENV_ASR="./env/rza_voice_env"
PATH_ASR="./ASR"
SCRIPT_ASR="./ASR/asr_service.py"

# 服务E: 导航服务 (Navigation Service)
VENV_Navigation="./env/TTSenv"
PATH_Navigation="./navigation"
SCRIPT_Navigation="./navigation/navigate_service.py"

# 服务F: 自然语言处理服务 (Qwen LLM Service)
VENV_QWEN="./env/qwenenv"
PATH_QWEN="./qwenvl25"
SCRIPT_QWEN="./qwen_vl/QwenVLClients.py"

# 服务G: 文本转语音服务 (TTS Service)
VENV_TTS="./env/TTSenv"
PATH_TTS="./TTS"
SCRIPT_TTS="./TTS/python/TTS_service.py"

# 服务H: 障碍物检测服务 (Vision1 Service)
VENV_Vision1="./env/sophon-demo/sample/YOLOv8_plus_det/python/myenv"
PATH_Vision1="./vision/Yolo_v2"
SCRIPT_Vision1="./vision/Yolo_v2/vision_clients.py"

# 服务K: RAG服务 (RAG Service)
VENV_Rag="./env/qwenenv"
PATH_Rag="./RAG"
SCRIPT_Rag="./RAG/RAG_services.py"


# --- 准备工作: 创建日志目录 ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")


# --- 启动逻辑: 使用子shell并行运行 ---

# 启动服务A: 麦克风与KWS服务
# 启动服务A: 麦克风与KWS服务 (最高优先级 cpu资源更多)
(
  echo "[PID: $$] 🚀 正在启动 麦克风与KWS服务 (A) [高优先级]..."
  source "$VENV_Microphone/bin/activate"
  cd "$PATH_Microphone"
  # 使用 nice 命令提升优先级。-n -10 表示优先级很高
  # 注意: 运行此脚本可能需要 sudo 权限才能使用负的nice值
  nice -n -10 python3 "$SCRIPT_Microphone" || send_error_alert "麦克风服务(A)"
  echo "[PID: $$] ✅ (A) 麦克风服务已执行完毕。"
) &

# 启动服务B: 手柄服务
LOG_B="${LOG_DIR}/B_gamepad_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 手柄服务 (B)... 日志: ${LOG_B}"
  source "$VENV_Gamepad/bin/activate"
  cd "$PATH_Gamepad"
  python3 -u "$SCRIPT_Gamepad" 2>&1 | tee "$LOG_B" || send_error_alert "手柄服务(B)" # <--- 修改点
  echo "[PID: $$] ✅ (B) 手柄服务已执行完毕。"
) &

# 启动服务C: 扬声器服务
LOG_C="${LOG_DIR}/C_speaker_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 扬声器服务 (C)... 日志: ${LOG_C}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_Speaker/bin/activate"
  cd "$PATH_Speaker"
  python3 -u "$SCRIPT_Speaker" 2>&1 | tee "$LOG_C"
  echo "[PID: $$] ✅ (C) 扬声器服务已执行完毕。"
) &

# 启动服务D: 语音转文字服务
LOG_D="${LOG_DIR}/D_asr_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 语音转文字服务 (D)... 日志: ${LOG_D}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_ASR/bin/activate"
  cd "$PATH_ASR"
  python3 -u "$SCRIPT_ASR" 2>&1 | tee "$LOG_D"
  echo "[PID: $$] ✅ (D) 语音转文字服务已执行完毕。"
) &

# 启动服务E: 导航服务
LOG_E="${LOG_DIR}/E_navigation_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 导航服务 (E)... 日志: ${LOG_E}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_Navigation/bin/activate"
  cd "$PATH_Navigation"
  python3 -u "$SCRIPT_Navigation" 2>&1 | tee "$LOG_E"
  echo "[PID: $$] ✅ (E) 导航服务已执行完毕。"
) &

# 启动服务F: 自然语言处理服务
LOG_F="${LOG_DIR}/F_qwen_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 Qwen LLM 服务 (F)... 日志: ${LOG_F}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_QWEN/bin/activate"
  cd "$PATH_QWEN"
  python3 -u "$SCRIPT_QWEN" 2>&1 | tee "$LOG_F"
  echo "[PID: $$] ✅ (F) Qwen服务已执行完毕。"
) &

# 启动服务G: 文本转语音服务
LOG_G="${LOG_DIR}/G_tts_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 文本转语音服务 (G)... 日志: ${LOG_G}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_TTS/bin/activate"
  cd "$PATH_TTS"
  python3 -u "$SCRIPT_TTS" 2>&1 | tee "$LOG_G"
  echo "[PID: $$] ✅ (G) TTS服务已执行完毕。"
) &

# 启动服务H: 障碍物检测服务
LOG_H="${LOG_DIR}/H_vision_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 障碍物检测服务 (H)... 日志: ${LOG_H}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_Vision1/bin/activate"
  cd "$PATH_Vision1"
  python3 -u "$SCRIPT_Vision1" 2>&1 | tee "$LOG_H"
  echo "[PID: $$] ✅ (H) 障碍物检测服务已执行完毕。"
) &


# 启动服务H: 障碍物检测服务
LOG_K="${LOG_DIR}/K_vision_service_${TIMESTAMP}.log"
(
  echo "[PID: $$] 🚀 正在启动 障碍物检测服务 (H)... 日志: ${LOG_H}" || send_error_alert "手柄服务(B)" # <--- 修改点
  source "$VENV_Rag/bin/activate"
  cd "$PATH_Rag"
  python3 -u "$SCRIPT_Rag" 2>&1 | tee "$LOG_H"
  echo "[PID: $$] ✅ (K) RAG服务已执行完毕。"
) &

# --- 等待与收尾 ---

echo "--------------------------------------------------------"
echo "✅ 所有九个服务已在后台启动。正在等待它们完成..."
echo "日志目录: ${LOG_DIR}"
echo "你可以通过 'ps aux | grep python' 查看进程。"
echo "使用 'tail -f ${LOG_DIR}/<log_file_name>.log' 单独监控某个服务的日志。"

# `wait` 命令会暂停脚本，直到所有在后台运行的子进程都结束。
# 如果任何一个服务是无限循环的（像web服务），脚本会一直停留在这里。
# 如果你想在按下Ctrl+C时能优雅地关闭所有子进程，可以添加一个trap。
trap "echo '...捕获到 Ctrl+C, 正在关闭所有后台服务...'; trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

wait

echo "🏁 所有后台服务均已结束。"
