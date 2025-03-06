#!/bin/bash

# 获取所有与/dev/nvidia*相关的进程ID
PIDS=$(fuser -v /dev/nvidia* 2>/dev/null | awk '{for(i=3;i<=NF;i++) print $i}')

# 检查是否找到任何PID
if [ -z "$PIDS" ]; then
  echo "没有找到占用NVIDIA设备的进程。"
else
  # 终止每个找到的进程
  for PID in $PIDS; do
    echo "终止进程: $PID"
    kill -9 $PID
  done
  echo "所有占用NVIDIA设备的进程已被终止。"
fi
