#!/bin/bash

# ================= 0. 环境准备 =================
# 每次启动前自动清理旧进程，防止端口占用
echo "Cleaning up old processes..."
pkill -9 -f mlora_pp_train.py
sleep 2 # 等操作系统回收端口

# 设置多卡之间通信
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23456  # 确保端口没被占用


# ================= 跳回项目根目录 =================
# 因为这个脚本在 zien_work/baseline/ 下，退回两层去找到 mlora_pp_train.py
cd ../..
echo " Working Directory switched to: $(pwd)"

# ================= 配置区域 =================

# 1. 模型路径
MODEL="/scr/dataset/yuke/models/Llama-3.1-8B-Instruct"

# 2. 配置文件路径
# 注意：因为我们已经跳回了根目录，所以这里要写相对于根目录的路径
# 确保这个路径下真的有生成好的 yaml 文件
CONFIG="zien_work/baseline/experiment_configs_90tasks/gsm8k/c8.yaml"

# 3. 日志保存目录
LOG_DIR="zien_work/baseline/logs_pp_gsm8kfull_c8"
mkdir -p $LOG_DIR

# 4. 显卡物理编号 (根据 nvidia-smi 看到的编号 1,2,3,4)
GPU_IDS=(0 1 2 3)

# 5. 层分配策略 (32层 / 4卡 ) 不知道为啥35层 Embedding 层（输入层） 和 RMSNorm 层（输出层） 也算作了需要分配的“层”。 
BALANCE="8 9 9 9"

# ================= 启动逻辑 =================

echo "   Starting mLoRA Pipeline Parallelism (PP)..."
echo "   Config: $CONFIG"
echo "   Balance: $BALANCE"
echo "   Master:  $MASTER_ADDR:$MASTER_PORT"

# 循环启动 4 个 Rank
for RANK in 0 1 2 3
do  

    DEVICE_ID=${GPU_IDS[$RANK]}
    
    echo "   -> Launching Rank $RANK on Device cuda:$DEVICE_ID ..."

    nohup python mlora_pp_train.py \
        --base_model $MODEL \
        --config $CONFIG \
        --precision bf16 \
        --pipeline \
        --rank $RANK \
        --balance $BALANCE \
        --recompute \
        --device "cuda:$DEVICE_ID" \
        --log_file $LOG_DIR/rank_${RANK}_internal.log \
        > $LOG_DIR/rank_${RANK}_console.log 2>&1 &
        
    sleep 2
done

echo "   All Ranks Launched!"
echo "   主进程日志: tail -f $LOG_DIR/rank_0_console.log"