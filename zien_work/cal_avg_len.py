import json
from transformers import AutoTokenizer
import numpy as np

# 1. 设定模型路径 (用你服务器上的 Llama-3.1)
model_path = "/scr/dataset/yuke/models/Llama-3.1-8B-Instruct"

# 2. 设定数据路径
data_path = "zien_work/gsm8k_train.json"

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"Loading data from {data_path}...")
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. 计算长度
token_counts = []
print("Calculating token lengths...")

for item in data:
    # mLoRA 训练时通常把 instruction/input/output 拼在一起训练
    # 我们把所有文本字段拼起来算长度，这是最准确的估计
    full_text = ""
    for key, value in item.items():
        full_text += str(value) + " "
    
    # 计算 token 数量
    tokens = tokenizer.encode(full_text)
    token_counts.append(len(tokens))

# 4. 输出统计结果
avg_len = np.mean(token_counts)
max_len = np.max(token_counts)

print("="*30)
print(f"Total Samples: {len(data)}")
print(f"Average Length: {avg_len:.2f} tokens")  # <--- 这个填进表格
print(f"Max Length:     {max_len} tokens")
print("="*30)