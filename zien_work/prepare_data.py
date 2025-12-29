import json
import os
from datasets import load_dataset

# 路径设置
output_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(output_dir, "gsm8k_train.json")

print(f"正在下载 GSM8K 数据集...")
dataset = load_dataset("gsm8k", "main", split="train")

mlora_data = []
print(f"正在转换数据...")

for item in dataset:
    mlora_data.append({
        "instruction": item['question'],
        "input": "",
        # 【关键修改】这里必须叫 "chosen"，才能匹配 demo/prompt.yaml 模板
        "chosen": item['answer']  
    })

# 重新保存覆盖旧文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(mlora_data, f, indent=4, ensure_ascii=False)

print(f"✅ 数据已修复并覆盖保存到: {output_file}")