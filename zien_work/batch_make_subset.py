import json
import os

# 1. 定义数据目录和文件名列表
data_dir = "zien_work/baseline/data"  # 数据都在这个目录下
file_list = [
    "finqa_train.json",
    "gsm8k_train.json",
    "mrpc_train.json",
    "winogrande_train.json"
]

# 2. 循环处理每一个文件
for file_name in file_list:
    input_path = os.path.join(data_dir, file_name)
    
    # 构造输出文件名，例如: finqa_train_subset_128.json
    file_base = file_name.replace(".json", "")
    output_name = f"{file_base}_subset_128.json"
    output_path = os.path.join(data_dir, output_name)
    
    print(f"Processing {file_name}...")
    
    try:
        # 读取
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 切片 (前128条)
        subset_data = data[:128]
        
        # 写入
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, indent=4)
            
        print(f"  -> Sliced {len(data)} to {len(subset_data)} samples.")
        print(f"  -> Saved to {output_path}")
        print("-" * 30)
        
    except FileNotFoundError:
        print(f"  [Error] File not found: {input_path}")
    except Exception as e:
        print(f"  [Error] Failed to process {file_name}: {e}")

print("All done!")
#Processing finqa_train.json...
#   -> Sliced 6251 to 128 samples.
#   -> Saved to zien_work/baseline/data/finqa_train_subset_128.json
# ------------------------------
# Processing gsm8k_train.json...
#   -> Sliced 7473 to 128 samples.
#   -> Saved to zien_work/baseline/data/gsm8k_train_subset_128.json
# ------------------------------
# Processing mrpc_train.json...
#   -> Sliced 3668 to 128 samples.
#   -> Saved to zien_work/baseline/data/mrpc_train_subset_128.json
# ------------------------------
# Processing winogrande_train.json...
#   -> Sliced 9248 to 128 samples.
#   -> Saved to zien_work/baseline/data/winogrande_train_subset_128.json
# ------------------------------
# All done!
