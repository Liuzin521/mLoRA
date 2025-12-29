import os
import yaml
import random

# ================= 全混合负载配置 =================

# 1. 基础工作目录 (你的 baseline 目录的绝对路径)
# 这样无论从哪里启动，都能找到文件
WORK_DIR = "/scr/dataset/yuke/zien/mLoRA/zien_work/baseline"

# 2. Config 输出目录
BASE_OUTPUT_DIR = os.path.join(WORK_DIR, "experiment_configs_90tasks")

# 3. 数据集路径 (指向 baseline/data)
# 数据都在 zien_work/baseline/data/ 下
DATASETS = {
    "gsm8k":      os.path.join(WORK_DIR, "data/gsm8k_train_subset.json"),
    "winogrande": os.path.join(WORK_DIR, "data/winogrande_train_subset.json"),
    "mrpc":       os.path.join(WORK_DIR, "data/mrpc_train_subset.json"),
    "finqa":      os.path.join(WORK_DIR, "data/finqa_train_subset.json")
}

# 4. Adapter 保存路径 (指向 baseline/adapters)
ADAPTER_SAVE_DIR = os.path.join(WORK_DIR, "adapters")

# 5. 其他参数保持不变
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
TOTAL_TASKS_PER_FILE = 90

LORA_PARAM_POOL = [
    {"r": 128, "alpha": 32, "lr": 0.0001},
    {"r": 128, "alpha": 64, "lr": 0.0001},
    {"r": 64,  "alpha": 32, "lr": 0.0001},
    {"r": 64,  "alpha": 16, "lr": 2e-05},
    {"r": 32,  "alpha": 32, "lr": 5e-05},
]

BATCH_SIZE_POOL = [2, 4, 8, 16]

TARGET_MODULES = {
    "q_proj": True, "k_proj": True, "v_proj": True, "o_proj": True,
    "gate_proj": True, "up_proj": True, "down_proj": True
}

# ================= 🛠️ 生成逻辑 =================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_yaml():
    ensure_dir(BASE_OUTPUT_DIR)
    
    print(f"🚀 Generating Configs with FIXED PATHS...")
    print(f"   Data Dir: {os.path.join(WORK_DIR, 'data')}")
    print(f"   Adapter Dir: {ADAPTER_SAVE_DIR}")

    for dataset_name, data_path in DATASETS.items():
        # 检查数据文件是否存在，避免生成无效配置
        if not os.path.exists(data_path):
            print(f"⚠️ Warning: Data file not found: {data_path}")
        
        dataset_dir = os.path.join(BASE_OUTPUT_DIR, dataset_name)
        ensure_dir(dataset_dir)
        
        print(f"\n📂 Processing {dataset_name}...")
        
        for concurrency in CONCURRENCY_LEVELS:
            config = {
                "dispatcher": {
                    "name": "default",
                    "concurrency_num": concurrency
                },
                "datasets": [{
                    "name": f"{dataset_name}_data",
                    "data": data_path,  # <--- 这里现在是绝对路径了
                    "prompt": "demo/prompt.yaml",
                    "prompt_type": "instruction",
                    "preprocess": "shuffle"
                }],
                "adapters": [],
                "tasks": []
            }

            for i in range(TOTAL_TASKS_PER_FILE):
                adapter_name = f"lora_{dataset_name}_{i}"
                task_name = f"task_{dataset_name}_{i}"
                
                lora_params = random.choice(LORA_PARAM_POOL)
                bs = random.choice(BATCH_SIZE_POOL)
                
                # Adapter 路径也改到 baseline/adapters 下
                adapter_path = os.path.join(ADAPTER_SAVE_DIR, f"{dataset_name}_pool", adapter_name)

                adapter = {
                    "name": adapter_name,
                    "type": "lora",
                    "path": adapter_path, # <--- 绝对路径
                    "optimizer": "adamw",
                    "dropout": 0.05,
                    "target_modules": TARGET_MODULES.copy(),
                    "r": lora_params["r"],
                    "alpha": lora_params["alpha"],
                    "lr": lora_params["lr"]
                }
                config["adapters"].append(adapter)
                
                task = {
                    "type": "train",
                    "name": task_name,
                    "adapter": adapter_name,
                    "dataset": f"{dataset_name}_data",
                    "batch_size": bs,
                    "mini_batch_size": bs,
                    "num_epochs": 1,
                    "cutoff_len": 512,
                    "save_step": 10000
                }
                config["tasks"].append(task)

            filename = f"c{concurrency}.yaml"
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
                
        print(f"  ✅ Generated c1..c{max(CONCURRENCY_LEVELS)} for {dataset_name}")

    print(f"\n✨ All done! Configs updated in: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    generate_yaml()