import torch
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 配置区域 =================
BASE_MODEL_PATH = "/scr/dataset/yuke/models/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./zien_work/adapters/llama3_gsm8k_res"
#为了快速测试，先只跑前 50 题看看效果？(设为 None 则跑全量 1319 题)
TEST_SAMPLE_SIZE = None 
# ===========================================

def extract_answer_number(text):
    """
    从 GSM8K 的标准答案格式中提取最终数字。
    标准答案通常以 '#### 数字' 结尾。
    """
    # 找 #### 后面的内容
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        # 去掉逗号，比如 1,234 -> 1234
        return match.group(1).replace(',', '')
    return None

def extract_model_number(text):
    """
    从模型输出中尝试提取最后一个数字
    """
    # 简单的策略：找文本中最后一个出现的数字
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def main():
    print(f"正在加载测试集 (Test Split)...")
    # 【这里就是你要找的测试集】
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if TEST_SAMPLE_SIZE:
        print(f"⚠️ 为了省时间，当前只测试前 {TEST_SAMPLE_SIZE} 道题。")
        dataset = dataset.select(range(TEST_SAMPLE_SIZE))

    print("正在加载模型 (这可能需要几分钟)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # 必须设为 padding_side='left' 才能进行批处理生成(虽然这里我们用单条循环)
    tokenizer.padding_side = 'left' 
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    # 挂载你的 LoRA
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    correct_count = 0
    total_count = 0

    print("开始考试 (Running Inference)...")
    # 使用 tqdm 显示进度条
    for item in tqdm(dataset):
        question = item['question']
        ground_truth_str = item['answer']
        
        # 1. 提取标准答案里的数字
        ground_truth_num = extract_answer_number(ground_truth_str)
        if not ground_truth_num:
            continue # 如果标准答案都没解析出来，跳过

        # 2. 构造 Prompt (跟训练时保持一致)
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 3. 模型生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, # 给够解题步骤的长度
                temperature=0.1,    # 测试时温度低一点，让答案更稳定
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 4. 解析模型输出
        # 只看新生成的部分
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_num = extract_model_number(generated_text)

        # 5. 判分
        if pred_num and float(pred_num) == float(ground_truth_num):
            correct_count += 1
        
        total_count += 1

    # ================= 输出成绩单 =================
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print("\n" + "="*30)
    print(f"Baseline 评测结果")
    print(f"正确: {correct_count}")
    print(f"错误: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()