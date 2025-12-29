import json
import os
import re
from datasets import load_dataset

# ================= 配置 =================
OUTPUT_DIR = "./data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 通用工具 =================
def flatten_messages(system_content, user_content, assistant_content):
    """
    将师兄的 Chat 格式 (System + User) 展平为 mLoRA 的 Instruction 格式。
    mLoRA 的 prompt.yaml 会自动加 "### Instruction:"，所以我们只需要拼接内容。
    """
    # 拼接 System 和 User，中间加换行
    full_instruction = f"{system_content}\n\n{user_content}"
    return {
        "instruction": full_instruction,
        "input": "", # 留空，因为内容都在 instruction 里了
        "chosen": assistant_content
    }

# ================= 1. GSM8K (复刻师兄逻辑) =================
def process_gsm8k():
    print("Processing GSM8K...")
    # 师兄定义的 System Prompt
    SYSTEM_PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context."
        "Write a response that appropriately completes the request.\n\n"
        "Solve the following grade-school math problem. Show your reasoning step-by-step"
        "and give the final numeric answer in the format '#### <answer>'.\n\n"
    )
    
    ds = load_dataset("openai/gsm8k", "main", split="train")
    processed = []
    
    for example in ds:
        user_content = example["question"].strip()
        assistant_content = example["answer"].strip()
        processed.append(flatten_messages(SYSTEM_PROMPT, user_content, assistant_content))
        
    return processed

# ================= 2. Winogrande (复刻 winogrande_utils.py) =================
def process_winogrande():
    print("Processing Winogrande...")
    SYSTEM_PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "You will be given a sentence with a blank (indicated by _) and two options. "
        "Choose the option that best fits the blank to create a coherent sentence. "
        "Respond with ONLY the number 1 or 2, corresponding to your choice."
    )
    
    # 注意：师兄用的是 "winogrande_debiased" 子集
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="train")
    processed = []
    
    for example in ds:
        sentence = example.get("sentence", "").strip()
        option1 = example.get("option1", "").strip()
        option2 = example.get("option2", "").strip()
        answer = example.get("answer", "").strip()
        
        user_content = (
            f"Sentence: {sentence}\n\n"
            f"Option 1: {option1}\n"
            f"Option 2: {option2}\n\n"
            f"Which option best fits the blank? Answer with 1 or 2."
        )
        
        assistant_content = f"The answer is: {answer}"
        processed.append(flatten_messages(SYSTEM_PROMPT, user_content, assistant_content))
        
    return processed

# ================= 3. MRPC (复刻 mrpc_utils.py) =================
def process_mrpc():
    print("Processing MRPC...")
    SYSTEM_PROMPT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "You will be given two sentences. Your task is to determine whether they are paraphrases of each other. "
        "Respond with ONLY 'equivalent' if they are paraphrases or 'not_equivalent' if they are not."
    )
    
    # 师兄代码指定了 "SetFit/mrpc"，这是 HF 上 MRPC 的一个镜像源，或者通常 glue/mrpc
    # 这里我们使用 glue mrpc 保持稳定，如果报错可尝试 SetFit/mrpc
    try:
        ds = load_dataset("glue", "mrpc", split="train")
    except:
        ds = load_dataset("SetFit/mrpc", split="train")
        
    processed = []
    
    for example in ds:
        # GLUE dataset keys: sentence1, sentence2, label
        # SetFit dataset keys: text1, text2, label
        text1 = example.get("sentence1", example.get("text1", "")).strip()
        text2 = example.get("sentence2", example.get("text2", "")).strip()
        label = example.get("label", 0)
        
        label_text = "equivalent" if label == 1 else "not_equivalent"
        
        user_content = (
            f"Sentence 1: {text1}\n\n"
            f"Sentence 2: {text2}\n\n"
            f"Are these sentences paraphrases of each other?"
        )
        
        processed.append(flatten_messages(SYSTEM_PROMPT, user_content, label_text))
        
    return processed

# ================= 4. FinQA (复刻 finqa_utils.py + 自动 Few-shot) =================
def format_finqa_table(table):
    """复刻 _format_table 逻辑"""
    if not table: return ""
    rows = [" | ".join(str(c) for c in row) for row in table]
    header = rows[0]
    sep = " | ".join(["---"] * len(table[0]))
    body = rows[1:]
    return "\n".join([header, sep, *body])

def convert_gold_inds_to_reasoning(gold_inds):
    """复刻 _convert_gold_inds_to_reasoning 逻辑"""
    if not gold_inds: return ""
    reasoning_steps = []
    step_num = 1
    for evidence in gold_inds:
        evidence = evidence.strip()
        if evidence.endswith(';'): evidence = evidence[:-1]
        facts = [fact.strip() for fact in evidence.split(';') if fact.strip()]
        for fact in facts:
            if " is " in fact:
                parts = fact.split(" is ")
                if len(parts) == 2:
                    subject_part = parts[0].strip()
                    value_part = parts[1].strip()
                    if " of " in subject_part:
                        key_part, entity_part = subject_part.split(" of ", 1)
                        key_part = key_part.replace("the ", "").replace("company ", "").replace("date ", "")
                        reasoning_steps.append(f"Step {step_num}: From the data, {key_part} for {entity_part} is {value_part}.")
                    else:
                        reasoning_steps.append(f"Step {step_num}: {subject_part} is {value_part}.")
                else:
                    reasoning_steps.append(f"Step {step_num}: {fact}.")
            else:
                reasoning_steps.append(f"Step {step_num}: {fact}.")
            step_num += 1
    return "\n".join(reasoning_steps)

def generate_finqa_fewshot_block(dataset, num_shots=2):
    """自动生成 Few-shot 块，替代读取 sample_first5.jsonl"""
    examples = []
    for i in range(min(num_shots, len(dataset))):
        ex = dataset[i]
        q = ex.get("question", "").strip()
        pre_text = [l for l in ex.get("pre_text", []) if l.strip() != "."] # 过滤逻辑同师兄
        post_text = [l for l in ex.get("post_text", []) if l.strip() != "."]
        
        ctx_parts = []
        if pre_text: ctx_parts.append("[PRE]\n" + "\n".join(pre_text[:10]))
        if ex.get("table"): ctx_parts.append("[TABLE]\n" + format_finqa_table(ex["table"]))
        if post_text: ctx_parts.append("[POST]\n" + "\n".join(post_text[:10]))
        ctx = "\n\n".join(ctx_parts)
        
        gold_inds = ex.get("gold_inds", [])
        reasoning = convert_gold_inds_to_reasoning(gold_inds)
        
        # Answer normalization logic simplified but sufficient for cache
        ans = ex.get("final_result", ex.get("answer", ""))
        
        example_text = f"Example {i+1}\nContext:\n{ctx}\n\nQuestion: {q}\n\nReasoning:\n{reasoning}\n\nTherefore, the answer is:\n#### {ans}"
        examples.append(example_text)
        
    return (
        "=== FEW-SHOT START ===\n"
        f"Few-Shot Demonstrations ({num_shots})\n"
        + "\n\n".join(examples)
        + "\n=== FEW-SHOT END ===\n\n"
    )

def process_finqa():
    print("Processing FinQA...")
    SYSTEM_PROMPT = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "You are given a financial question along with surrounding textual evidence and a table excerpt. "
        "You should provide a succinct step-by-step reasoning showing how you extract relevant information from the context and perform calculations to arrive at the answer."
        "The output length should be less than 256 tokens."
        "Then provide the final answer in the exact format '#### <answer>'. "
    )
    
    ds = load_dataset("dreamerdeo/finqa", split="train")
    
    # 生成 Few-shot block (师兄代码默认 FEW_SHOT_NUM=2)
    fewshot_block = generate_finqa_fewshot_block(ds, num_shots=2)
    
    processed = []
    for example in ds:
        # Build Context
        pre_text = [l for l in example.get("pre_text", []) if l.strip() != "."]
        post_text = [l for l in example.get("post_text", []) if l.strip() != "."]
        table_block = format_finqa_table(example.get("table", []))
        
        context_parts = []
        if pre_text: context_parts.extend(["[PRE]", "\n".join(pre_text)])
        if table_block: context_parts.extend(["[TABLE]", table_block])
        if post_text: context_parts.extend(["[POST]", "\n".join(post_text)])
        context_text = "\n\n".join(context_parts)
        
        # User Content = Fewshot + Context + Question + Instruction
        user_content = (
            f"{fewshot_block}Context:\n{context_text}\n\n"
            f"Question: {example.get('question','').strip()}\n"
            "Please provide step-by-step reasoning and then give your final answer in the format '#### <answer>'."
        ).strip()
        
        # Assistant Content
        gold_inds = example.get("gold_inds", [])
        reasoning = convert_gold_inds_to_reasoning(gold_inds)
        raw_ans = example.get("final_result", example.get("answer", ""))
        
        reasoning_section = f"\n\nReasoning:\n{reasoning}\n" if reasoning else ""
        assistant_content = f"{reasoning_section}Therefore, the answer is:\n#### {raw_ans}"
        
        processed.append(flatten_messages(SYSTEM_PROMPT, user_content, assistant_content))
        
    return processed

# ================= 保存逻辑 =================
def save_json(data, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} items to {path}")

if __name__ == "__main__":
    # 1. GSM8K
    save_json(process_gsm8k(), "gsm8k_train.json")
    
    # 2. Winogrande
    save_json(process_winogrande(), "winogrande_train.json")
    
    # 3. MRPC
    save_json(process_mrpc(), "mrpc_train.json")
    
    # 4. FinQA
    save_json(process_finqa(), "finqa_train.json")
    
    print("\nAll datasets processed successfully matching colleague's logic!")