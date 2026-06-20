import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# 1. Cấu hình Model bản nhẹ để test máy cá nhân
MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat" [cite: 41]
MAX_NEW_TOKENS = 128 [cite: 42]

def load_model():
    print(f"[*] Đang tải mô hình {MODEL_NAME}...") [cite: 44]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) [cite: 45]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, [cite: 48]
        torch_dtype=torch.float16, [cite: 49]
        device_map="auto" # Tự động nhận diện thiết bị [cite: 50]
    )
    return tokenizer, model [cite: 51]

def process_single_question(tokenizer, model, question_text):
    # Prompt Engineering
    system_prompt = "Bạn là một AI Agent chuyên nghiệp. Hãy trả lời câu hỏi sau thật chính xác." [cite: 54, 55]
    messages = [
        {"role": "system", "content": system_prompt}, [cite: 57]
        {"role": "user", "content": question_text} [cite: 58]
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) [cite: 60, 61]
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device) [cite: 62]
    
    generated_ids = model.generate(
        model_inputs.input_ids, [cite: 65]
        max_new_tokens=MAX_NEW_TOKENS, [cite: 66]
        pad_token_id=tokenizer.eos_token_id [cite: 67]
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) [cite: 69, 70]
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] [cite: 72]
    return response.strip()

def main():
    input_file = "data/public_test.csv" [cite: 74]
    output_file = "pred.csv" [cite: 75]
    
    if not os.path.exists(input_file):
        print(f"[!] Lỗi: Không tìm thấy file {input_file}.") [cite: 77]
        return

    df = pd.read_csv(input_file) [cite: 78]
    predictions = [] [cite: 79]
    tokenizer, model = load_model() [cite: 80, 81]
    
    print("[*] Đang chạy Inference...") [cite: 82]
    for index, row in tqdm(df.iterrows(), total=df.shape[0]): [cite: 83]
        q_id = row['id'] [cite: 84]
        question = row['question'] [cite: 87]
        answer = process_single_question(tokenizer, model, question) [cite: 88]
        predictions.append({'id': q_id, 'answer': answer}) [cite: 89]
        
    pred_df = pd.DataFrame(predictions) [cite: 90]
    pred_df.to_csv(output_file, index=False, encoding='utf-8') [cite: 91]
    print(f"[+] Đã xuất file: {output_file}") [cite: 91]

if __name__ == "__main__": [cite: 92, 93]
    main() [cite: 94]
