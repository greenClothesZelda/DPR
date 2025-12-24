#GPT로 작성함
import ijson
import json
import random
import os
from decimal import Decimal  # [추가됨] Decimal 타입 처리를 위해 import

# ================= 설정 부분 =================
train_input_path = 'downloads/data/retriever/nq-train.json'
dev_input_path   = 'downloads/data/retriever/nq-dev.json'

train_output_path = 'data/train.jsonl'  # 경로가 data/ 폴더인 것 같아 수정했습니다
valid_output_path = 'data/valid.jsonl'
test_output_path  = 'data/test.jsonl'

VALID_RATIO = 0.5 
SEED = 42
# ===========================================

# [추가됨] Decimal 타입을 float으로 바꿔주는 도우미 함수
def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def convert_train():
    # 저장할 폴더가 없으면 에러나므로 미리 생성
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    
    print(f"🔹 [1/2] Train 데이터 변환 시작: {train_input_path} -> {train_output_path}")
    count = 0
    with open(train_input_path, 'rb') as infile, open(train_output_path, 'w', encoding='utf-8') as outfile:
        for item in ijson.items(infile, 'item'):
            # [수정됨] default=decimal_to_float 추가
            line = json.dumps(item, default=decimal_to_float, ensure_ascii=False)
            outfile.write(line + '\n')
            count += 1
            if count % 10000 == 0:
                print(f"   - {count}개 처리 중...")
    print(f"✅ Train 변환 완료! (총 {count}개)")

def split_dev():
    os.makedirs(os.path.dirname(valid_output_path), exist_ok=True)
    
    print(f"🔹 [2/2] Dev 데이터 분할 및 변환 시작: {dev_input_path}")
    print(f"   - 비율: Valid({VALID_RATIO*100}%) / Test({(1-VALID_RATIO)*100}%)")
    
    random.seed(SEED)
    valid_count = 0
    test_count = 0
    
    with open(dev_input_path, 'rb') as infile, \
         open(valid_output_path, 'w', encoding='utf-8') as f_valid, \
         open(test_output_path, 'w', encoding='utf-8') as f_test:
        
        for item in ijson.items(infile, 'item'):
            # [수정됨] default=decimal_to_float 추가
            line = json.dumps(item, default=decimal_to_float, ensure_ascii=False) + '\n'
            
            if random.random() < VALID_RATIO:
                f_valid.write(line)
                valid_count += 1
            else:
                f_test.write(line)
                test_count += 1
                
    print(f"✅ Dev 분할 완료!")
    print(f"   - Valid: {valid_count}개 저장됨 -> {valid_output_path}")
    print(f"   - Test : {test_count}개 저장됨 -> {test_output_path}")

if __name__ == "__main__":
    if os.path.exists(train_input_path):
        convert_train()
    else:
        print(f"❌ 오류: {train_input_path} 파일이 없습니다.")

    print("-" * 30)

    if os.path.exists(dev_input_path):
        split_dev()
    else:
        print(f"❌ 오류: {dev_input_path} 파일이 없습니다.")