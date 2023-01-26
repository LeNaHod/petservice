## 저장한 모델의 binary 파일을 이용해서 요약문을 생성한 후 csv로 저장하는 코드
# cmd 실행 명령어 : CUDA_VISIBLE_DEVICES=0 python make_summary_csv.py --model_dir 3e9_batch8_model --output_dir summary_all/ --start_num 0

import argparse, torch, os
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
import pandas as pd
from datetime import datetime, timedelta
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# py 파일 실행시 입력받을 argument 를 설정한다.
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default=None, type=str)  # 모델이 저장된 디렉토리 입력
parser.add_argument("--output_dir", default='summary/', type=str) # 요약문csv를 저장할 디렉토리 입력
parser.add_argument("--start_num", default=0, type=int) # 이어서 작업할 행 번호 입력

args = parser.parse_args()


# 저장된 모델을 불러온다.
def load_model(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return model


def create_new_csv(n, filename): # 결과를 저장할 csv가 없을 경우 헤더만 새로 생성하는 함수
    df = pd.DataFrame(None, columns=['summary'])
    df.to_csv(filename, mode='w', index=False, sep='\t')
    del df
    print('##### ', filename, ' 파일 생성 완료')
    
    
def save(summary_list, n):
    df = pd.DataFrame(summary_list, columns=['summary'])
    filename = args.output_dir + args.model_dir.split('model')[0]+'nbeams' +str(n) + '.csv'
    
    if not os.path.exists(filename):  # csv 파일이 없을 경우 먼저 새로 생성하기
        create_new_csv(n, filename)
        
    
    # 기존 결과 파일에 한 행 추가하기
    df.to_csv(filename, mode='a', header=False, index=False, sep='\t')
    del df
    print('##### ', filename, ' 파일 저장 완료')
    
    
def process(x):
    with torch.cuda.device(0): # gpu 장치 사용하기
        print("##### cuda 디바이스 :{}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
        
        # 토크나이저로 인코딩
        x_encodings = [tokenizer.encode(text) for text in x] # 토크나이저로 인코딩 (텍스트 -> 숫자)
        print('##### 토크나이저 인코딩 완료')


        # 텐서로 변환
        x_tensors = [torch.tensor([x_encoding], device=device) for x_encoding in x_encodings] # 데이터 타입을 tensor로 변환
        print('##### 텐서 변환 완료')


        # 요약문 생성
        for n in [1, 5, 10, 15, 20]: # num_beams 변화에 따른 결과 비교를 위해 n 수를 반복한다.
            for_start = datetime.now() +timedelta(hours=9)  # 시작 시간
            print(f'##### nbeams {n} for문 시작 시간 : ', for_start.strftime('%Y-%m-%d %H:%M:%S'))

            output_list = [ model.generate(x_tensor, eos_token_id=1, max_length=256, num_beams=n, min_length=10) for x_tensor in x_tensors ] 
            print('##### ', n ,' beams 요약문 생성 완료')

            summary_list = [ tokenizer.decode(output[0], skip_special_tokens=True) for output in output_list ]
            print('##### ', n ,' beams 토크나이저 디코딩 완료')

            save(summary_list, n)
            del output_list, summary_list
            for_end = datetime.now() +timedelta(hours=9) # 끝 시간
            print(f'##### nbeams {n} for문 종료 시간 : ', for_end.strftime('%Y-%m-%d %H:%M:%S'))
            print(f'##### nbeams {n} for문 소요 시간 : ', for_end - for_start)
            print('-'*40)

        del x_encodings, x_tensors
    

    
# GPU 사용 설정
print('# gpu 사용 가능 여부 : ', torch.cuda.is_available())
print("# cuda 디바이스 갯수 :{}".format(torch.cuda.device_count()))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('# 사용할 장치 : ', device)
    
    
start = datetime.now() +timedelta(hours=9)  # 시작 시간
print('# 시작 시간 : ', start.strftime('%Y-%m-%d %H:%M:%S'))

    
    
    
# 모델과 토크나이저 로드
model = load_model(args.model_dir)
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model.to(device) # 불러온 모델을 gpu에 로드
print('# 모델과 토크나이저 로드 완료')


# 데이터 원문 로드
valid = pd.read_csv('../dataset/valid.csv', sep='\t', index_col=0, low_memory=False)
print('# 원본 요약문 로드 완료')

# 요약 시작 행
idx = args.start_num
print('# 시작 idx : ', idx)

# 반복하면서 df 행 개수만큼 반복
# i를 백씩 올려서 작업
# 만약 i+100이 df 개수를 넘는 경우 인덱싱 끝 숫자는 df의 마지막 수 -> 반복문 종료

while True:
    while_start = datetime.now() +timedelta(hours=9)  # 시작 시간
    print(f'### 시작인덱스 {idx} while 시작 시간 : ', while_start.strftime('%Y-%m-%d %H:%M:%S'))
    
    if idx+100 > len(valid): # idx+100 이 데이터 전체 개수보다 많다면, 남은 데이터를 모두 요약하고 반복문을 빠져나간다
        # 데이터 분리
        x = valid['text'].values[idx:len(valid)]
        print(f'### [{idx} : {len(valid)}] 데이터 분리 완료, shape: ' ,x.shape)
        process(x)
        break
        
    else:
        # 데이터 분리
        x = valid['text'].values[idx:idx+100]
        print(f'### [{idx} : {idx+100}] 데이터 분리 완료, shape: ' ,x.shape)
        process(x)
    
    while_end = datetime.now() +timedelta(hours=9) # 끝 시간
    print(f'### {idx} 작업 while 종료 시간 : ', while_end.strftime('%Y-%m-%d %H:%M:%S'))
    print(f'### {idx} 작업 while 소요 시간 : ', while_end - while_start)
    
    idx+=100 # 마지막 100 행이 아니라면 idx에 100을 더해서 계속 반복한다.
    
    
print('# ', args.model_dir,' 모델의 make_summary_csv.py 전체 작업 종료')


end = datetime.now() +timedelta(hours=9) #  끝 시간
print('# 종료 시간 : ', end.strftime('%Y-%m-%d %H:%M:%S'))
print('# 소요 시간 : ', end - start)

