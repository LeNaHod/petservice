# hparams과 ckpt의 형태로 저장된 모델을 이용해서 요약문을 생성한 후 csv로 저장하는 코드
# KoBART-summarization 내에 넣어 사용하였음
# 실행 예 !python hparams_based_make_summary_csv.py --hparams ./logs/tb_logs/default/version_4/hparams.yaml \
#                                                  --model_binary ./logs/model_chp/epoch=02-val_loss=1.491.ckpt \
#                                                  --output_dir summary_all/

import argparse, torch, os
import pandas as pd
import yaml
from datetime import datetime, timedelta
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from train import KoBARTConditionalGeneration


# py 파일 실행시 입력받을 argument를 설정한다.
parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str) # 모델의 버전
parser.add_argument("--model_binary", default=None, type=str) # 
parser.add_argument("--output_dir", default='summary/', type=str) # 요약문 csv를 저장할 디렉토리 입력
parser.add_argument("--start_num", default=0, type=int) # 이어서 작업할 행 번호 입력

args = parser.parse_args()


# 저장된 모델을 불러온다.

with open(args.hparams) as f: # get_model_binary에서 가져옴. ckpt의 hparam을 가져온다
    hparams = yaml.load(f)
    
inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)

def create_new_csv(n, filename): # 결과를 저장할 csv가 없을 경우 헤더만 새로 생성하는 함수
    df = pd.DataFrame(None, columns=['summary'])
    df.to_csv(filename, mode='w', index=False, sep='\t')
    del df
    print('### ', filename, ' 파일 생성 완료')
    
    
def save(summary_list, n):
    df = pd.DataFrame(summary_list, columns=['summary'])
    #filename = args.output_dir + args.model_dir.split('model')[0]+'nbeams' +str(n) + '.csv'
    filename = args.output_dir + 'batch8_3e-5_epoch02_'+'nbeams' +str(n) + '.csv'
    
    if not os.path.exists(filename):  # csv 파일이 없을 경우 먼저 새로 생성하기
        create_new_csv(n, filename)
        
    
    # 기존 결과 파일에 한 행 추가하기
    df.to_csv(filename, mode='a', header=False, index=False, sep='\t')
    print('### ', filename, ' 파일 저장 완료')
    
    
def process(x):
    # 토크나이저로 인코딩
    x_encodings = [tokenizer.encode(text) for text in x] # 토크나이저로 인코딩 (텍스트 -> 숫자)
    print('### 토크나이저 인코딩 완료')


    # 텐서로 변환
    x_tensors = [torch.tensor([x_encoding]) for x_encoding in x_encodings] # 데이터 타입을 tensor로 변환
    print('### 텐서 변환 완료')


    # 요약문 생성
    #for n in [5, 20]: # num_beams 변화에 따른 결과 비교를 위해 n 수를 반복한다.
    for n in [20]: # 편의를 위해 5와 20 작업을 따로 분리하여 진행하였음
        output_list = [ model.generate(x_tensor, eos_token_id=1, max_length=256, num_beams=n, min_length=10) for x_tensor in x_tensors ] 
        print('### ', n ,' beams 요약문 생성 완료')

        summary_list = [ tokenizer.decode(output[0], skip_special_tokens=True) for output in output_list ]
        print('### ', n ,' beams 토크나이저 디코딩 완료')

        save(summary_list, n)
        del output_list, summary_list
    
    
    
    
start = datetime.now() +timedelta(hours=9)  # 시작 시간
print('#### 시작 시간 : ', start.strftime('%Y-%m-%d %H:%M:%S'))

    
    
    
# 모델과 토크나이저 로드
model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device) # device가 정의되지 않는 에러 발생하여 제외
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
print('### 모델과 토크나이저 로드 완료')


# 데이터 원문 로드
valid = pd.read_csv('/gdrive/MyDrive/Colab_Notebooks/KoBART-summarization/data/test.tsv', sep='\t')
print('### 데이터 로드 완료')

# 요약 시작 행
idx = args.start_num
print('## 시작 idx는 ', idx)

# 반복하면서 df 행 개수만큼 반복
# i를 백씩 올려서 작업
# 만약 i+100이 df 개수를 넘는 경우 인덱싱 끝 숫자는 df의 마지막 수 -> 반복문 종료

while True:
    while_start = datetime.now() +timedelta(hours=9)  # 시작 시간
    print(f'#### {idx} 작업 while 시작 시간 : ', while_start.strftime('%Y-%m-%d %H:%M:%S'))
    
    if idx+100 > len(valid): # idx+100 이 데이터 전체 개수보다 많다면, 남은 데이터를 모두 요약하고 반복문을 빠져나간다
        # 데이터 분리
        x = valid['news'].values[idx:len(valid)]
        print(f'### [{idx} : {len(valid)}] 데이터 분리 완료, shape: ' ,x.shape)
        process(x)
        break
        
    else:
        # 데이터 분리
        x = valid['news'].values[idx:idx+100]
        print(f'### [{idx} : {idx+100}] 데이터 분리 완료, shape: ' ,x.shape)
        process(x)
        idx+=100 # 마지막 100 행이 아니라면 idx에 100을 더해서 계속 반복한다.
    
    while_end = datetime.now() +timedelta(hours=9) # 끝 시간
    print(f'#### {idx} 작업 while 종료 시간 : ', while_end.strftime('%Y-%m-%d %H:%M:%S'))
    print(f'#### {idx} 작업 while 소요 시간 : ', while_end - while_start)
    
print('###, make_summary_csv.py 작업 종료')


end = datetime.now() +timedelta(hours=9) #  끝 시간
print('#### 종료 시간 : ', end.strftime('%Y-%m-%d %H:%M:%S'))
print('#### 소요 시간 : ', end - start)