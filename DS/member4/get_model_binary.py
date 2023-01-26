# 저장된 check point 파일로부터 모델을 binary 파일로 변환하는 코드
# cmd 실행 명령어  : python get_model_binary.py --log_dir [logs dir]

import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml, glob


# py 파일 실행시 입력받을 argument 를 설정한다.
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default=None, type=str) 
parser.add_argument("--hparams", default='/tb_logs/default/version_0/hparams.yaml', type=str) # 저장된 hparams.yaml 파일 경로
parser.add_argument("--model_binary", default='model_chp', type=str)
args = parser.parse_args()


# hparams 파일을 로드한다.
with open(args.log_dir + args.hparams) as f:
    hparams = yaml.full_load(f)
    
    
# check point 폴더의 파일목록을 읽어서 loss가 가장 적은 check point 파일의 이름을 저장한다.
paths = glob.glob(args.log_dir + '/' + args.model_binary+'/*.ckpt')
min_loss = 100.0
min_path = ''
for path in paths:
    loss = float(path.split('val_loss=')[1].split('.ckpt')[0])
    if min_loss > loss:
        min_loss = loss
        min_path = path
        

# kobart 모델에 check point 를 로드한다.
inf = KoBARTConditionalGeneration.load_from_checkpoint(min_path, hparams=hparams)


# 학습 파라미터를 로드한 kobart 모델을 저장한다.
inf.model.save_pretrained(args.log_dir.split('logs')[0] + 'model')
print('모델 binary 저장 완료')
