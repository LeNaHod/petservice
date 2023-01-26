## 한 폴더안에 있는 모든 요약문csv들의 Rouge Score 를 계산하여 csv로 저장하는 코드
# cmd 명령어 : python rouge_score.py --summary_dir [폴더명] --output [결과csv파일명] --max_row 2500


import torch, argparse, os, glob
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from rouge import Rouge
import pandas as pd


# py 파일 실행시 입력받을 argument 를 설정한다.
parser = argparse.ArgumentParser()
parser.add_argument("--summary_dir", default=None, type=str)  # 요약문들이 들어있는 폴더명을 입력한다.
parser.add_argument("--output", default='rouge_score.csv', type=str) # 실행결과 만들어질 rouge 점수 csv파일의 파일명을 지정한다.
parser.add_argument("--max_row", default=-1, type=int) # 몇 번째 요약문까지 계산할건지, 점수계산에 사용할 요약문 수 입력
args = parser.parse_args()


def format_rouge_scores(scores): # F1, Recall, Precision 보기 쉽게 출력 포맷 변환하는 함수
    return """\n
    ****** ROUGE SCORES ******
    ** ROUGE 1
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE 2
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}
    ** ROUGE L
    F1        >> {:.3f}
    Precision >> {:.3f}
    Recall    >> {:.3f}""".format(
            scores["rouge-1"]["f"],
            scores["rouge-1"]["p"],
            scores["rouge-1"]["r"],
            scores["rouge-2"]["f"],
            scores["rouge-2"]["p"],
            scores["rouge-2"]["r"],
            scores["rouge-l"]["f"],
            scores["rouge-l"]["p"],
            scores["rouge-l"]["r"],
        )


# 결과파일을 만들기위해 columns을 먼저 정의한다.
columns = ['model_name', 'ROUGE 1 F1', 'ROUGE 1 Precision','ROUGE 1 Recall','ROUGE 2 F1', 'ROUGE 2 Precision', 'ROUGE 2 Recall', 'ROUGE L F1', 'ROUGE L Precision', 'ROUGE L Recall']


# 만들어져있는 결과파일이 없을경우 csv 헤더만 미리 생성하는 함수
def create_new_csv(): 
    df = pd.DataFrame(None, columns=columns) # 컬럼만 있고 데이터는 없는 DataFrame을 만든다.
    df.to_csv(args.output, mode='w', index=False) # csv 파일로 저장한다.
    del df
    print(args.output,' 파일 생성 완료')
    
    
# rouge 점수를 csv 에 저장하는 함수
def save_result(scores, filename=None): 
    result = [] # 결과가 담길 빈 리스트를 생성한다.
    
    
    # 첫번째 컬럼인 model_name을 result 리스트에 추가한다.
    result.append(filename.split('/')[1].split('.csv')[0]) 
    
    
    # rouge 점수를 result 리스트에 추가한다.
    rouge_list = ['rouge-1', 'rouge-2', 'rouge-l'] # 컬럼을 지정한다.
    score_list = ['f', 'p', 'r'] # 컬럼을 지정한다.
    
    for r in rouge_list: # rouge 점수 종류만큼 반복한다.
        for s in score_list: # 평가지표 종류만큼 반복한다.
            result.append(scores[r][s]) # rouge 점수를 result 리스트에 추가한다.

    # 결과 리스트를 DataFrame으로 만든다. (df 는 1개의 row로 만들어진다.)
    df = pd.DataFrame([result], columns=columns) 

    
    # df를 csv로 저장하기    
    if not os.path.exists(args.output):  # csv 파일이 없을 경우 헤더만 있는 빈 파일을 먼저 생성한다.
        create_new_csv()
        
    # csv 파일에 한 줄 추가하기
    df.to_csv(args.output, mode='a', header=False, index=False) 
    print('csv에 결과 저장 완료')

    del result, rouge_list, score_list, df # 메모리에서 안쓰는 요소 삭제하기

    
# 원본 valid.csv 파일 (정답 데이터) 을 읽는 함수
def read_valid():
    valid = pd.read_csv('../dataset/valid.csv', sep='\t', index_col=0, low_memory=False)
    y = valid['abstractive'][:args.max_row].values # 요약문 컬럼만 가져옴
    del valid
    print('### 원본 요약문 로드 완료, ', y.shape, type(y))
    return y
    
    
# 요약문 csv 파일을 읽는 함수
def read_summary(filename): 
    df = pd.read_csv(filename, sep='\t', low_memory=False)
    x = df['summary'][:args.max_row].values
    del df
    print('### ', filename ,' 요약문 로드 완료, ', x.shape, type(x))
    return x




# 실행부
y = read_valid() # 정답데이터 (valid.csv 원본데이터의 요약문) 를 읽어온다
    
# 요약문들이 들어있는 폴더에 있는 모든 csv 파일명을 list로 가져옴
summary_dir_list = sorted(glob.glob(args.summary_dir+'/*.csv')) 
print('### 폴더내 csv파일 명 : ', summary_dir_list)

# 요약문 별로 rouge 점수를 계산해서 결과csv에 저장한다.
for summary_path in summary_dir_list:
    print('### '+summary_path+' for 문 시작!!')
    
    # rouge 점수 계산을 위한 Rouge객체 새로 생성
    rouge = Rouge() 
    
    # rouge 점수 계산
    scores = rouge.get_scores(y, read_summary(summary_path), avg=True) 
    
    # 화면에 점수 출력
    print(format_rouge_scores(scores)) 
    
    # csv 파일에 결과 저장
    save_result(scores, filename=summary_path) 
    
    del rouge, scores # 메모리에서 사용안할 요소 삭제
    


