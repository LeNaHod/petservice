텍스트 요약 코드 실행순서


1. 데이터준비.ipynb 파일을 통해 6개의 json파일을 train/valid 두개의 csv로 변환한다.


2. train.py 실행해서 모델 학습시키기
>>  cmd 실행 명령어 : python train.py --gradient_clip_val 1.0 --max_epochs 5 --default_root_dir 3e5_batch16_logs --batch_size 16 --lr 3e-5 --num_workers 4 --gpus 1 --checkpoint_path ./checkpoint --max_len 256


3. get_model_binary.py 실행해서 저장된 checkpoint로 모델 바이너리 파일 만들기
>>  cmd 실행 명령어 : python get_model_binary.py --log_dir 3e5_batch16_logs


4. make_summary_csv.py 실행해서 저장된 바이너리 모델파일로 요약문 생성하기
>>  cmd 실행 명령어 : CUDA_VISIBLE_DEVICES=0 python make_summary_csv.py --model_dir 3e5_batch16_model --output_dir summary_all/ --start_num 0


5. rouge_score.py 실행해서 저장된 요약문 파일들의 rouge 점수 산출하기
>>  cmd 실행 명령어 : python rouge_score.py --summary_dir [폴더명] --output [결과csv파일명] --max_row 5000
