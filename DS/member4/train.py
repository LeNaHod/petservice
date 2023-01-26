# train.py
# kobart 학습을 진행하는 코드

# import
import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KobartSummaryModule
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--checkpoint_path', # 체크포인트를 저장할 경로를 지정한다
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger() # 로그를 저장할 logger 객체 생성
logger.setLevel(logging.INFO)

class ArgsBase(): # 파일 실행시 입력받을 arguments들을 정의한다.
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',                 # train.csv 파일의 위치를 지정한다.
                            type=str,
                            default='../dataset/train.csv',
                            help='train file')

        parser.add_argument('--test_file',                 # valid.csv 파일의 위치를 지정한다.
                            type=str,
                            default='../dataset/valid.csv',
                            help='test file')

        parser.add_argument('--batch_size',                 # 배치 사이즈를 지정한다.
                            type=int,
                            default=16,
                            help='')
        parser.add_argument('--max_len',                 # 최대 문장수를 지정한다.
                            type=int,
                            default=256,
                            help='max seq len')
        return parser

class Base(pl.LightningModule): # pytorch_lightning 객체를 상속받는 클래스를 정의한다.
    def __init__(self, hparams, trainer, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams) # 학습에 사용될 하이퍼파라미터를 저장할 변수 정의
        self.trainer = trainer # 학습에 사용될 trainer 객체 정의

    @staticmethod
    def add_model_specific_args(parent_parser): # 입력받을 arguments 정의
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',                 # 배치 사이즈를 지정한다.
                            type=int,
                            default=16,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',                                  # 학습률을 지정한다.
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',          # warmup 비율을 지정한다.
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',               # 모델이 저장된 경로를 지정한다.
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser
    
    def setup_steps(self, stage=None):  # data loader 객체를 통해 데이터 개수를 리턴한다.
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader() 
        return len(train_loader)

    def configure_optimizers(self): # optimizer 설정
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, # adam w 옵티마이저 함수를 사용
                          lr=self.hparams.lr, correct_bias=False)          # 학습률 설정
        num_workers = self.hparams.num_workers                   # num_workers 설정

        data_len = self.setup_steps(self)                                         # 데이터 로더 객체를 통해 데이터의 수를 받아온다.
        logging.info(f'number of workers {num_workers}, data length {data_len}') # 로그 저장
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs) # 에폭과 배치사이즈를 통해 step 사이즈를 정한다.
        logging.info(f'num_train_steps : {num_train_steps}') # 로그저장
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio) # warmup step 사이즈를 정한다.
        logging.info(f'num_warmup_steps : {num_warmup_steps}') # 로그저장
        scheduler = get_cosine_schedule_with_warmup( # 옵티마이저와 step 사이즈들을 스케줄러에 담는다.
            optimizer,
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler] # 옵티마이저와 스케줄러 리턴


class KoBARTConditionalGeneration(Base): # kobart 모델 학습을 진행할 클래스 정의
    def __init__(self, hparams, trainer=None, **kwargs): # 변수 정의
        super(KoBARTConditionalGeneration, self).__init__(hparams, trainer, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1') # 사전학습된 kobart 모델 로드
        self.model.train()
        self.bos_token = '<s>' # 문장 시작 토큰을 지정
        self.eos_token = '</s>' # 문장 종료 토큰을 지정
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1') # kobart 토크나이저 모델 로드
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, inputs): # forward 함수 생성

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx): # 학습 step 함수 생성
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): # 검증 step 함수 생성
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs): # 검증 종료 후 실행되는 함수 생성
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

if __name__ == '__main__':
    # parser 변수에 arguments를 추가한다.
    parser = Base.add_model_specific_args(parser) 
    parser = ArgsBase.add_model_specific_args(parser) 
    parser = KobartSummaryModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    
    # 토크나이저 로드
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    args = parser.parse_args() # arguments 들을 사용하기 위해 args 변수에 저장
    logging.info(args) # 로그 저장

    # KobartSummaryModule 객체 생성
    dm = KobartSummaryModule(args.train_file, # train.csv 경로
                        args.test_file,                                         # valid.csv 경로
                        tokenizer,                                                # 토크나이저
                        batch_size=args.batch_size,            # 배치사이즈 지정
                        max_len=args.max_len,                      # 최대 문장수 지정
                        num_workers=args.num_workers) # workers 수 지정
    
    # 콜백 함수 지정 (checkpoint)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', # val loss가 적은 순으로 3개의 checkpoint 를 저장한다.
                                                       dirpath=args.default_root_dir,                               # checkpoint 저장 경로 지정
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}', # checkpoint 저장할 이름 지정
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=3)

    # 로그 저장
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    # trainer 객체 생성
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, # 콜백, 로그 객체를 설정한다
                                            callbacks=[checkpoint_callback, lr_logger])

    model = KoBARTConditionalGeneration(args, trainer) # kobart 생성 모델을 로드한다.
    trainer.fit(model, dm) # trainer 객체에 모델과 데이터셋을 전달해 학습시킨다. 

