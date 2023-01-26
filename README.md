# Project3: 반려동물 안구질환 판별서비스 (슬기로운 집사생활)

## Shot Services provided

반려동물의 이미지를 업로드하면, 사진속 반려동물의 안구를 분석하여 안구질환여부를 판단하는 웹 서비스를 제공하고,

반려동물을 키울때 알아두면 유용한상식이나 반려동물 복지에대한 뉴스를 요약하여 사용자들에게 제공한다.

※업로드 파일 유의사항
- 보안상 일부 데이터와 코드만올림

# Main
![서비스 main페이지 소개](/project_3_main_gif.gif)

## Why?

반려동물 인구가 증가함에따라, 반려동물복지와 의료서비스에대한 관심도가 높아지는 추세이다.
그에맞춰 발병률이 높은 질병인 백내장은 초기발견이 중요하지만, 검사비가 비싼 동물병원의 특성상 자주 방문하는것이 부담스러울것을 고려하여, 우리는 집에서도 언제든 반려동물의 백내장을 판별할수있는 AI분석
서비스를 기획하게되었습니다.

## The ultimate goal(목표)

1. 사진으로 백내장 판별을 할수있는 서비스
2. 반려동물 관련 뉴스 요약문 제공
3. 근처 동물병원 위치 표시 및 검색 기능

## Team

Leard(DE): PM,파이프라인 구축, 서버구축(GCP기반),카프카를통한 트위터 실시간데이터 수집, 워드클라우드 생성

Member1(DS):뉴스기사 요약 서비스 모델링(KoBART)

Member2(DS): 백내장 판별 서비스 모델링 및 최적화(EfficientNet)

Member3(DE,ME):서버구축(GCP기반),ELK 데이터마켓구축,ELK를통한 키워드로 워드클라우드 생성

Member4(DS): 백내장 판별 서비스 모델링(VGG16), 뉴스기사 요약 서비스 모델링(KoBART)

Member5(DS): 백내장 판별 서비스 모델링 (ResNet), 뉴스기사 요약 서비스 모델링(TextRank)

## ERD

![ERD](/petservice_erd.png/)

## WBS

![WBS1](/WBS1.PNG)

![WBS2](/WBS2.PNG)


## 데이터 명세서
![Data statement](/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%AA%85%EC%84%B8%EC%84%9C2%EC%B0%A8_1.PNG)

![Data statement](/%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%AA%85%EC%84%B8%EC%84%9C2%EC%B0%A8_2.PNG)

## 개발환경

### 웹서비스
- Python
- Django
- html, css, js

### 데이터 분석
- Jupyter
- Colabortory(google)
- PyTorch
- TensorFlow

### 서버구축
- GCP(Google Cloud Platform)
- aws

### 데이터 수집
- Selenium
- Kafka
  
### 데이터 처리
- Elasticsearch
- Logstash
- Spark
- Airflow

### 데이터 적재
- Hadoop
- Mysql
- Zookeeper
- Mongo DB

## 아키텍쳐 정의서

![Architecture definition](/%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98%EC%A0%95%EC%9D%98%EC%84%9C.PNG)

## 서비스환경 버전

- Ubuntu 18.04
- Python 3.13.6
- Django 4.1.5
- mysql-connector-python 8.0.31
- mysqlclient 2.1.1
- tensorflow 2.9.2
- torch 1.11
- opencv-python 4.7.68
- protobuf 3.22.1
  

# Result
[슬기로운 집사생활 서비스 배포링크](http://xn--ok0by6qo0gvfq9f86io3f972a.xn--h32bi4v.xn--3e0b707e/)