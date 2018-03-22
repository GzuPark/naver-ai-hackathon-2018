# 네이버 AI 해커톤 2018

### 1. 해커톤 안내
[[링크]](https://github.com/naver/ai-hackathon-2018/blob/master/README.md)
[[YouTube]](https://youtu.be/cSGPHtzPFQw)

### 2. 참가 신청 폼
[[링크]](http://naver.me/GyfLHzwg)

### 3. 일정

| 항목    | 일정                        |
|:------:|:--------------------------:|
| 참가 신청 | 18. 3. 12 (월) ~ 3. 25 (일) |
| 예선 1R | 18. 4. 2 (월) ~ 4. 8 (일)   |
| 예선 2R | 18. 4. 9 (월) ~ 4. 15 (일)  |
| 결선    | 18. 4. 26 (목) ~ 4. 27 (금) |

### 4. 미션
* 네이버 지식iN 질문 유사도 예측
* 네이버 평화 평점 예측
* 상세 설명은 4월 2일 오전 공개

### 5. 심사 기준
* 예선 1R: NSML 리더보드 순위로 2R 진출, 자동 컷오프
* 예선 2R: NSML 리더보드 순위로 결선 진출
* 결선: 1박 2일, NSML 리더보드 순위

### 6. NSML
* Naver Smart Machine Learning [[링크]](https://alpha.nsml.navercorp.com/)

### 7. 문의
* [FAQ](https://github.com/naver/ai-hackathon-2018/blob/master/FAQ.md)
* [dl_ai_hackathon_2018@navercorp.com](mailto:dl_ai_hackathon_2018@navercorp.com)

---

# Abhishek NLP tutorial

### 소개
[Kaggle](https://www.kaggle.com/)의 여러 NLP competition 에서 높은 순위를 기록하는 [Abhishek Thakur
](https://www.kaggle.com/abhishek) 가 공개한 자신의 노하우를 튜토리얼로 보여준 [코드](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle/notebook)가 있다. 이 튜토리얼이 [Docker 환경](https://hub.docker.com/r/hermits/korean-nlp/) 에서 문제없이 돌아가는 것을 확인하였고, `Approaching_Any_NLP_Problem_on_Kaggle.ipynb` 에 결과를 볼 수 있다.

### 필수 다운로드
1. Data: [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification/data)

##### 선택
1. Word2vec: [glove.840B.300d](http://www-nlp.stanford.edu/data/glove.840B.300d.zip) (2GB)
2. Word2vec: [glove.42B.300d](http://www-nlp.stanford.edu/data/glove.42B.300d.zip) (1.7GB)

### Docker 구성 방법
```
docker pull hermits/korean-nlp:1.0
docker run -p 8888:8888 -p 6006:6006 -v <공유경로>:/notebooks -it hermits/korean-nlp:1.0 bash
jupyter notebook --allow-root --NotebookApp.token=''
```
공유경로란, 로컬 컴퓨터와 docker 이미지와 공유하는 폴더로써 데이터의 이동이 편리한 장점이 있어서 사용했다.
공유경로 아래 다운받은 data를 `input` 디렉토리를 생성하여 넣어두고, Word2vec을 경로에 압축을 풀어놓으면 된다.

### 주의
1. docker 이미지 용량의 이유로 Word2vec에 쓰이는 데이터는 따로 다운로드를 받아야한다.
2. Only CPU로 구성을하여 Keras의 훈련시간은 매우 길다. nvidia-docker를 사용한다면, GPU 설정을 따로 해주면 된다.

---

# soynlp tutorial

### 소개
[soynlp](https://github.com/lovit/soynlp)란, [김현중](https://github.com/lovit)씨가 개발한 한국어 분석을 위한 pure python code 이다. KoNLPy 에서 찾을 수 없는 단어를 찾을 때 유용하게 쓰인다.

### 설치
```
# only for python3
pip install soynlp==0.0.41
```

### Docker 구성 방법
* 위에 설명한 **Abhisheck NLP tutorial** 을 참고 [[링크]](https://github.com/GzuPark/naver-ai-hackathon-2018#abhishek-nlp-tutorial)

### 실행
1. WordExtractor: jupyter notebook 에서 `soynlp/sonlp_wordextractor.ipynb` 실행
