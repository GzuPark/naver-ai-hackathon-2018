# soynlp tutorial

### 소개
[soynlp](https://github.com/lovit/soynlp)란, [김현중](https://github.com/lovit)씨가 개발한 한국어 분석을 위한 pure python code 이다. KoNLPy 에서 찾을 수 없는 단어를 찾을 때 유용하게 쓰인다.

### 설치
```
# only for python3, you should check pip: pip or pip3.
pip install soynlp==0.0.41
```

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

### 실행
1. WordExtractor: jupyter notebook 에서 `soynlp/sonlp_wordextractor.ipynb` 실행
