# 네이버 AI 해커톤 2018

### 소개
NAVER AI Hackathon 2018 예선 및 결선에서 사용했던 코드이고, private 기록을 갱신하지 못하였지만 시도한 방법도 제시하였다. 이번 해커톤에서 사용한 라이브러리는 `pytorch==0.3.0`이고, 네트워크는 [VDCNN](https://arxiv.org/abs/1606.01781)의 논문을 참고하였다.

### 환경 구성
Preproces에서 normalize를 시도하기 위해 [mecab-ko](https://bitbucket.org/eunjeon/mecab-ko)를 설치한 Docker 환경을 빌드하였다. 그러나, 지식인과 같은 키워드 추출이 아니고, 감성분석을 하는 측면에서 **문장의 normalize는 성능이 떨어졌다**.

편집이나 데이터 추가를 위해 외부 마운트 경로를 설정할 수 있고, `jupyter notebook --allow-root` 명령어를 실행하면 웹브라우저에서도 작업할 수 있다.

##### Docker for CPU
```script
docker run -it --name <컨테이너이름> -p 8888:8888 -v <마운트할경로>:/home hermits/korean-nlp:latest bash
```

##### Docker for GPU
```script
nvidia-docker run -it --name <컨테이너이름> -p 8888:8888 -v <마운트할경로>:/home hermits/korean-nlp:latest bash
```

### 구성
* **kor_short_char**: 가장 좋은 성능을 보여준 음소 단위의 전처리
* **kor_long_char**: 결선 동안 시도하였던 음절 단위의 전처리

### 실행
1. on nsml environment, example:
```
nsml run -d movie_final -a "--train_ratio 0.9"
```

2. on local environment, example:
```
# with GPU
python main.py --sample sample --train_ratio 0.9 --batch 10 --gpus 0
# without GPU
python main.py --sample sample --train_ratio 0.9 --batch 10
```

3. configuration parameters
```python
# DONOTCHANGE: They are reserved for nsml
args.add_argument('--mode', type=str, default='train')
args.add_argument('--pause', type=int, default=0)
args.add_argument('--iteration', type=str, default='0')
# User options
args.add_argument('--output', type=int, default=1)
args.add_argument('--epochs', type=int, default=20)
args.add_argument('--batch', type=int, default=128)
args.add_argument('--strmaxlen', type=int, default=100)
args.add_argument('--embedding_dim', type=int, default=192)
args.add_argument('--depth', type=int, default=29)
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--train_ratio', type=float, default=1.0)
args.add_argument('--sample', type=str, default='nsml')
args.add_argument('--gpus', type=int, default=2)
```

4. Best result: `2.79518`
```python
nsml run -d movie_final
```
`main.py`의 기본 parameters가 최상의 결과를 보여주도록 세팅하였다. `epoch=7~13`에서 최상의 결과를 기대할 수 있고, 제출 기준 `epoch=11`에서 결과를 얻을 수 있었다. nsml 환경기준(P40) `960sec/epoch`의 성능을 보여준다.

### Further more
CNN 네트워크로 바로 적용하지 않고, RNN 네트워크를 통한 결과를 CNN으로 보내는 방법을 테스트 예정, asap.

### 기타
1. [issues](https://github.com/GzuPark/naver-ai-hackathon-2018/issues) 탭을 통하여 질의를 받습니다.
2. Personal data는 제공하지 않습니다.
3. Data를 구성하려면, `sample_data`나 `data`의 구조를 참고해주세요.
4. `kor_long_char`에 사용한 음절(1951개)이 경우 Personal data에서 `빈도수 > 10`을 추출한 것이며, [extract.ipynb](./kor_long_char/extract.ipynb)에서 과정을 살펴 볼 수 있다.

### License
코드 및 파일은 기본 제공한 NAVER의 규약을 따르고, 네트워크 및 개발한 알고리즘의 경우 허락없이 사용할 수 있음.
```python
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
