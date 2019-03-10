# ko_novel_generator
### Deep learning model writing korean novel
### 한글 소설을 생성하는 딥러닝 모델

Lablup(래블업) - [Just model it 이벤트](https://events.backend.ai/just-model-it/) 참가하여 진행한 프로젝트입니다.


### 블랙오리바나나 팀
<img src="https://github.com/IllgamhoDuck/ko_novel_generator/blob/master/blackoribanana.png" width="300">

웹사이트 기반으로 사람들이 함께 참여하여 소설을 작성하는 인공지능을 학습하고 이를 활용하여 인공지능과 함께 릴레이 소설을 써내려가는 것을 목적으로 하는 프로젝트입니다. 그리고 이 프로젝트의 핵심적인 부분인 한글 소설을 작성하는 딥러닝 모델을 구축하고자 합니다.

#### 1. 인간들의 참여 - 학습 데이터 선정 및 결과물 생성

> 사람들이 데이터로 사용할 소설을 고르는 과정부터 참여하여 최종적으로 만들어진 인공지능을 활용하여 인공지능이 소설을 작성할시 선택할 키워드와 문장 길이 수도 참여하여 선택하게 됩니다.   
> 여러 사람이 다양한 선택지를 제시하면 최종 선택은 사람들의 투표를 통해서 결정하게 됩니다.  
> 이를 통해 사람들이 직접 데이터를 고르는 과정부터 학습된 인공지능을 활용하여 결과물을 만들어내는 모든 과정에 함께 참여하여 다함께 하나의 소설을 완성해나가는 프로젝트입니다.

#### 2. 릴레이 소설 작성 방식 - 인간 -> 인공지능 턴 방식

> 릴레이 소설처럼 소설의 일부를 작성후 새로운 키워드로 이어서 소설을 써내려가며 이를 반복하여 소설을 점차점차 완성시켜나가는 방식입니다.  
> 또한, 소설을 써내려가는 도중에 새로운 소설을 학습시켜 인공지능이 다양한 스타일의 소설을 소화할 수 있도록 하고자 합니다.  

성능이 가장 우수한 모델을 만드는 것을 목표로 하지 않습니다.  
인공지능을 모르는 인간들이 인공지능을 체험하고 즐길 수 있는 것을 목표로 하고 있습니다.  

## 어떤 계기로 만들었는지
인공지능이란 용어가 인간들에게 익숙해진지 오래지만, 인공지능이 직접 활용해볼 수 있는 곳은 한정적입니다. 일상 생활에서 인공지능 기술이 녹아들어있지만 이를 인지하기 힘듭니다. https://experiments.withgoogle.com/ 와 같은 대표적인 인공지능 활용 웹사이트는 있지만, 학습된 인공지능을 사용할 수 있을 뿐입니다. 물론 인공지능을 자기 입맛대로 직접 학습하는 것은 쉽습니다.

다양한 딥러닝 프레임워크의 출현으로 인공지능 기술은 인간들이 사용하기에 그 어느 때보다 용이한 시대에 와있습니다. 하지만 개발자가 아닌 이상 직접 활용하기는 역시 힘듭니다. 개발을 전혀 모르는 일반인들도 인공지능을 직접 학습시키고 활용할 수 있는 곳이 보이지 않았기에 만들어야 겠다는 생각을 하게 되었습니다.

잠깐 즐기고 끝나는 것이 아니라, 지속적으로 참여하여 인공지능을 직접 제작하고 체험할 수 있게 하고 싶었으며, 여러가지 주제 중에서 소설 작성 인공지능이 지속적으로 사람들이 참여할 수 있다는 점에서 적합하다고 여겨 참여형 소설 인공지능 작성 프로젝트를 기획하게 되었습니다.

## 어떻게 만들었는지
#### 1. 현재 적용된 모델
현재 사용되고 있는 모델은 글자 단위 `GRU(Gated Recurrent Neural network)` 모델입니다. 한 글자씩 한글을 입력받아 다음 한글을 예측하여 내놓는 방식으로 매우 간단한 원리이며, 이는 임시로 사용하는 모델로 추후 더 나은 모델을 구축하여 성능을 끌어올리고자 합니다.

![lstm](https://github.com/IllgamhoDuck/DLND/blob/master/intro-to-rnns/assets/charRNN%400.5x.png)

저희는 미리 제작된 다양한 RNN모델을 활용하여 구조를 이해하고 성능을 테스트 한 후, 자체적인 커스텀 모델을 제작하고자 합니다.

#### 2. 데이터
인터넷에 존재하는 소설 1,000여편을 크롤링하여 학습 데이터로 활용하였습니다.

#### 3. 결과
학습은 정상적으로 진행되며 소설을 그대로 복제하기 시작합니다. 여러 소설을 학습하면 다양한 문체를 학습할 것으로 기대했으나, 당장 학습하고 있는 소설의 문체에 적응하기 시작합니다. 아직까지는 입력하는 문장에 적절히 대응하는 모델을 만들지 못했으며, 이는 Open ai의 gpt-2모델을 연구하면서 그 방법을 찾아보고자 합니다.

## 파일 구성
1. `data/` - 학습시킬 소설 데이터를 집어넣는 폴더
2. `generate/` -  생성된 소설 데이터가 저장되는 폴더
3. `save/` - 학습된 모델 parameter값을 저장하는 폴더
4. `vocab/` - 생성된 소설 사전 데이타를 저장하는 폴더
5. `data_loader` - 데이터를 전처리한 후 batch단위로 생성
6. `generate.py` - 학습된 모델을 바탕으로 소설을 생성
7. `model.py` - Pytorch GRU 모델 코드
8. `multi_train.py` - 여러 소설을 연속으로 학습시키고자 할 때 사용. `train.py`를 기반으로 작성.
9. `opt.py` - Option의 약자로, 전반적인 딥러닝 학습에 관련한 중요한 변수들을 저장
10. `train.py` - 소설 학습
11. `vocab_generator.py` - 소설을 바탕으로 사전 생성
  

## 사용 방법
사용 방법은 매우 간단합니다. 소설 전처리 과정을 제외하면 말이죠. 그 소설 전처리 코드는 아직 올려지지 않았기에 직접 하셔야 합니다.

1. 폴더 생성
`mkdir save generate save vocab`
2. 소설 전처리
소설 모든 문장과 문장 사이에 구분자(delimiter) `</s>`를 집어넣어야 합니다. 이쯤에서 번거로움을 느끼셨다면 맞습니다. 그 악명높은 정규표현식을 시작하실 때입니다.

- 다음의 사이트를 추천합니다. https://www.regexpal.com/

3. save 폴더에 소설 텍스트 집어넣기
- `save` 폴더에 원하는 소설을 집어넣습니다.
4. 단어 사전 생성
`python vocab_generator.py`
5. 학습 시작
`python multi_train.py`
6. 소설 생성
`python generate.py --epoch [int] --prime [str] --len [int] --resume [bool]`
- `epoch` - 학습된 모델 중 몇 번째 epoch 모델을 사용할 것인지 정합니다.
- `prime` - 입력 문구를 정합니다. 문구를 입력하면 적합한 문구를 이어서 생성해냅니다.
- `len` - 생성할 문장 길이를 정합니다.
- `resume` - True로 설정하면, 이전에 생성했던 문구에 이어서 생성합니다.

#### 소설 생성 과정 좀 더 상세 설명
1. 입력 - `python generate --epoch 20 --prime ""안녕 난 오리라고 해."" --len 10`
2. 출력 - `"난 거위라 해!"`
3. 입력 - `python generate --epoch 20 --prime ""거위?"" --len 10 --resume True`
4. 출력 - `""맞아 난 거위야""`

resume은 첫 입력에는 default가 False이기에 별도로 신경쓸 필요가 없습니다. 이후 두 번째 문구 생성시 앞에 문구에 이어서 하고 싶다면 `--resume True`라고 설정하면 됩니다. 여기서는 총 네 개의 문장이 이어서 작성되었습니다.

- "안녕 난 오리라고 해."
- "난 거위라 해!"
- "거위?"
- "맞아 난 거위야"

생성된 출력 텍스트 파일은 `generate`폴더의 `result.txt`에 저장되어 있습니다.



## API(Flask)
ko_novel_generator를 web에서 사용하기 위한 API 서버 (python Flask)

#### API LIST
- `put_Human_txt (get, post)`  
사용자의 텍스트를 입력하여 추가 학습 후, 이어서 text를 generate함  
  ###### paramters
  1. `contents_id` : 사용자가 web에서 입력한 내용(contents)의 id
  2. `is_first` : 최초 작성여부, True일 경우 신규 학습, False일 경우 이어서 학습함    
  ###### result
  생성한 text를 DB에 저장, return값은 없음
  
#### API 사용 방법
###### depengency
  - `flask` / `flask_restful` / `flask_script` / `flask_migrate`
  - `sqlalchemy`
  - `marshmallow`
  - `pytorch`  
###### execute
1. `config.py`  
  `SQLALCHEMY_DATABASE_URI`에 DB 정보 입력
2. `run.py`  
  host(`YOUR_LOCAL_HOST`)에 호스트 정보 입력 후 실행
  
  

## 그외 테스트해본 모델
#### l2w(Learning to Write) - https://github.com/ari-holtzman/l2w
> `word 단위로 학습`시키는 `GRU` 방식의 모델입니다. `Adaptive softmax`를 적용하고 있고 decoder를 이용하여 결과물을 생성하기 위한 `Beam search`시 **4개의 discriminator**를 통하여 점수를 매겨, 반복적이지 않고, 문맥과 기존 소설 스타일에 부합하는 문장을 작성해나가는 모델입니다.

**4개의 discriminator**
- `Repetition`
- `Entailment`
- `Relevance`
- `Lexical Style`

 **[SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/)** 를 이용하여서 `Entailment discriminator`를 학습하여야 하는데, 이는 영문 데이터라 이와 유사한 목적을  한글 데이터를 찾지 못해 모델 사용을 포기하였습니다. Beam search에서의 discriminator적용을 제외한 나머지 부분은 테스트 해보았습니다.

#### Seq2seq attention - https://github.com/IBM/pytorch-seq2seq
> 문장을 받아서 문장을 예측하여 출력하는 `seq2seq`에 `attention`이 적용된 모델입니다. RNN을 사용하긴 하지만, 문장 대 문장 단위로 기억이 전달되기 때문에 번역과 같은 작업에 적합합니다.

seq2seq 모델은 학습을 지속하면서 hidden state(기억)을 전달하지 않기에 소설과 같이 연속적으로 쓰는 작업에는 적합하지 않습니다. 소설에 적합한 형태로 만드려면 hidden state가 seq2seq 모델 간 전달되도록 수정하여야 합니다. seq2seq attention은 미리 학습된 문장에는 적절한 소설을 작성하였지만, 처음보는 문장에는 의미를 알 수 없는 문구만을 내놓았습니다.

#### Transformer - https://github.com/JayParks/transformer
![transformer](https://cdn-images-1.medium.com/max/1200/0*plM2xPXX4TppwUeQ.)
> 구글에서 우수한 성능의 번역기를 만드는데 많은 도움을 제공한 `transformer`입니다. `RNN`방식과 `CNN`방식이 아니며, 둘의 우수한 장점을 각각 취득하여 모델을 만들었습니다. `Self-attention`이라는 기존의 `attention`과는 다른 방식의 알고리즘입니다. `Seq2Seq`처럼 장기기억이 존재하지 않기에 소설과 같은 장기기억이 필요한 일보다는, 번역기, 리뷰 분석 등과 같이 즉석에서 판단을 내리는 업무에 적합합니다.

적용한 결과 학습은 진행이 되지만, 결과물을 생성하는 단계에서 무의미한 결과를 내놓습니다. 장기기억을 추가해야지 정상적으로 작동할 것으로 기대됩니다.

## 어떤 부분을 더 해보고 싶은지
#### Open AI GPT-2 - https://github.com/openai/gpt-2
최근 OpenAI가 공개하면서 화두가 되고 있는 GPT-2모델입니다. 작문 성능이 매우 우수하여 가짜 뉴스와 같이 잘못 활용될 것을 우려하여서 비공개로 결정되고 간단한 모델만을 공개하였습니다.

이 모델을 연구하여서 릴레이 소설 작성이 인간들이 주고받는 것처럼 자연스러운 인공지능 모델을 만들고자 합니다.
