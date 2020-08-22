#%% 1 
# 길이가 1~2인 단어들을 삭제
import re
text="I was wondering if anyone out there could enlighten me on this car"
shortword=re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('',text))

#%% 2
# 표제어 추출
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer()
words=['policy','doing','organization','have','going','love','lives','fly','dies','watched','has','starting']
print([n.lemmatize(w) for w in words])

#%% 3
#품사를 정보를 포함한 표제어 추출
print(n.lemmatize('dies','v'))
print(n.lemmatize('watched','v'))
print(n.lemmatize('has','v'))

#%% 4 
#어간 추출(proter 알고리즘)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s=PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate\
 copy, complete in all things--names and heights and soundings--with the single\
 exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)
print([s.stem(w) for w in words])

words=['formalize','allowance','electrical']
print([s.stem(w) for w in words])

#%% 5
# 어간 추출(Lancaster stemmer 알고리즘)
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
s=PorterStemmer()
l=LancasterStemmer()

words=['policy','doing','organization','have','going','love','lives','fly','dies','watched','has','starting']
print([s.stem(w) for w in words])
print([l.stem(w) for w in words])
# %% 6
from nltk.corpus import stopwords
stopwords.words('english')[:10]
# %%7
#불용어 제거(영어)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="Family is not an important thing. It's everything."
stop_words=set(stopwords.words('english'))
word_tokens=word_tokenize(example)

result=[]
for w in word_tokens:
    if w not in stop_words:
        result.append(w)
print(word_tokens)
print(result)

#%% 8
#불용어 제거(한글)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example="고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든.\
 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words="아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"

stop_words=stop_words.split(' ')
word_tokens=word_tokenize(example)

result=[word for word in word_tokens if not word in stop_words]

print(word_tokens)
print(result)
#%% 9
#정규 표현식
# . : 임의의 문자
import re
r=re.compile("a.c")
r.search("kkk")
r.search("abc")

# ?: 앞의 문자가 존재할 수도 있고 존재 안할수도 있음
r=re.compile("ab?c")
r.search("abbc")
r.search("abc")
r.search("ac")

# *: 앞의문자가 0개 또는 여러개 올경우
r=re.compile("ab*c")
r.search("a")
r.search("ac")
r.search("abc")
r.search("abbbbbc")

# +: *와 비슷한데 앞의문자가 최소 1개이상
r=re.compile("ab+c")
r.search("ac")
r.search("abc")
r.search("abbbbc")

# ^:시작되는 글자 지정
r=re.compile("^a")
r.search("bbc")
r.search("ab")

#{}: 해당문자를 숫자만큼 반복
r=re.compile("ab{2}c")
r.search("ac")
r.search("abc")
r.search("abbc")
r.search("abbbc")

# {}:숫자 2개 붙이면 이상 이하
r=re.compile("ab{2,8}c")
r.search("ac")
r.search("abbc")
r.search("abbbbbbc")


#{숫자,}
r=re.compile("a{2,}bc")
r.search("bc")
r.search("aabc")
r.search("aaaaabc")

# []: 문자들 중에 한개 매치
r=re.compile("[abc]")#a-c와 같음
r.search("zzz")
r.search("a")
r.search("aaaaaa")
r.search("baac")

r=re.compile("[a-z]")
r.search("AAA")
r.search("aBC")

#[^문자] : 뒤에 붙은 문자들을 제외한 모든 문자 매치
r=re.compile("[^abc]")
r.search("a")
r.search("ab")
r.search("d")
r.search("1b")
        
# match와 search의 차이
r=re.compile("ab.")
r.search("kkkkkabc")

r.match("kkkabc")
r.match("abckk")

# split
text="사과 딸기 수박 메론 바나나"
re.split(" ",text)

text="사과+딸기+수박+메론+바나나"
re.split("\+",text)

# findall()
text="이름: 김철수\
 전화번호: 010-1234-1234\
 나이: 30\
 성별: 남"""
re.findall("\d+",text)

# re.sub() 문자열 대체
text="Regular expression : A regular expression, regex or regexp[1] (sometimes\
 called a rational expression)[2][3] is, in theoretical computer science and\
 formal theory, a sequence of characters that define a search pattern."
 
re.sub('[^a-zA-Z]',' ',text)

# example
text="""100 John   PROF\
 101 James   STUD
 102 Mac    STUD"""
re.split('\s+',text)  # 빈공간
re.findall('\d+',text)
re.findall('[A-Z]',text)
re.findall('[A-Z]{4}',text)
re.findall('[A-Z][a-z]+',text)

letters_only=re.sub('[^a-zA-Z]',' ',text)

#%% 10
#정규 표현식을 이용한 토큰화
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as\
                          cheery goes for a pastry shop"))
tokenizer=RegexpTokenizer("[\s]+",gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as\
                          cheery goes for a pastry shop"))
#%% ch02-6 정수 인코딩

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#문장 토큰화
text="A barber is a person. a barber is good person. a barber is huge person.\
 he Knew A secret! The Secrete He Kept is huge secret. Huge secret. His barber\
 kept his word. a barber kept his word. His barber kept his secret. But keeping\
 and keeping such a huge secrete to himself was driving the barber crazy. the\
 barber went up a huge mountain."
text=sent_tokenize(text)
print(text)
# 정제와 단어 토큰화
vocab={}
sentences=[]
stop_words=set(stopwords.words('english'))

for i in text:
    sentence=word_tokenize(i)
    result=[]
    
    for word in sentence:
        word=word.lower()
        if word not in stop_words:
            if len(word)>2:
                result.append(word)
                if word not in vocab:
                    vocab[word]=0
                vocab[word]+=1
    sentences.append(result)
print(sentences)
print(vocab)
print(vocab["barber"])
#빈도수대로 정렬
vocab_sorted=sorted(vocab.items(),key= lambda x:x[1], reverse=True)
print(vocab_sorted)
#인덱스 부여
word_to_index={}
i=0
for (word,frequency) in vocab_sorted:
    if frequency > 1:
        i=i+1
        word_to_index[word]=i
print(word_to_index)
# 빈도수 상위 5개
vocab_size=5
words_frequency=[w for w,c in word_to_index.items() if c>vocab_size] # 인덱스 5초과인 던어 제거
for w in words_frequency:
    del word_to_index[w]
print(word_to_index)

word_to_index['OOV']=len(word_to_index)+1
encoded=[]
for s in sentences:
    temp=[]
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)

# Counter 사용하기
from collections import Counter
print(sentences)

words=sum(sentences, [])
print(words)
vocab=Counter(words)
print(vocab)
print(vocab["barber"])
vocab_size=5
vocab=vocab.most_common(vocab_size)
vocab
word_to_index={}
i=0
for (word,frequency) in vocab:
    i=i+1
    word_to_index[word]=i
print(word_to_index)

#%% NLTK의 FreqDist 사용하기
from nltk import FreqDist
import numpy as np

vocab=FreqDist(np.hstack(sentences))
print(vocab["barber"])

vocab_size=5
vocab=vocab.most_common(vocab_size)
vocab
word_to_index={word[0] : index+1 for index,word in enumerate(vocab)}
print(word_to_index)

#%% keras의 텍스트 전처리
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(sentences))
vocab_size=5
tokenizer=Tokenizer(num_words=vocab_size+1)
tokenizer.fit_on_texts(sentences)
print(tokenizer.texts_to_sequences(sentences))

tokenizer=Tokenizer(num_words=vocab_size+2, oov_token='OOV')
tokenizer.fit_on_texts(sentences)
print('단어 oov의 인덱스: {}'.format(tokenizer.word_index['OOV']))
print(tokenizer.texts_to_sequences(sentences))

#%% padding

# Numpy로 padding
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)
encoded=tokenizer.texts_to_sequences(sentences)
print(encoded)
max_len=max(len(item) for item in encoded)
print(max_len)

for item in encoded:
    while len(item)<max_len:
        item.append(0)
padded_np=np.array(encoded)
padded_np

# keras로 padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded=tokenizer.texts_to_sequences(sentences)
print(encoded)
padded=pad_sequences(encoded)
padded
padded=pad_sequences(encoded,padding='post')
padded
padded=pad_sequences(encoded,padding='post',maxlen=5)
padded

#%% One hot encoding
from konlpy.tag import Okt
okt=Okt()
token=okt.morphs("나는 자연어 처리를 배운다")
print(token)

word2index={}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca]=len(word2index)
print(word2index)

def one_hot_encoding(word,word2index):
    one_hot_vector=[0]*(len(word2index))
    index=word2index[word]
    one_hot_vector[index]=1
    return one_hot_vector

one_hot_encoding("자연어",word2index)
# keras로 one hot encoding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"
t=Tokenizer()
t.fit_on_texts([text])
print(t.word_index)

sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded=t.texts_to_sequences([sub_text])[0]
print(encoded)
one_hot=to_categorical(encoded)
print(one_hot)

#%% 데이터의 분리
X,y=zip(['a',1],['b',2],['c',3])
print(X)
print(y)

sentences=[['a',1],['b',2],['c',3]]
X,y=zip(*sentences)
print(X)
print(y)
# 데이터프레임을 이용하여 분리하기
import pandas as pd
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns=['메일 본문','스팸 메일 유무']

df=pd.DataFrame(values,columns=columns)
df

X=df['메일 본문']
y=df['스팸 메일 유무']

#skitlearn 분리하기
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
