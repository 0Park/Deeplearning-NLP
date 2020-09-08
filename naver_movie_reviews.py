#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
#%%
train_data=pd.read_table('ratings_train.txt')
test_data=pd.read_table('ratings_test.txt')
#%%
print('훈련용 리뷰 개수:',len(train_data))
#%%
print(train_data.head())
#%% data preprocessing
train_data.drop(['id'],axis=1,inplace=True)
test_data.drop(['id'],axis=1,inplace=True)
#%% remove duplicate
print(train_data['document'].nunique(),train_data['label'].nunique())
train_data.drop_duplicates(subset=['document'],inplace=True)
#%%
print(len(train_data))
print(train_data['label'].value_counts())
#%% check null
print(train_data.isnull().sum())
#%% remove null
train_data=train_data.dropna(how='any')
print(len(train_data))
#%% remove character except the space and korean alphabet
train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
print(train_data[:3])
#%% check null
train_data['document'].replace('',np.nan,inplace=True)
print(train_data.isnull().sum())
#%%remove null
train_data=train_data.dropna(how='any')
#%% preprocess the test dat as train
test_data.drop_duplicates(subset=['document'],inplace=True)
test_data['document']=test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['document'].replace('',np.nan,inplace=True)
test_data=test_data.dropna(how='any')
print('after preprocessing:',len(test_data))
#%%
stopwords=['은','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# 형태소 분석기
okt=Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔',stem=True) # 어간 추출
#%%
X_train=[]
for sentence in train_data['document']:
    temp_X=[]
    temp_X=okt.morphs(sentence,stem=True)
    temp_X=[word for word in temp_X if not word in stopwords]
    X_train.append(temp_X)
#%%
print(X_train[:3])
#%%
X_test=[]
for sentence in test_data['document']:
    temp_X=[]
    temp_X=okt.morphs(sentence,stem=True)
    temp_X=[word for word in temp_X if not word in stopwords]
    X_test.append(temp_X)

#%% encoding
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
#%%
print(tokenizer.word_index)
#%% check word distribution
threshold=3
total_cnt=len(tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0

for key,value in tokenizer.word_counts.items():
    total_freq+=value
    
    if (value<threshold):
        rare_cnt+=1
        rare_freq+=value

print('the size of vocabulary:',total_cnt)
print('the number of word whose frequency is below three:',rare_cnt)
print('the ratio of the number of rare words:',(rare_cnt/total_cnt)*100)
print('the ratio of rare words frequency:',(rare_freq/total_freq)*100)
#%% remove the words of which frequency is below three
vocab_size=total_cnt-rare_cnt+2
print('단어 집합의 크기:',vocab_size)
#%%
tokenizer=Tokenizer(vocab_size,oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)
#%%
print(X_train[:3])
#%%
y_train=np.array(train_data['label'])
y_test=np.array(test_data['label'])
#%% remove empty samples because of the removed words
drop_train=[index for index,sentence in enumerate(X_train) if len(sentence)<1]
#%%
X_train=np.delete(X_train,drop_train,axis=0)
y_train=np.delete(y_train,drop_train,axis=0)
print(len(X_train))
print(len(y_train))
#%%
a=0
for sentence in X_train:
    if 1 in sentence:
        a+=1
print(a+411)
print(rare_cnt)
#%%
print('the max length:',max(len(l) for l in X_train))
print('the average length:',sum(map(len,X_train))/len(X_train))
plt.hist([len(s) for s in X_train],bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
#%%
def below_threshold_len(max_len,nested_list):
    cnt=0
    for s in nested_list:
        if(len(s)<=max_len):
            cnt=cnt+1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
#%%
max_len = 30
below_threshold_len(max_len, X_train)
#%%
X_train=pad_sequences(X_train,maxlen=max_len)
X_test=pad_sequences(X_test,maxlen=max_len)

#%%
from tensorflow.keras.layers import Embedding,Dense,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

model=Sequential()
model.add(Embedding(vocab_size,100))
model.add(LSTM(128))
model.add(Dense(1,activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
#%%
loaded_model=load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" %(loaded_model.evaluate(X_test,y_test)[1]))
#%%
def sentiment_predict(new_sentence):
    new_sentence=okt.morphs(new_sentence,stem=True)
    new_sentence=[word for word in new_sentence if not word in stopwords]
    encoded=tokenizer.texts_to_sequences([new_sentence])
    pad_new=pad_sequences(encoded,maxlen=max_len)
    score=float(loaded_model.predict(pad_new))
    if(score >0.5):
        print('{:.2f}% positive review\n'.format(score*100))
    else:
        print('{:2f}% negative review\n'.format((1-score)*100))
#%%
sentiment_predict('이 영화 개꿀잼')
#%%
sentiment_predict('이 영화 핵노잼 ㅠㅠ')
#%%
sentiment_predict('내가 만들어도 이것보다는 잘만들듯')
#%%
sentiment_predict('전체적인 영화의 줄거리는 아쉽지만 그래도 액션은 볼만했어.')
#%%
sentiment_predict('초호화 캐스팅이라 기대를 많이 했는데 영화는 영 아니네')
