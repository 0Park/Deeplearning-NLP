#%%
import re
from lxml import etree
import urllib.request
import zipfile
from nltk.tokenize import word_tokenize,sent_tokenize
#%%
urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
# 데이터 다운로드

with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
  target_text = etree.parse(z.open('ted_en-20160408.xml', 'r'))
  parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
#%% data preprocess
context_txt=re.sub(r'\([^)]*\)','',parse_text)
#this code remove content inside ()
sent_text=sent_tokenize(context_txt)
# sentence tokenize
normalized_text=[]
for string in sent_text:
    tokens=re.sub(r"[^a-z0-9]+"," ",string.lower())
    normalized_text.append(tokens)
    # remove the punctuation int the sentence and lower the alphabet
result=[word_tokenize(sentence) for sentence in normalized_text]
#%%
print("총 샘플의 개수:{}".format(len(result)))
for line in result[:3]:
    print(line)
#%% train Word2Vec
from gensim.models import Word2Vec
model=Word2Vec(sentences=result,size=100,window=5,min_count=5,workers=4,sg=0)
#%%
model_result=model.wv.most_similar("man")
print(model_result)
#%% save model
from gensim.models import KeyedVectors
model.wv.save_word2vec_format('eng_w2v')
loaded_model=KeyedVectors.load_word2vec_format("eng_w2v")
#%%
model_result=loaded_model.most_similar("man")
print(model_result)
#%% korean Word2Vec
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
#%%
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data=pd.read_table('ratings.txt')
#%%
print(train_data[:5])
#%%
print(len(train_data))
print(train_data.isnull().values.any())# check NULL
train_data=train_data.dropna(how='any')
print(train_data.isnull().values.any())
#%%ㅜ
print(len(train_data))
#%% remove the letters except korean
train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]","")
#%%
# define stopwords
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으','로','자','에','와','한','하다']

# tokenize using okt
okt=Okt()
tokenized_data=[]
for sentence in train_data['document']:
    temp_X=okt.morphs(sentence,stem=True)
    temp_X=[word for word in temp_X if not word in stopwords]
    tokenized_data.append(temp_X)
#%% 
print('리뷰의 최대 길이:',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이:',sum(map(len,tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data],bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
#%%
from gensim.models import Word2Vec
model=Word2Vec(sentences=tokenized_data,size=100,window=5,min_count=5,workers=4,sg=0)
#%%
print(model.wv.vectors.shape)
print(model.wv.most_similar("최민식"))
print(model.wv.most_similar("히어로"))
#%%
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
#%% get data
dataset=fetch_20newsgroups(shuffle=True,random_state=1,remove=('headers','footers','quotes'))
documents=dataset.data
print("총 샘플 수:",len(documents))
#%%
news_df = pd.DataFrame({'document':documents})
# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
#%%
print(news_df.isnull().values.any())
#%%
news_df.replace("",float("NaN"),inplace=True)
print(news_df.isnull().values.any())
#%%
news_df.dropna(inplace=True)
#%%
print('총 샘플 수:',len(news_df))
#%% remove stop words
stop_words=stopwords.words('english')
tokenized_doc=news_df['clean_doc'].apply(lambda x: x.split())
#%%
tokenized_doc=tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc=tokenized_doc.to_list()
#%% remove 1 character text
drop_train=[index for index,sentence in enumerate(tokenized_doc) if len(sentence) <=1 ]
tokenized_doc=np.delete(tokenized_doc,drop_train,axis=0)
print('총 샘플 수:',len(tokenized_doc))
#%%
tokenizer=Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)
word2idx=tokenizer.word_index
idx2word={v:k for k,v in word2idx.items()}
encoded=tokenizer.texts_to_sequences(tokenized_doc)
#%%
print(encoded[:2])
#%%
vocab_size=len(word2idx)+1
print('단어 집합의 크기:',vocab_size)
#%%
from tensorflow.keras.preprocessing.sequence import skipgrams

skip_gramms=[skipgrams(sample,vocabulary_size=vocab_size,window_size=10)for sample in encoded[:10]]
#%%
# check skipgram data
pairs,labels=skip_gramms[0][0],skip_gramms[0][1]
for i in range(5):
    print("({:s} ({:d}),{:s},({:d})) -> {:d}".format(
        idx2word[pairs[i][0]],pairs[i][0],
        idx2word[pairs[i][1]],pairs[i][1],
        labels[i]))
#%%
skip_grams=[skipgrams(sample,vocabulary_size=vocab_size,window_size=10) for sample in encoded]
#%%
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding,Reshape,Activation,Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
#%% 
embed_size=100

w_inputs=Input(shape=(1,),dtype='int32')
word_embedding=Embedding(vocab_size,embed_size)(w_inputs)

c_inputs=Input(shape=(1,),dtype='int32')
context_embedding=Embedding(vocab_size,embed_size)(c_inputs)

dot_product=Dot(axes=2)([word_embedding,context_embedding])
dot_product=Reshape((1,),input_shape=(1,1))(dot_product)
output=Activation('sigmoid')(dot_product)
#%%
model=Model(inputs=[w_inputs,c_inputs],outputs=output)
model.summary()
#%%
model.compile(loss='binary_crossentropy', optimizer='adam')
for epoch in range(1, 6):
    loss = 0
    for _, elem in enumerate(skip_grams):
        first_elem = np.array(list(zip(*elem[0]))[0], dtype='int32')
        second_elem = np.array(list(zip(*elem[0]))[1], dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [first_elem, second_elem]
        Y = labels
        loss += model.train_on_batch(X,Y)  
    print('Epoch :',epoch, 'Loss :',loss)
    