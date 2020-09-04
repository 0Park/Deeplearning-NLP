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

