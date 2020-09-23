#%%
import json
with open('C:/Users/Young Hun Park/Downloads/KorQuAD_v1.0_train.json') as train_file:
    train_data=json.load(train_file)
#%%
train_data=train_data['data']
#%%
# make data into DataFrame
import pandas as pd
def json_to_df(data):
    arrayForDF=[]
    for current_subject in data:
        subject=current_subject['title']
        for current_context in current_subject['paragraphs']:
            context=current_context['context']
            for current_question in current_context['qas']:
                question=current_question['question']
                for answer in current_question['answers']:
                    answer_text=answer['text']
                    answer_start=answer['answer_start']
                    
                    record={
                        "answer_text":answer_text,
                        "answer_start":answer_start,
                        "question":question,
                        "context":context,
                        "subject":subject
                        
                    }
                    arrayForDF.append(record)
    df=pd.DataFrame(arrayForDF)
    return df
#%%
data_df=json_to_df(train_data)
#%%
from nltk.tokenize import sent_tokenize

def get_answer_context(df):
    length_context=0
    answer= ""
    for sentence in sent_tokenize(df.context):
        length_context+=len(sentence)+1
        if df.answer_start <= length_context:
            if len(sentence) >= len(str(df.answer_text)):
                if answer=="":
                    return sentence
                else:
                    return answer+""+sentence
            else:
                answer+=sentence
data_df['entire_answer_text']=data_df.apply(lambda row: get_answer_context(row),axis=1)
#%%
from konlpy.tag import Okt

twitter=Okt()
answer_text=data_df['entire_answer_text']
p_answer=[]
for answer in answer_text:
     #형태소 분석하기, 단어 기본형 사용
    malist = twitter.pos( answer, norm=True, stem=True)
    r=[]
    for word in malist:
         #Josa”, “Eomi”, “'Punctuation” 는 제외하고 처리
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
    
    p_answer.append(r)
#%%
def cleaningText(text):
    twitter=Okt()
    p_question=[]
    for sentence in text:
        malist=twitter.pos(sentence,norm=True,stem=True)
        r=[]
        for word in malist:
            if not word[1] in ["Josa","Eomi","Punctuation"]:
                r.append(word[0])
        p_question.append(r)
    
    return p_question
#%%
question_text=data_df['question']
#%%
question_text=cleaningText(question_text)
#%%
QA_df=pd.DataFrame(columns=['question','answer'])
#%%
QA_df['question']=question_text
QA_df['answer']=p_answer
#%%
QA_df.to_csv('C:/Users/Young Hun Park/Desktop/python beginner/NLP/QA_df.csv')
#%%
import pandas as pd
QA_df=pd.read_csv('C:/Users/Young Hun Park/Desktop/python beginner/NLP/QA_df.csv')

#%%
import gensim
model=gensim.models.Word2Vec.load('C:/Users/Young Hun Park/Desktop/python beginner/NLP/KorQuAD.model')

#%%
# integer Encoding
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer()
tokenizer.fit_on_texts(QA_df['question'])
q_sequences=tokenizer.texts_to_sequences(QA_df['question'])
#%%
q_word_index=tokenizer.word_index
q_vocab_size=len(q_word_index)+1

#%%
import matplotlib.pyplot as plt
print('문장의 최대 길이 :',max(len(l) for l in q_sequences))
print('문장의 평균 길이 :',sum(map(len, q_sequences))/len(q_sequences))
plt.hist([len(s) for s in q_sequences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
#%%
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len=100
q_sequences=pad_sequences(q_sequences,maxlen=max_len)
#%%
tokenizer=Tokenizer()
tokenizer.fit_on_texts(QA_df['answer'])
a_sequences=tokenizer.texts_to_sequences(QA_df['answer'])
a_word_index=tokenizer.word_index
a_vocb_size=len(a_word_index)+1

print('the length of the longest sentence :',max(len(l) for l in a_sequences))
print('the mean length of the sentences',sum(map(len,a_sequences))/len(a_sequences))
plt.hist([len(s) for s in a_sequences],bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
#%%
i=0
for s in a_sequences:
    if len(s)>100:
        i+=1
print(i)
print((381/60407)*100)
#%%
max_len=100
a_sequences=pad_sequences(a_sequences,maxlen=max_len)

#%%
import numpy as np
a_Embedding_matrix=np.zeros((a_vocb_size,200))
q_Embedding_matrix=np.zeros((q_vocab_size,200))

for word, i in a_word_index.items():
    word=word.replace("'","")
    try:
        embedding_vector=model[word]
    except KeyError:
        continue
    a_Embedding_matrix[i]=embedding_vector
#%%
for word, i in q_word_index.items():
    word=word.replace("'","")
    try:
        embedding_vector=model[word]
    except KeyError:
        continue
    q_Embedding_matrix[i]=embedding_vector
#%%
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional,MaxPool1D,Dropout,concatenate
from tensorflow.keras.models import Sequential

embedding_dim=200
max_len=100
hidden_size=50

Q_model=Sequential()
q=Embedding(q_vocab_size,embedding_dim,weights=[q_Embedding_matrix],input_length=max_len,trainable=True,mask_zero=True)
Q_model.add(q)
Q_model.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))
Q_model.add(MaxPool1D(max_len))
Q_model.add(Dropout(0.2))
Q_model.summary()

A_model=Sequential()
a=Embedding(a_vocb_size,embedding_dim,weights=[a_Embedding_matrix],input_length=max_len,trainable=True,mask_zero=True)
A_model.add(a)
A_model.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))
A_model.add(MaxPool1D(max_len))
Q_model.add(Dropout(0.2))
A_model.summary()

QA_model=Sequential()
QA_model.add(concatenate([Q_model,A_model]))
QA_model.summary()
#%%
