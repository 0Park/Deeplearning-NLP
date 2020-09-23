df#%% use embedding layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sentences=['nice great best amazing','stop lies','pitful nerd','excellent work','super quality','bad','highly respectable']
y_train=[1,0,0,1,1,0,1]

# word dictionary 만들기
t=Tokenizer()
t=Tokenizer()
t.fit_on_texts(sentences)
vocab_size=len(t.word_index)+1

print(vocab_size)
#%% change text into sequences
X_encoded=t.texts_to_sequences(sentences)
print(X_encoded)
#%%  
max_len=max(len(l) for l in X_encoded)
print(max_len)
#%% padding sequences
X_train=pad_sequences(X_encoded,maxlen=max_len,padding='post')
y_train=np.array(y_train)
print(X_train)
#%% make model including embedding layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Flatten
model=Sequential()
model.add(Embedding(vocab_size,4,input_length=max_len))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
#%%
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit(X_train,y_train,epochs=100,verbose=2)
#%% use Glove pretrained embedding matrix
n=0
f=open('C:/Users/Young Hun Park/Desktop/python beginner/NLP/glove.6B.100d.txt',encoding="utf8")

for line in f:
    word_vector=line.split()
    print(word_vector)
    word=word_vector[0]
    print(word)
    n=n+1
    if n==2:
        break
f.close()
#%%
print(type(word_vector))
print(len(word_vector))
#%% 
import numpy as np
embedding_dict=dict()
f=open('C:/Users/Young Hun Park/Desktop/python beginner/NLP/glove.6B.100d.txt',encoding="utf8")

for line in f:
    word_vector=line.split()
    word=word_vector[0]
    word_vector_arr=np.asarray(word_vector[1:],dtype='float32')
    embedding_dict[word]=word_vector_arr
f.close()
print('%s개의 Embedding vector가 있습니다.'%len(embedding_dict))
#%%
print(embedding_dict['respectable'])
print(len(embedding_dict['respectable']))
#%%
embedding_matrix=np.zeros((vocab_size,100))
print(np.shape(embedding_matrix))
#%%
print(t.word_index.items())
#%%
for word,i in t.word_index.items():
    temp=embedding_dict.get(word)
    if temp is not None:
        embedding_matrix[i]=temp
        
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Flatten

model=Sequential()
e=Embedding(vocab_size,100,weights=[embedding_matrix],input_length=max_len,trainable=False)
model.add(e)
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit(X_train,y_train,epochs=100,verbose=2)

