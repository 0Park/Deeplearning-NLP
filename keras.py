#%% mnist
from keras.datasets import mnist
from keras import models,layers
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

network=models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255*0.99+0.01

test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255*0.99+0.01

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=5,batch_size=128)
#%% test accuracy
test_loss,test_acc=network.evaluate(test_images,test_labels)
print(test_acc)
#%% 영화 리뷰
from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key)for(key,value)in word_index.items()])
decoded_review=' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])
#%%
print(decoded_review)
#%% 정수 시퀸스를 이진 행렬로 인코딩하기
import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    
    return results
X_train=vectorize_sequences(train_data)
X_test=vectorize_sequences(test_data)
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')
#%% model 정의하기
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#%% model compile
#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# 옵티망저 설정하기
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_cross_entropy',metrics=['accuracy'])
#%% 훈련 검증
# 검증 세트 준비하기
X_val=X_train[:10000]
partial_X_train=X_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
# 모델 훈련하기
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(partial_X_train,partial_y_train,epochs=20,batch_size=512,validation_data=(X_val,y_val))
#%%
history_dict=history.history
history_dict.keys()
#%% 손실그리기
import matplotlib.pyplot as plt

history_dict=history.history
loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Trainingloss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% 정확도 그리기
plt.clf()
acc=history_dict['acc']
val_acc=history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# epoch 4번이후 overfitting 문제 포착
#%% model 재구성
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=4,batch_size=512)
results=model.evaluate(X_test,y_test)

print(results)

#%% 내가 모델 구성 은닉층 1개 추가 은닉층 유닛 추가
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(partial_X_train,partial_y_train,epochs=4,batch_size=512,validation_data=(X_val,y_val))
#%%
import matplotlib.pyplot as plt

plt.clf()
history_dict=history.history
loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Trainingloss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
results=model.evaluate(X_test,y_test)
print(results)
# 거의 차이 없음

#%% 뉴스를 토픽별로 분류하기(다중 분류)
from keras.datasets import reuters
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)
#%% 데이터 인코딩
X_train=vectorize_sequences(train_data)
X_test=vectorize_sequences(test_data)
#%% one hot encoding
def to_one_hot(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1
    return results

one_hot_train_labels=to_one_hot(train_labels)
one_hot_test_labels=to_one_hot(test_labels)

# 내장 함수 사용하기
from keras.utils.np_utils import to_categorical

one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)

#%% 모델 정의하기
from keras import models
from keras import layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

# 모델 컴파일하기
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#%% 검증 세트 준비하기
X_val=X_train[:1000]
y_val=one_hot_train_labels[:1000]

partial_X_train=X_train[1000:]
partial_y_train=one_hot_train_labels[1000:]
#%% 모델 훈련하기
history=model.fit(partial_X_train,partial_y_train,epochs=20,batch_size=512,validation_data=(X_val,y_val))
history_dict=history.history
history_dict.keys()
#%% 손실그리기
import matplotlib.pyplot as plt

history_dict=history.history
loss=history_dict['loss']
val_loss=history_dict['val_loss']
epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'bo',label='Trainingloss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
print(history_dict.keys())
#%% 정확도 그리기
plt.clf()
acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#%%
model.fit(partial_X_train,partial_y_train,epochs=9,batch_size=512,validation_data=(X_val,y_val))
results=model.evaluate(X_test,one_hot_test_labels)
#%% 
print(results)
#%%
import copy
test_labels_copy=copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hists_array=np.array(test_labels)==np.array(test_labels_copy)
print(float(np.sum(hists_array)/len(test_labels)))
#%% 은닉층 유닛증가 은닉층 하나 축소
model=models.Sequential()
model.add(layers.Dense(128,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(46,activation='softmax'))

# 모델 컴파일하기
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#%%
model.fit(partial_X_train,partial_y_train,epochs=9,batch_size=512,validation_data=(X_val,y_val))
results=model.evaluate(X_test,one_hot_test_labels)
print(results)

#%% 보스턴 주택 데이터
from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data()