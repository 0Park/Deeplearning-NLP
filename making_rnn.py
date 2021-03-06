import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_function(y_target,y,batch_size):
    loss=-np.sum(((y_target*np.log(y))+((1-y_target)*np.log(1-y))),axis=1)
    loss/=batch_size
    
    return loss

class RNN_layer:
    
    def __init__(self,hidden_size,time_steps,input_dim,batch_size):
        self.hidden_size=hidden_size
        self.time_steps=time_steps
        self.input_dim=input_dim
        self.batch_size=batch_size
        
        self.W_h=np.random.uniform(-np.sqrt(6./hidden_size),np.sqrt(6./hidden_size),(hidden_size,hidden_size))
        self.W_x=np.random.uniform(-np.sqrt(6./(input_dim+hidden_size)),np.sqrt(6./(input_dim+hidden_size)),(hidden_size,input_dim))
        self.b=np.zeros((hidden_size,1))
        self.h_times=np.zeros((time_steps,hidden_size,batch_size))
        self.W_y=np.random.uniform(-np.sqrt(6./hidden_size),np.sqrt(6./hidden_size),(1,hidden_size))
        self.b_y=np.zeros((1,1))
    
    def forward_RNN(self,x_data):
        for i in range(self.time_steps):
            if i==0:
                self.h_times[i]=np.tanh(np.dot(self.W_x,x_data[i])+self.b)
            else:
                self.h_times[i]=np.tanh(np.dot(self.W_x,x_data[i])+np.dot(self.W_h,self.h_times[i-1])+self.b)
        a=np.dot(self.W_y,self.h_times[self.time_steps-1])+self.b_y
        y=sigmoid(a)
        return y
    
    def backward_RNN(self,y_target,y,x_data):
        dW_h=np.zeros_like(self.W_h)
        dW_x=np.zeros_like(self.W_x)
        dW_y=np.zeros_like(self.W_y)
        db=np.zeros_like(self.b)
        db_y=np.zeros_like(self.b_y)
        dh=np.zeros_like(self.h_times)
        da=y-y_target
        
        n=self.time_steps
        for i in range(1,n+1):
            if i==1:
                dh[n-i]=np.dot(self.W_y.T,da)
            else:
                dh[n-i]=np.dot(self.W_h.T,(1-np.square(self.h_times[n-i+1]))*dh[n-i+1])
        
        for i in range(n):
            db_y+=np.sum(da,axis=1).T/self.batch_size
            a=np.sum((1-np.square(self.h_times[i]))*dh[i],axis=1)/self.batch_size
            db+=(db.T+a).T
            dW_y+=np.dot(da,self.h_times[i].T)
            if i!=0:           
                dW_h+=np.dot((1-np.square(self.h_times[i]))*dh[i],self.h_times[i-1].T)
            dW_x+=np.dot((1-np.square(self.h_times[i]))*dh[i],x_data[i].T)
            
        return dW_h,dW_x,dW_y,db,db_y
    
    def update_model(self,learning_rate,y_target,y,x_data):
        derivatives=self.backward_RNN(y_target,y,x_data)
        self.W_h=self.W_h-learning_rate*derivatives[0]
        self.W_x=self.W_x-learning_rate*derivatives[1]
        self.W_y=self.W_y-learning_rate*derivatives[2]
        self.b=self.b-learning_rate*derivatives[3]
        self.b_y=self.b_y-learning_rate*derivatives[4]
        
#%%

hidden_size=4
time_steps=6
input_dim=12
data_size=10000

x_data=np.random.randint(0,2,size=(time_steps,input_dim,data_size))
y_train=np.random.randint(0,2,size=(1,data_size))

x_data_test=np.random.randint(0,2,size=(time_steps,input_dim,data_size))
y_test=np.random.randint(0,2,size=(1,data_size))

#%%
import matplotlib.pyplot as plt

r=RNN_layer(hidden_size, time_steps, input_dim,data_size)

n_epochs=30
loss=np.zeros((n_epochs,1))
for i in range(n_epochs):   
    y=r.forward_RNN(x_data)
    loss[i]=loss_function(y_train, y, data_size)
    print(i)
    r.update_model(0.0001, y_train, y, x_data)  # learning rate 조절

plt.plot(range(1,n_epochs+1),loss)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()
#%%

y=r.forward_RNN(x_data_test)
loss_test=loss_function(y_test,y,data_size)

for i,result in enumerate(y.T):
    if result>=0.1:
        y[0][i]=1
    else:
        y[0][i]=0
acc=(1-(np.sum(abs(y-y_test),axis=1)/data_size))*100

print("test loss, test acc:",loss_test,acc)
#%%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
x_data_tf=np.random.randint(0,2,size=(data_size,time_steps,input_dim))
y_train_tf=np.random.randint(0,2,size=(data_size,))

model=Sequential()
model.add(SimpleRNN(hidden_size,input_shape=(time_steps,input_dim)))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_data_tf,y_train_tf,epochs=n_epochs,verbose=1)
#%%
x_data_tf_test=np.random.randint(0,2,size=(data_size,time_steps,input_dim))
y_test_tf=np.random.randint(0,2,size=(data_size,))
result=model.evaluate(x_data_tf_test,y_test_tf)
print("test loss, test acc:",result)
