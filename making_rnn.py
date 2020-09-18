import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss_function(y_target,y,batch_size):
    loss=-np.sum(((y_target*np.log(y))+(1-y_target*np.log(1-y))),axis=1)
    loss/=batch_size
    
    return loss

def softmax(x):
    """ applies softmax to an input x"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class RNN_layer:
    
    def __init__(self,hidden_size,time_steps,input_dim,batch_size):
        self.hidden_size=hidden_size
        self.time_steps=time_steps
        self.input_dim=input_dim
        self.batch_size=batch_size
        
        self.W_h=np.random.uniform(-np.sqrt(1./hidden_size),np.sqrt(1./hidden_size),(hidden_size,hidden_size))
        self.W_x=np.random.uniform(-np.sqrt(1./input_dim),np.sqrt(1./input_dim),(hidden_size,input_dim))
        self.b=np.random.random((hidden_size,1))
        self.h_times=np.random.random((time_steps,hidden_size,batch_size))
        self.W_y=np.random.uniform(-np.sqrt(1./hidden_size),np.sqrt(1./hidden_size),(1,hidden_size))
        self.b_y=np.random.random((1,1))
    
    def forward_RNN(self,x_data):
        # batch_size,time_stpes,input_dim
        '''
        data=np.zeros((self.batch_size,self.input_dim,self.time_steps))
        for i,temp in enumerate(x_data):
            data[i]=temp.T
        data=data.T
        '''
        for i in range(self.time_steps):
            if i==0:
                #batch time input
                self.h_times[i]=np.tanh(np.dot(self.W_x,x_data[i])+self.b)
            else:
                self.h_times[i]=np.tanh(np.dot(self.W_x,x_data[i])+np.dot(self.W_h,self.h_times[i-1])+self.b)
        a=np.dot(self.W_y,self.h_times[self.time_steps-1])+self.b_y
        y=sigmoid(a)
        return y
    
    def backward_RNN(self,y_target,y,x_data):
        '''
        data=np.zeros((self.batch_size,self.input_dim,self.time_steps))
        for i,temp in enumerate(x_data):
            data[i]=temp.T
        data=data.T
        '''
        dW_h=np.zeros_like(self.W_h)
        dW_x=np.zeros_like(self.W_x)
        dW_y=np.zeros_like(self.W_y)
        db=np.zeros_like(self.b)
        db_y=np.zeros_like(self.b_y)
        dh=np.zeros_like(self.h_times)
        da=y-y_target
        
        # y shape=(1,batch_size4)
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

hidden_size=7
time_steps=6
input_dim=12
data_size=1000

x_data=np.random.randint(0,2,size=(time_steps,input_dim,data_size))
#x_data=np.random.randint(0,2,size=(data_size,time_steps,input_dim))
y_train=np.random.randint(0,2,size=(1,data_size))
#%%
import matplotlib.pyplot as plt

r=RNN_layer(hidden_size, time_steps, input_dim,data_size)

epochs=100
loss=np.zeros((epochs,1))
for i in range(epochs):   
    y=r.forward_RNN(x_data)
    y_softmax = softmax(y)
    loss[i]=loss_function(y_train, y_softmax, data_size)
    print(i)
    r.update_model(0.0001, y_train, y, x_data)  # learning rate 조절

plt.plot(range(1,epochs+1),loss)
plt.show()
#%%
a=[[[1,2,3]],[[7,8,9]]]
a=np.array(a)
print(a.shape)
print(a.T.shape)   

    