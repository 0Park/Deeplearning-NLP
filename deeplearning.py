#%% function
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(f,var):
    if var.ndim==1:  # vectpr
        temp_var=var # 원본 값 저장
        delta=1e-5
        diff_val=np.zeros(var.shape) #미분 계수 보관 변수
        
        for index in range(len(var)):
            target_var=float(temp_var[index])
            temp_var[index]=target_var+delta
            func_val_plust_delta=f(temp_var)
            
            temp_var[index]=target_var-delta
            func_val_minus_delta=f(temp_var) #x+delta에 대한 함수 값 계산
            
            diff_val[index]=(func_val_plust_delta-func_val_minus_delta)/(2*delta)
            # x-delta에 대한 함수 값 계산
            temp_var[index]=target_var
            
        return diff_val
    
    elif var.ndim ==2:
        temp_var=var
        delta=1e-5
        diff_val=np.zeros(var.shape)
        
        rows=var.shape[0]
        columns=var.shape[1]
        
        for row in range(rows):
            for column in range(columns):
                target_var=float(temp_var[row,column])
                
                temp_var[row,column]=target_var+delta
                func_val_plus_delta=f(temp_var)
                
                temp_var[row,column]=target_var-delta
                func_val_minus_delta=f(temp_var)
                
                diff_val[row,column]=(func_val_plus_delta-func_val_minus_delta)/(2*delta)
                temp_var[row,column]=target_var
                
                
        return diff_val

class LogicGate:
    def __init__(self,gate_name,xdata,tdata):
        self.name=gate_name
        self.xdata=xdata.reshape(4,2)
        self.tdata=tdata.reshape(4,1)
        
        #입력층 노드 2개, 은닉층 노드 6개
        self.W2=np.random.rand(2,6)
        self.b2=np.random.rand(6)
        
        #은닉층 노드 6개, 출력층 노드 1개
        self.W3=np.random.rand(6,1)
        self.b3=np.random.rand(1)
        # 학습률 초기화
        self.learning_rate=1e-2
        
    def feed_forward(self):
        delta=1e-7 # log 무한대 발산 방지
        z2=np.dot(self.xdata, self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        return -np.sum(self.tdata*np.log(y+delta)+(1-self.tdata)*np.log((1-y)+delta))
        # 오차 함수
        
    def loss_val(self):
        delta=1e-7 # log 무한대 발산 방지
        z2=np.dot(self.xdata, self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        return -np.sum(self.tdata*np.log(y+delta)+(1-self.tdata)*np.log((1-y)+delta))
        # 오차 함수
        
    def train(self):
        f= lambda x: self.feed_forward()
        print("Initial loss value=",self.loss_val())
        
        for step in range(20001):
            self.W2 -=self.learning_rate*derivative(f,self.W2)
            self.b2 -=self.learning_rate*derivative(f,self.b2)
            
            self.W3-=self.learning_rate*derivative(f,self.W3)
            self.b3-=self.learning_rate*derivative(f,self.b3)
            
            if (step % 1000==0):
                print("step =",step,"loss value=",self.loss_val())
                
    def predict(self,input_data):
        z2=np.dot(input_data,self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        
        if y > 0.5:
            result=1
        else:
            result=0

        return y,result
    
#%% main

xdata=np.array([[0,0],[0,1],[1,0],[1,1]])
tdata=np.array([0,1,1,0])
xor_obj=LogicGate("XOR",xdata,tdata)
xor_obj.train()


#%% test
test_data=np.array([[0,0],[0,1],[1,0],[1,1]])
for data in test_data:
    (sigmoid_val,logical_val)=xor_obj.predict(data)
    
    print(data,"=",logical_val)
