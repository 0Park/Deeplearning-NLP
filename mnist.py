#%% mnist 가져오기
import numpy as np

training_data=np.loadtxt('F:/mnist_train.csv',delimiter=',',dtype=np.float32)
test_data=np.loadtxt('F:/mnist_test.csv',delimiter=',',dtype=np.float32)

print('training_data.shape=',training_data.shape,'test_data.shape=',test_data.shape)

#%%  mnist 이미지 표현
import matplotlib.pyplot as plt

image=training_data[0,1:].reshape(28,28)

plt.imshow(image,cmap='gray')
plt.show()

#%% minist_test 클래스
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

class MNIST_Test:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        # 가중치 W2,W3 Xavier/He 방법으로 초기화. 성능과 정확도 높일 수 있어 실무에 많이 사용
        self.W2=np.random.randn(self.input_nodes,self.hidden_nodes)/np.sqrt(self.input_nodes/2)
        self.b2=np.random.rand(self.hidden_nodes)
        self.W3=np.random.randn(self.hidden_nodes,self.output_nodes)/np.sqrt(self.hidden_nodes/2)
        self.b3=np.random.rand(self.output_nodes)
        
        self.learning_rate=learning_rate
        
    def feed_forward(self):
        delta=1e-7
        z2=np.dot(self.input_data,self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        
        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))
    
    def loss_val(self):
        delta=1e-7
        z2=np.dot(self.input_data,self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        
        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))
    
    def train(self,input_data,target_data):
        self.input_data=input_data
        self.target_data=target_data
        
        f= lambda x: self.feed_forward()
        
        self.W2-=self.learning_rate*derivative(f, self.W2)
        self.b2-=self.learning_rate*derivative(f, self.b2)
        self.W3-=self.learning_rate*derivative(f, self.W3)
        self.b3-=self.learning_rate*derivative(f, self.b3)
        
    def predict(self,input_data):
        z2=np.dot(input_data,self.W2)+self.b2
        a2=sigmoid(z2)
        z3=np.dot(a2,self.W3)+self.b3
        y=a3=sigmoid(z3)
        
        # MNIST 경우는 One-Hot Encoding을 적용하기 때문에
        #0 또는 1이 아닌 argmax()를 통해 최대 인덱스를 넘겨준다.
        
        predicted_num=np.argmax(y)
        return predicted_num
    
    def accuracy(self,input_data,target_data):
        matched_list=[]
        not_matched_list=[]
        
        for index in range(len(input_data)):
            label=int(target_data[index])
            #오버 플로우 나지 않게 정규화
            data=(input_data[index,:]/255.0*0.99)+0.01
            predicted_num=self.predict(data)
            
            if label==predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
                
        print("Current Accuracy",len(matched_list)/len(not_matched_list))
        
        return matched_list,not_matched_list


#%% MNIST_Test 객체 생성 및 정확도 검증
i_nodes=training_data.shape[1]-1
h1_nodes=30
o_nodes=10
lr=1e-2
epochs=1

obj=MNIST_Test(i_nodes,h1_nodes,o_nodes,lr)

for step in range(epochs):
    for index in range(len(training_data)):
        
        input_data=((training_data[index,1:]/255.0)*0.99)+0.01
        target_data=np.zeros(o_nodes)+0.01
        target_data[int(training_data[index,0])]=0.99
        
        obj.train(input_data,target_data)
        
        if(index % 200 == 0):
            print("epochs= ",step,", index = ",index,", loss value =",obj.loss_val())
        
        

#%% NeuralNetwork 클래스
import numpy as np
class NeuralNetwork:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes=input_nodes
        self.hidden_nodes=hidden_nodes
        self.output_nodes=output_nodes
        self.learning_rate=learning_rate
        # Xavier/He 방법으로 초기화
        self.W2=np.random.randn(self.input_nodes,self.hidden_nodes)/np.sqrt(self.input_nodes/2)
        self.b2=np.random.rand(self.hidden_nodes)
        
        self.W3=np.random.randn(self.hidden_nodes,self.output_nodes)/np.sqrt(self.hidden_nodes/2)
        self.b3=np.random.rand(self.output_nodes)
        
        self.Z3=np.zeros([1,output_nodes])
        self.A3=np.zeros([1,output_nodes])
        
        self.Z2=np.zeros([1,hidden_nodes])
        self.A2=np.zeros([1,hidden_nodes])
        
        self.Z1=np.zeros([1,input_nodes])
        self.A1=np.zeros([1,input_nodes])
        
    def feed_forward(self):
        delta=1e-7
        
        self.Z1=self.input_data
        self.A1=self.input_data
        
        self.Z2=np.dot(self.A1,self.W2)+self.b2
        self.A2=sigmoid(self.Z2)
        
        self.Z3=np.dot(self.A2,self.W3)+self.b3
        y=self.A3=sigmoid(self.Z3)

        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))

    def loss_val(self):
        delta=1e-7
        
        self.Z1=self.input_data
        self.A1=self.input_data
        
        self.Z2=np.dot(self.A1,self.W2)+self.b2
        self.A2=sigmoid(self.Z2)
        
        self.Z3=np.dot(self.A2,self.W3)+self.b3
        y=self.A3=sigmoid(self.Z3)

        return -np.sum(self.target_data*np.log(y+delta)+(1-self.target_data)*np.log((1-y)+delta))
        
        
    def train(self,input_data,target_data):        
        self.input_data=input_data
        self.target_data=target_data
        
        self.feed_forward()
        #출력층 loss인 loss_3구함
        loss_3=(self.A3-self.target_data)*self.A3*(1-self.A3)
        
        #출력층 가중치 W3, 출력층 바이어스 b3 업데이트
        self.W3=self.W3-self.learning_rate*np.dot(self.A2.T,loss_3)
        self.b3=self.b3-self.learning_rate*loss_3
        #은닉층 loss인 loss_2 구함
        loss_2=np.dot(loss_3,self.W3.T)*self.A2*(1-self.A2)
        self.W2=self.W2-self.learning_rate*np.dot(self.A1.T,loss_2)
        self.b2=self.b2-self.learning_rate*loss_2
        
    def predict(self,input_data):
        Z2=np.dot(input_data,self.W2)+self.b2
        A2=sigmoid(Z2)
        Z3=np.dot(A2,self.W3)+self.b3
        y=A3=sigmoid(Z3)
        
        predicted_num=np.argmax(y)
        
        return predicted_num
    
    def accuracy(self,test_input_data,test_target_data):
        matched_list=[]
        not_matched_list=[]
        
        for index in range(len(test_input_data)):
            label=int(test_target_data[index])
            
            #정규화
            data=(test_input_data[index,:]/255.0*0.99)+0.01
            #predict를 위해서 vector를 matrix로 변환하여 인수로 넘겨줌
            
            predicted_num=self.predict(np.array(data,ndmin=2))
            if label==predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
        
        print("Current Accuracy",len(matched_list)/len(test_input_data))
        
        return matched_list,not_matched_list
#%% NeuralNetwork 객체 생성 밎 정확도 검증
i_nodes=training_data.shape[1]-1
h1_nodes=100
o_nodes=10
lr=0.3
epochs=1

obj=NeuralNetwork(i_nodes,h1_nodes,o_nodes,lr)

for step in range(epochs):
    for index in range(len(training_data)):
        
        input_data=((training_data[index,1:]/255.0)*0.99)+0.01
        target_data=np.zeros(o_nodes)+0.01
        target_data[int(training_data[index,0])]=0.99
        
        obj.train(np.array(input_data,ndmin=2),target_data)
        
        if(index % 1000 == 0):
            print("epochs= ",step,", index = ",index,", loss value =",obj.loss_val())    

#%% test data
test_input_data=test_data[:,1:]
test_target_data=test_data[:,0]
(true_list,false_list)=obj.accuracy(test_input_data, test_target_data)
