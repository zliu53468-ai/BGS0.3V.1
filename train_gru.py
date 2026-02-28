# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

########################################
# 真正 GRU 模型
########################################

class GRUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=48,   # 原32 → 提升記憶能力
            num_layers=2,
            batch_first=True
        )

        self.fc1 = nn.Linear(48,24)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(24,3)



    def forward(self,x):

        out,_ = self.gru(x)

        out = out[:,-1,:]

        out = self.fc1(out)

        out = self.relu(out)

        out = self.fc2(out)

        return torch.softmax(out,dim=1)



########################################
# 模擬百家樂序列資料
########################################

def generate_baccarat_sequence(length=40):

    seq=[]

    last=np.random.randint(0,2)

    for i in range(length):

        r=np.random.rand()

        # Tie 約9%
        if r<0.09:

            seq.append(2)

            continue

        # 連莊機率
        if np.random.rand()<0.62:

            seq.append(last)

        else:

            last=1-last

            seq.append(last)


    return np.array(seq)



########################################
# 產生訓練資料
########################################

def generate_data(n=8000):

    X=[]
    Y=[]

    for _ in range(n):

        seq=generate_baccarat_sequence(41)

        X.append(seq[:-1])

        Y.append(seq[-1])


    return np.array(X),np.array(Y)



########################################
# 建立模型
########################################

model=GRUNet()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn=nn.CrossEntropyLoss()



########################################
# 建立資料
########################################

print("Generating data...")

X,Y=generate_data()

X_onehot=np.eye(3)[X]

X_tensor=torch.tensor(X_onehot).float()

Y_tensor=torch.tensor(Y)



########################################
# 訓練
########################################

print("Training...")

for epoch in range(25):   # 原10 → 25

    pred=model(X_tensor)

    loss=loss_fn(pred,Y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("epoch",epoch,"loss",loss.item())



########################################
# 儲存模型
########################################

torch.save(model.state_dict(),"gru_model.pt")

print("Saved gru_model.pt")
