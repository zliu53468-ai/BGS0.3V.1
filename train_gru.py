# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

# 真正GRU模型
class GRUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(32,3)

    def forward(self,x):

        out,_ = self.gru(x)

        out = out[:,-1,:]

        out = self.fc(out)

        return torch.softmax(out,dim=1)


model = GRUNet()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

loss_fn = nn.CrossEntropyLoss()


# 隨機訓練資料（先讓模型可用）
def generate_data(n=3000):

    X=[]
    Y=[]

    for _ in range(n):

        seq=np.random.randint(0,3,40)

        target=seq[-1]

        X.append(seq[:-1])
        Y.append(target)

    return np.array(X),np.array(Y)


X,Y=generate_data()

X_onehot=np.eye(3)[X]

X_tensor=torch.tensor(X_onehot).float()
Y_tensor=torch.tensor(Y)


for epoch in range(10):

    pred=model(X_tensor)

    loss=loss_fn(pred,Y_tensor)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("epoch",epoch,"loss",loss.item())


torch.save(model.state_dict(),"gru_model.pt")

print("Saved gru_model.pt")
