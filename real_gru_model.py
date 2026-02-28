# -*- coding: utf-8 -*-
"""
real_gru_model.py
真正的GRU神經網路模型（推論用）

可直接與 gru_model.py 整合
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
except:
    torch=None


# ==========================
# 真正GRU網路
# ==========================

class BaccaratGRUNet(nn.Module):

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


# ==========================
# 推論器
# ==========================

class GRUPredictor:

    def __init__(self,model_path="gru_model.pt"):

        self.model=None

        if torch is None:

            print("Torch not installed → GRU disabled")

            return


        try:

            self.device="cpu"

            self.model=BaccaratGRUNet().to(self.device)

            self.model.load_state_dict(
                torch.load(model_path,map_location=self.device)
            )

            self.model.eval()

            print("GRU model loaded")

        except:

            print("GRU model not found → using neutral output")

            self.model=None


    # =======================
    # 預測
    # =======================

    def predict(self,history):

        # 沒模型
        if self.model is None:

            return np.array([0.458,0.446,0.096])


        # 歷史太短
        if len(history)<10:

            return np.array([0.458,0.446,0.096])


        try:

            seq=np.array(history[-40:])

            length=len(seq)

            onehot=np.zeros((length,3))

            for i,v in enumerate(seq):

                onehot[i,v]=1


            x=torch.tensor(
                onehot,
                dtype=torch.float32
            )

            x=x.unsqueeze(0)


            with torch.no_grad():

                y=self.model(x).cpu().numpy()[0]


            return y


        except:

            return np.array([0.458,0.446,0.096])
