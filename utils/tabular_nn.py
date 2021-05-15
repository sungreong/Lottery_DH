import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from torch import nn 
from torch.utils.data import Dataset, DataLoader

from copy import deepcopy
"""
tabular nn

continuous , category split 

# TODO

* Category 개수에 따라서 나눠서 처리하기
* 2개인 경우
    * 하나의 컬럼으로 사용
* 특정 n개 이하인 경우
    * one hot으로 사용
* 나머지
    * embedding 으로 처리 
    * min(50, (n+1)//2)

* 현재(210515)
    * min(50, (n+1)//2)

"""

def collect_cat_info(df_input, fac_col ) :
    categorical_columns = []
    categorical_dims =  {}
    categorical_positions =  {}
    categorical_idxs =  []
    encodding_scaler = {}
    for idx , col in enumerate(df_input.columns.tolist()) :
        if col in fac_col :
            l_enc = LabelEncoder()
            df_input[col] = df_input[col].fillna("VV_likely")
            df_input[col] = l_enc.fit_transform(df_input[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = (len(l_enc.classes_), min(50,(len(l_enc.classes_)+1)//2))
            categorical_positions[idx] = (len(l_enc.classes_), min(50,(len(l_enc.classes_)+1)//2))
            categorical_idxs.append(idx)
            encodding_scaler[col] = deepcopy(l_enc)
    return categorical_columns , categorical_idxs, categorical_dims , categorical_positions , encodding_scaler


class TabularNN(nn.Module) :
    def __init__(self, input_len,emb_dims_dict,output_len,lin_layer_sizes,emb_dropout_ratio) :
        super(TabularNN,self).__init__()
        self.emb_layers = nn.ModuleDict([[str(idx) , nn.Embedding(x, y)] for idx, (x, y) in emb_dims_dict.items()])
        no_of_embs = sum([y for idx, (x, y) in emb_dims_dict.items()])
        no_of_fac = len([idx for idx, (x, y) in emb_dims_dict.items()])
        no_of_cont = input_len - no_of_fac
        first_lin_layer = nn.Linear(no_of_embs +no_of_cont, lin_layer_sizes[0])
        self.emb_dropout_layer = nn.Dropout(emb_dropout_ratio)
        lin_layers =[first_lin_layer] + \
            [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1]) for i in range(len(lin_layer_sizes) - 1)]
        for lin_layer in lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)
        lin_acts = [nn.SELU() for _ in range(len(lin_layer_sizes))]
        bn_layers = [nn.BatchNorm1d(size) for size in lin_layer_sizes]
        inter_seq_layers = []
        for l , b , a in zip(lin_layers , bn_layers , lin_acts) :
            inter_seq_layers.append(l)
            inter_seq_layers.append(b)
            inter_seq_layers.append(a)
        self.inter_layers = nn.Sequential(*inter_seq_layers)
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_len)

    def forward(self, con_tensor , fac_tensor) :
        emb_tensor = torch.cat([emb_layer(fac_tensor[:,idx]) for idx, (col_idx , emb_layer) in enumerate(self.emb_layers.items())],axis=1)
        emb_tensor = self.emb_dropout_layer(emb_tensor)
        in_tensor = torch.cat([con_tensor,emb_tensor],axis=1)
        inter_tensor = self.inter_layers(in_tensor)
        output_tensor = self.output_layer(inter_tensor)
        return output_tensor

from torch.utils.data import Dataset, DataLoader


class TabularDataset(Dataset):
    def __init__(self, X,Y,fac_cols) :
        self.num_cols = []
        for col in X.columns.tolist() :
            if col not in fac_cols :
                self.num_cols.append(col)
        self.fac_cols = fac_cols
        self.n = X.shape[0]
        if self.num_cols :
            self.cont_x = X[self.num_cols].astype(np.float32).values     
        else :
            self.cont_X = np.zeros((self.n, 1))
        if self.fac_cols :
            self.fac_x = X[fac_cols].astype(np.int32).values
        else :
            self.fac_x = np.zeros((self.n, 1))
        if Y is None :
            self.y = np.zeros((self.n, 1))
        else : 
            self.y = Y.values.astype(np.int32)

    def __len__(self) :
        return self.n

    def __getitem__(self,idx) :
        return [self.cont_x[idx], self.fac_x[idx] , self.y[idx]]