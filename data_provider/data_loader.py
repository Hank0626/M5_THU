import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_M5(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info

        self.seq_len = 336
        self.label_len = 0
        self.pred_len = 28

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))


        cols = list(df_raw.columns)
        cols = [c for c in cols if "d_" in c]
        df_data = df_raw[cols]
        
        border1s = [0, 1800 - self.seq_len]
        border2s = [1800, 1941]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data = data.T
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.total_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.totel_item = self.data_x.shape[-1]
        print(self.data_x.shape)

    def __getitem__(self, index):
        feat_id = index // self.total_len
        s_begin = index % self.total_len

        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]

        return seq_x, seq_y

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.totel_item

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)