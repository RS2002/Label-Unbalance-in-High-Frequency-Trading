from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self,feature,gt,label,window=60):
        self.feature = feature
        self.gt = gt
        self.label = label
        self.window = window
        self.num = len(self.label) - self.window

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index += self.window
        while np.isnan(self.label[index]) or np.isnan(self.gt[index]):
            index = (index + 1) % self.num
        return self.feature[index-self.window+1:index+1], self.gt[index], self.label[index]


def load_data(data_path="/disk1/imb/202305_all/data_all_au2308_factors_and_label.csv",train_prop=0.8,valid_prop=None,window=60):
    df = pd.read_csv(data_path, na_values=np.nan)
    features = df[['mid_price', 'diff_last_price', 'diff_bid_price1', 'diff_bid_price2', 'diff_bid_price3', 'diff_bid_price4', 'diff_bid_price5', 'diff_ask_price1', 'diff_ask_price2', 'diff_ask_price3', 'diff_ask_price4', 'diff_ask_price5', 'log_volume']]
    gt = df['return']
    label = df['label']
    features = np.array(features)
    gt = np.array(gt)
    label = np.array(label)

    data_num = len(label)
    train_num = int(train_prop*data_num)
    if valid_prop is None:
        return MyDataset(features[:train_num],gt[:train_num],label[:train_num],window), MyDataset(features[train_num:],gt[train_num:],label[train_num:],window)
    else:
        valid_num = int(valid_prop*data_num)
        return MyDataset(features[:train_num],gt[:train_num],label[:train_num],window), MyDataset(features[train_num:train_num+valid_num],gt[train_num:train_num+valid_num],label[train_num:train_num+valid_num],window), MyDataset(features[train_num+valid_num:],gt[train_num+valid_num:],label[train_num+valid_num:],window)

