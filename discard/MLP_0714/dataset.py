from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import pandas as pd


class MyDataset(Dataset):
    def __init__(self, feature, gt, label, window=60):
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
        return self.feature[index - self.window + 1:index + 1], self.gt[index], self.label[index]


def test():
    print('test: work well!')


def standardize_array(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data


def load_data(data_path="/disk1/imb/202305_all/data_all_au2308_factors_and_label.csv", train_prop=0.8, window=60,
              normalization=False):
    df = pd.read_csv(data_path, na_values=np.nan)
    features = df[
        ['mid_price', 'diff_last_price', 'diff_bid_price1', 'diff_bid_price2', 'diff_bid_price3', 'diff_bid_price4',
         'diff_bid_price5', 'diff_ask_price1', 'diff_ask_price2', 'diff_ask_price3', 'diff_ask_price4',
         'diff_ask_price5', 'log_volume']]
    gt = df['return']
    label = df['label']
    features = np.array(features)
    gt = np.array(gt)
    label = np.array(label)

    data_num = len(label)  # note that above: self.num = len(self.label) - self.window
    train_num = int(train_prop * data_num)

    if normalization:
        return (MyDataset(standardize_array(features[:train_num]), gt[:train_num], label[:train_num]),
                MyDataset(standardize_array(features[train_num:]), gt[train_num:], label[train_num:]))
    else:
        return (MyDataset(features[:train_num], gt[:train_num], label[:train_num]),
                MyDataset(features[train_num:], gt[train_num:], label[train_num:]))


def load_data_rolling(
        data_path="/disk1/imb/202305_all/data_all_au2308_factors_and_label.csv", train_prop=0.8, cross_prop=0.2,
        rolling_num=6, normalization=False):
    df = pd.read_csv(data_path, na_values=np.nan)
    features = df[
        ['mid_price', 'diff_last_price', 'diff_bid_price1', 'diff_bid_price2', 'diff_bid_price3', 'diff_bid_price4',
         'diff_bid_price5', 'diff_ask_price1', 'diff_ask_price2', 'diff_ask_price3', 'diff_ask_price4',
         'diff_ask_price5', 'log_volume']]
    gt = df['return']
    label = df['label']
    features = np.array(features)
    gt = np.array(gt)
    label = np.array(label)

    total_sum = 1 + (1 - cross_prop) * (rolling_num - 1)
    train_start_num = [(1 - cross_prop) * N / total_sum for N in range(0, rolling_num)]
    train_end_num = [(train_prop + (1 - cross_prop) * N) / total_sum for N in range(0, rolling_num)]
    test_start_num = train_end_num.copy()
    test_end_num = [(1 + (1 - cross_prop) * N) / total_sum for N in range(0, rolling_num)]
    # print(test_end_num)

    train_set_dict = {i: None for i in range(rolling_num)}
    test_set_dict = {i: None for i in range(rolling_num)}
    # note that above: self.num = len(self.label) - self.window, thus we need to substract 60 below
    # or we may use xxx[xxx:] in the end
    # (think about this process)
    data_num = len(label)

    if normalization:
        dataset_all = MyDataset(standardize_array(features), gt, label)
    else:
        dataset_all = MyDataset(features, gt, label)

    for i in range(rolling_num):
        train_index_list = list(range(int(train_start_num[i] * data_num), int(train_end_num[i] * data_num)))
        test_index_list = list(range(int(test_start_num[i] * data_num), int(test_end_num[i] * data_num)-60))
        train_set_dict[i] = Subset(dataset_all, train_index_list)
        test_set_dict[i] =Subset(dataset_all, test_index_list)

    test_set_all = ConcatDataset(test_set_dict.values())
    return train_set_dict, test_set_dict, test_set_all
