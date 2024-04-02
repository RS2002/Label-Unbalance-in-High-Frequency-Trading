from torch.utils.data import Dataset
import numpy as np
import pickle

# class My_dataset(Dataset):
#     def __init__(self, features, label):
#         super().__init__()
#         self.features = features
#         self.label = label
#
#     def __len__(self):
#         return self.label.shape[0]
#
#     def __getitem__(self, index):
#         return self.features[index], self.label[index]
#
# def load_data(path="/disk1/imb/202305_all/data.pkl"):
#     features, label=None, []
#
#     with open(path, 'rb') as f:
#         data_pkl = pickle.load(f)
#         for data in data_pkl:
#             if features is None:
#                 f=data['features']
#                 features = f.reshape(-1,f.shape[0],f.shape[1])
#                 label.append(data['label'])
#             else:
#                 f=data['features']
#                 f = f.reshape(-1,f.shape[0],f.shape[1])
#                 features = np.concatenate([features,f],axis=0)
#                 label.append(data['label'])
#
#     label=np.array(label)
#     return features, label
#
# # test
# if __name__ == '__main__':
#     features, label = load_data()
#     print("Features: ",features.shape)
#     print("Label: ",label.shape)
#     dataset=My_dataset(features,label)
#     print(dataset)

class My_dataset(Dataset):
    def __init__(self, data_pkl):
        super().__init__()
        self.dataset = data_pkl

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]['features'], self.dataset[index]['label']

# test
if __name__ == '__main__':
    with open("/disk1/imb/202305_all/data.pkl", 'rb') as f:
        data_pkl = pickle.load(f)
    dataset=My_dataset(data_pkl)
    print(len(dataset))