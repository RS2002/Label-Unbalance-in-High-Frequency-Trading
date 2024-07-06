import pickle
import numpy as np

features, timestamp, item_id, label=None, None, [], []

path="/disk1/imb/202305_all/data.pkl"
with open(path, 'rb') as f:
    data_pkl = pickle.load(f)
    for data in data_pkl:
        if features is None:
            f=data['features']
            features = f.reshape(-1,f.shape[0],f.shape[1])
            t=data['time']
            timestamp = t.reshape(-1,t.shape[0])
            label.append(data['label'])
            item_id.append(data['item_id'])
        else:
            f=data['features']
            f = f.reshape(-1,f.shape[0],f.shape[1])
            features = np.concatenate([features,f],axis=0)
            t=data['time']
            t = t.reshape(-1,t.shape[0])
            timestamp = np.concatenate([timestamp,t],axis=0)
            label.append(data['label'])
            item_id.append(data['item_id'])

label=np.array(label)
item_id=np.array(item_id)
print("Features: ", features.shape)
print("Label: ", label.shape)
print("Timestamp: ", timestamp.shape)
print("Item_id: ", item_id.shape)
np.save("/disk1/imb/202305_all/features.npy", np.array(features))
np.save("/disk1/imb/202305_all/label.npy", np.array(label))
np.save("/disk1/imb/202305_all/timestamp.npy", np.array(timestamp))
np.save("/disk1/imb/202305_all/item_id.npy", np.array(item_id))