import numpy as np
import pickle
import os
import pandas as pd

root="/disk1/imb/202305_all"
data=[]
limit=0
if limit!=0:
    pattern="limit{:}".format(limit)
else:
    pattern=None
length=60 # 数据长度
swing_length=1 # 滑动窗口间隔

item_id=0
for file_name in os.listdir(root):
    if (pattern is None and "limit" in file_name) or (pattern is not None and pattern not in file_name):
        continue
    item = file_name.split("_")[-1][:-4]
    print(item_id,item)
    path=os.path.join(root,file_name)
    df = pd.read_csv(path, na_values=np.nan)
    timestamp = df['TimeStamp']
    features = df[['mid_price', 'diff_last_price', 'diff_bid_price1', 'diff_bid_price2', 'diff_bid_price3', 'diff_bid_price4', 'diff_bid_price5', 'diff_ask_price1', 'diff_ask_price2', 'diff_ask_price3', 'diff_ask_price4', 'diff_ask_price5', 'log_volume']]
    timestamp = np.array(timestamp)
    features = np.array(features)
    label = df['return']
    label=np.array(label)
    i=0
    while i<label.shape[0]:
        #TODO: 不知道limit！=0时数据如何使用!
        if not np.isnan(label[i]):
            data.append({
                'time': timestamp[i-length+1:i+1],
                'features': features[i-length+1:i+1],
                'item_id': item_id,
                'item': item,
                'label': label[i]
            })
        i += swing_length
    item_id+=1

if pattern is not None:
    output_file = 'data_{:}.pkl'.format(limit)
else:
    output_file = "data.pkl"
output_file=os.path.join(root,output_file)
with open(output_file, 'wb') as f:
    pickle.dump(data, f)