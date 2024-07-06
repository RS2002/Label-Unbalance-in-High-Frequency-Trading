from dataset import My_dataset
from model import My_Mamba,RNN,LSTM,My_BERT,TCN,MLP
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
import tqdm
from torch.utils.data import DataLoader
import pickle
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# scale=10000.0
scale=1.0


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=13)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--state_size', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=16)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument("--data_path",type=str,default="/disk1/imb/202305_all")
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--model",type=str,default="mamba")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--norm", action="store_true",default=False)
    # parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--test_prop', type=float, default=0.2)
    args = parser.parse_args()
    return args

def iteration(model,classifier,predictor,data_loader,optim,loss_func,device,train=True):
    if train:
        model.train()
        classifier.train()
        predictor.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        classifier.eval()
        predictor.eval()
        torch.set_grad_enabled(False)

    loss_list = []
    mse_list = []
    mape_list = []
    result = None
    gt = None
    acc_list = []
    loss_cls = nn.CrossEntropyLoss()
    relu = nn.ReLU()

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, label in pbar:
        x=x.float().to(device)
        label=label.float().to(device)
        # label*=scale

        sign = torch.sign(label)
        sign[sign>=0]=1
        sign[sign<0]=0
        # sign = sign + 1
        value = torch.abs(label)
        value *= scale

        x_emb=model(x)
        value_hat = torch.abs(predictor(x_emb))
        # value_hat = relu(predictor(x_emb))
        value_hat = torch.squeeze(value_hat,dim=-1)
        sign_hat = classifier(x_emb)
        loss_value = loss_func(value_hat,value)
        loss_sign = loss_cls(sign_hat,sign.long())
        loss = loss_value + loss_sign
        # loss = loss_sign
        sign_hat = torch.argmax(sign_hat,dim=-1)
        acc = torch.mean((sign==sign_hat).float())
        sign_hat[sign_hat==0]=-1
        # sign_hat = sign_hat - 1

        # sign_hat=torch.unsqueeze(sign_hat,dim=-1)
        y = sign_hat * value_hat

        # print(y.shape)

        y /= scale

        # y = torch.squeeze(y,dim=-1)

        mse = loss_func(y, label)
        mape = torch.mean(torch.abs(y-label)/(torch.abs(label)+1e-8))

        loss_list.append(loss.item())
        mape_list.append(mape.item())
        acc_list.append(acc.item())
        mse_list.append(mse.item())

        if train:
            model.zero_grad()
            classifier.zero_grad()
            predictor.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
        else:
            # y /= scale
            # label /= scale
            if result is None:
                result=y
                gt=label
            else:
                result=torch.cat([result,y],dim=0)
                gt=torch.cat([gt,label],dim=0)

    return np.mean(loss_list),np.mean(mape_list),np.mean(mse_list),np.mean(acc_list),result,gt

def main():
    args=get_args()
    device_name = "cuda"
    cuda_devices=args.cuda_devices
    if cuda_devices is not None and len(cuda_devices) >= 1:
        device_name += ":" + str(cuda_devices[0])
    else:
        device_name="cpu"
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    # features=np.load(args.data_path+"/features.npy")
    # label=np.load(args.data_path+"/label.npy")
    # dataset=My_dataset(features,label)

    # with open(args.data_path+"/data.pkl", 'rb') as f:
    with open(args.data_path+"/data_au.pkl", 'rb') as f:
        data_pkl = pickle.load(f)
    dataset=My_dataset(data_pkl)

    # train_data, test_data = train_test_split(dataset, test_size=args.test_prop)
    data_num = len(dataset)
    test_num = int(data_num*args.test_prop)
    train_num = data_num-test_num
    # train_data = dataset[:train_num]
    # test_data = dataset[train_num:]
    train_data = torch.utils.data.Subset(dataset, list(range(train_num)))
    test_data = torch.utils.data.Subset(dataset, list(range(train_num, data_num)))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


    # model = My_Mamba(input_dim=args.input_dim,output_dim=args.output_dim,hidden_dim=args.hidden_dim, state_size=args.state_size, num_hidden_layers=args.num_hidden_layers, norm=args.norm).to(device)
    # name="{:}_{:}_{:}".format(args.hidden_dim,args.state_size,args.num_hidden_layers)
    match args.model:
        case 'mamba':
            model = My_Mamba(input_dim=args.input_dim, output_dim=128, hidden_dim=args.hidden_dim,
                             state_size=args.state_size, num_hidden_layers=args.num_hidden_layers,norm=args.norm).to(device)
            name = "Mamba_{:}_{:}_{:}".format(args.hidden_dim, args.state_size, args.num_hidden_layers)
        case 'rnn':
            model = RNN(input_dim=args.input_dim, class_num=128,norm=args.norm).to(device)
            name="RNN"
        case 'lstm':
            model = LSTM(input_dim=args.input_dim, class_num=128, norm=args.norm).to(device)
            name="LSTM"
        case 'bert':
            model = My_BERT(input_dim=args.input_dim, output_dim=128, hidden_dim=args.hidden_dim,
                             intermediate_size=args.intermediate_size, num_attention_heads=args.num_attention_heads, num_hidden_layers=args.num_hidden_layers, norm=args.norm).to(device)
            name = "BERT_{:}_{:}_{:}_{:}".format(args.hidden_dim, args.intermediate_size, args.num_attention_heads,args.num_hidden_layers)
        case 'tcn':
            model = TCN(input_size=args.input_dim,output_size=128,num_channels=[30]*8,kernel_size=7,dropout=0.1).to(device)
            name="TCN"
        case _:
            print("No Such Model!")
            exit(-1)

    # classifier = MLP([128,64,32,3]).to(device)
    classifier = MLP([128,64,32,2]).to(device)
    predictor = MLP([128,64,32,1]).to(device)


    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)
        classifier = nn.DataParallel(classifier, device_ids=cuda_devices)
        predictor = nn.DataParallel(predictor, device_ids=cuda_devices)


    parameters = list(model.parameters()) + list(classifier.parameters()) + list(predictor.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr)#, weight_decay=0.01)
    loss_func=nn.MSELoss()

    best_mse = 1e8
    mse_epoch = 0
    j = 0

    while True:
        j += 1
        loss,mape,mse,acc,_,_ = iteration(model,classifier,predictor,train_loader,optim,loss_func,device,train=True)
        log = "Epoch {} | Train Loss {:06f}, Train MAPE {:06f}, Train MSE {}, Train Acc {:06f} | ".format(j, loss, mape, mse, acc)
        print(log)
        # with open("log.txt", 'a') as file:
        with open(name+".txt", 'a') as file:
            file.write(log)
        loss,mape,mse,acc,result,gt = iteration(model,classifier,predictor,test_loader,optim,loss_func,device,train=False)

        correlation, _ = pearsonr(result.cpu(), gt.cpu())
        r2 = r2_score(result.cpu(), gt.cpu())

        result=result*gt
        stability=torch.mean(result)/torch.std(result)
        log = "Test Loss {:06f}, Test MAPE {:06f}, Test MSE {}, Test Acc {:06f} | Test Stability {:06f}, Correlation {:06f}, R2 {:06f}".format(loss, mape, mse, acc, stability, correlation, r2)
        print(log)
        # with open("log.txt", 'a') as file:
        with open(name+".txt", 'a') as file:
            file.write(log + "\n")

        if j%1==0:
            result=torch.cumsum(result,dim=0)
            np.save(name + ".npy",result.cpu())

        if mse <= best_mse:
            torch.save(model.state_dict(), name+".pth")
            torch.save(classifier.state_dict(), name+"_classifier.pth")
            torch.save(predictor.state_dict(), name+"_predictor.pth")
            # torch.save(model.state_dict(), "model.pth")
            mse_epoch = 0
            best_mse = mse
        else:
            mse_epoch += 1

        print("MSE Epcoh {:}".format(mse_epoch))
        if mse_epoch > args.epoch:
            break

if __name__ == '__main__':
    main()