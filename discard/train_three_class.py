from model import My_Mamba,RNN,LSTM,My_BERT
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
import tqdm
from torch.utils.data import DataLoader
import pickle
from dataset import Three_label_dataset

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=13)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--state_size', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=16)
    parser.add_argument('--intermediate_size', type=int, default=3072)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument("--data_path",type=str,default="/disk1/imb/202305_all")
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--model",type=str,default="mamba")
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--norm", action="store_true",default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--test_prop', type=float, default=0.1)
    args = parser.parse_args()
    return args


def iteration(model,data_loader,optim,loss_func,device,train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_list = []
    acc_list = []
    result=None
    gt=None

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, label, target in pbar:
        x=x.float().to(device)
        target=target.float().to(device)
        label=label.long().to(device)
        label+=1

        y=model(x)
        loss=loss_func(y,label)
        loss_list.append(loss.item())

        y = torch.argmax(y,dim=-1)

        # print(label)
        # print(y)

        acc=torch.mean((y==label).float())
        acc_list.append(acc.item())

        y-=1
        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
        else:
            if result is None:
                result=y
                gt=target
            else:
                result=torch.cat([result,y],dim=0)
                gt=torch.cat([gt,target],dim=0)
    return np.mean(loss_list),np.mean(acc_list),result,gt


def main():
    args=get_args()
    device_name = "cuda"
    cuda_devices=args.cuda_devices
    if cuda_devices is not None and len(cuda_devices) >= 1:
        device_name += ":" + str(cuda_devices[0])
    else:
        device_name="cpu"
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    with open(args.data_path + "/data_au_label.pkl", 'rb') as f:
        data_pkl = pickle.load(f)

    # loss_func = nn.CrossEntropyLoss()
    weight = torch.tensor([6.0,1.0,6.0]).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)
    class_num = 3

    dataset = Three_label_dataset(data_pkl)
    data_num = len(dataset)
    test_num = int(data_num * args.test_prop)
    train_num = data_num - test_num
    train_data = torch.utils.data.Subset(dataset, list(range(train_num)))
    test_data = torch.utils.data.Subset(dataset, list(range(train_num, data_num)))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    match args.model:
        case 'mamba':
            model = My_Mamba(input_dim=args.input_dim, output_dim=class_num, hidden_dim=args.hidden_dim,
                             state_size=args.state_size, num_hidden_layers=args.num_hidden_layers,norm=args.norm).to(device)
            name = "Mamba_{:}_{:}_{:}".format(args.hidden_dim, args.state_size, args.num_hidden_layers)
        case 'rnn':
            model = RNN(input_dim=args.input_dim, class_num=class_num,norm=args.norm).to(device)
            name="RNN"
        case 'lstm':
            model = LSTM(input_dim=args.input_dim, class_num=class_num, norm=args.norm).to(device)
            name="LSTM"
        case 'bert':
            model = My_BERT(input_dim=args.input_dim, output_dim=class_num, hidden_dim=args.hidden_dim,
                             intermediate_size=args.intermediate_size, num_attention_heads=args.num_attention_heads, num_hidden_layers=args.num_hidden_layers, norm=args.norm).to(device)
            name = "BERT_{:}_{:}_{:}_{:}".format(args.hidden_dim, args.intermediate_size, args.num_attention_heads,args.num_hidden_layers)
        case _:
            print("No Such Model!")
            exit(-1)

    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)

    parameters = set(model.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01)

    best_loss = 1e8
    best_acc = 0

    loss_epoch = 0
    acc_epoch = 0
    j = 0

    while True:
        j += 1
        loss,acc,_,_ = iteration(model,train_loader,optim,loss_func,device,train=True)
        log = "Epoch {} | Train Acc {:03f} , Train Loss {:06f} | ".format(j, acc, loss)
        print(log)
        with open(name+".txt", 'a') as file:
            file.write(log)
        loss,acc,result,gt = iteration(model,test_loader,optim,loss_func,device,train=False)
        result=result*gt
        stability=torch.mean(result)/torch.std(result)
        log = "Test Acc {:03f} , Test Loss {:06f} | Test Stability {:06f}".format(acc, loss, stability)
        print(log)
        with open(name+".txt", 'a') as file:
            file.write(log + "\n")

        # if j%10==0:
        result=torch.cumsum(result,dim=0)
        np.save(name + ".npy",result.cpu())

        if acc >= best_acc or loss <= best_loss:
            torch.save(model.state_dict(), name+".pth")

        if acc >= best_acc:
            acc_epoch = 0
        else:
            acc_epoch += 1
        if loss <= best_loss:
            loss_epoch = 0
        else:
            loss_epoch += 1

        print("Acc Epoch {:},Loss Epcoh {:}".format(acc_epoch,loss_epoch))
        if loss_epoch > args.epoch and acc_epoch > args.epoch:
            break

if __name__ == '__main__':
    main()