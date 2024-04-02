from dataset import My_dataset
from model import My_Mamba
import torch.nn as nn
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
import tqdm
from torch.utils.data import DataLoader
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=13)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--state_size', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=16)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument("--data_path",type=str,default="/disk1/imb/202305_all")
    parser.add_argument("--model_path",type=str,default=None)
    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epoch', type=int, default=30)
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

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, label in pbar:
        x=x.float().to(device)
        label=label.float().to(device)
        y=model(x)
        loss=loss_func(y,label)
        loss_list.append(loss.item())

        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()

    return np.mean(loss_list)

def main():
    args=get_args()
    device_name = "cuda:"+args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    # features=np.load(args.data_path+"/features.npy")
    # label=np.load(args.data_path+"/label.npy")
    # dataset=My_dataset(features,label)

    with open(args.data_path+"/data.pkl", 'rb') as f:
        data_pkl = pickle.load(f)
    dataset=My_dataset(data_pkl)

    train_data, test_data = train_test_split(dataset, test_size=args.test_prop)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


    model = My_Mamba(input_dim=args.input_dim,output_dim=args.output_dim,hidden_dim=args.hidden_dim, state_size=args.state_size, num_hidden_layers=args.num_hidden_layers).to(device)
    parameters = set(model.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01)
    loss_func=nn.MSELoss()

    best_loss = 1000
    loss_epoch = 0
    j = 0

    while True:
        j += 1
        loss = iteration(model,train_loader,optim,loss_func,device,train=True)
        log = "Epoch {} | Train Loss {:06f} | ".format(j, loss)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log)
        loss = iteration(model,test_loader,optim,loss_func,device,train=False)
        log = "Test Loss {:06f}".format(loss)
        print(log)
        with open(args.task + ".txt", 'a') as file:
            file.write(log + "\n")
        if loss <= best_loss:
            torch.save(model.state_dict(), args.task + ".pth")
            loss_epoch = 0
        else:
            loss_epoch += 1
        print("Loss Epcoh {:}".format(loss_epoch))
        if loss_epoch > args.epoch:
            break

if __name__ == '__main__':
    main()