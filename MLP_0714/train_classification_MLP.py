from models import Mamba, BERT, LSTM, MLP_ZXM
import torch.nn as nn
import numpy as np
import argparse
import torch
import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from dataset import load_data

IMPORTANT_CONSTANTS = {
    'hidden_dim': 128,
    'batch_size': 512,
    'lr': 0.00001,
    'l_epoch': 30,
    'a_epoch': 30,
    'normalization': True,
}

# version_num = 9, lr = 0.000025, normalization = True；hidden_dim = 128 训练117次

version_num = 10
date_str = '0711'
nor_str = 'Nor' if IMPORTANT_CONSTANTS['normalization'] else 'noNor'

store_dir = f"/home/imb_xm/mlp_result_{date_str}/"
log_file_name = f"_log_{nor_str}_{date_str}_v{version_num}.txt"
output_file_name = f"_output_{nor_str}_{date_str}_v{version_num}.npy"
gt_file_name = f"_gt_{nor_str}_{date_str}_v{version_num}.npy"
label_file_name = f"_label_{nor_str}_{date_str}_v{version_num}.npy"
model_save_name = f"_{nor_str}_{date_str}_v{version_num}.pth"

def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input_dim', type=int, default=13)
    parser.add_argument('--output_dim', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=IMPORTANT_CONSTANTS['hidden_dim'])
    # mamba & bert
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    # mamba
    parser.add_argument('--state_size', type=int, default=4)
    # bert
    parser.add_argument('--intermediate_size', type=int, default=64)
    parser.add_argument('--num_attention_heads', type=int, default=4)

    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--norm", action="store_true", default=False)

    parser.add_argument("--data_path", type=str, default="/disk1/imb/202305_all")
    parser.add_argument("--item_list", type=str, nargs='+',
                        default=["ag2308", "au2308", "fu2309", "ni2306", "rb2310", "sn2306"])

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[2, 3], help="CUDA device ids")
    parser.add_argument('--num_workers', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=IMPORTANT_CONSTANTS['batch_size'])
    parser.add_argument('--lr', type=float, default=IMPORTANT_CONSTANTS['lr'])
    parser.add_argument('--l_epoch', type=int, default=IMPORTANT_CONSTANTS['l_epoch'])
    parser.add_argument('--a_epoch', type=int, default=IMPORTANT_CONSTANTS['a_epoch'])
    parser.add_argument('--test_prop', type=float, default=0.2)
    parser.add_argument('--normalization', type=bool, default=IMPORTANT_CONSTANTS['normalization'])
    args = parser.parse_args()
    return args


def iteration(model, data_loader, optim, loss_func, device, train=True):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_list = []
    acc_list = []
    result = None
    gt = None
    label_list = None

    pbar = tqdm.tqdm(data_loader, disable=False)
    for x, ret, label in pbar:
        label += 1
        x = x.float().to(device)
        label = label.long().to(device)
        
        y = model(x)
        loss = loss_func(y, label)
        loss_list.append(loss.item())

        y = torch.argmax(y, dim=-1)
        acc = torch.mean((y == label).float())
        acc_list.append(acc.item())
        y -= 1

        if train:
            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optim.step()
        else:
            if result is None:
                result = y
                gt = ret
                label_list = label - 1
            else:
                result = torch.cat([result, y], dim=0)
                gt = torch.cat([gt, ret], dim=0)
                label_list = torch.cat([label_list, label - 1], dim=0)

    return np.mean(loss_list), np.mean(acc_list), result, gt, label_list


def main():
    args = get_args()
    device_name = "cuda"
    cuda_devices = args.cuda_devices
    if cuda_devices is not None and len(cuda_devices) >= 1:
        device_name += ":" + str(cuda_devices[0])
    else:
        device_name = "cpu"
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    match args.model:
        case 'mamba':
            model = Mamba(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim,
                          state_size=args.state_size, num_hidden_layers=args.num_hidden_layers, norm=args.norm).to(
                device)
            name = "Mamba_{:}_{:}_{:}_{:}".format(args.hidden_dim, args.state_size, args.num_hidden_layers,
                                                  int(args.norm))
        case 'lstm':
            model = LSTM(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim,
                         norm=args.norm).to(device)
            name = "LSTM_{:}_{:}".format(args.hidden_dim, int(args.norm))
        case 'bert':
            model = BERT(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim,
                         intermediate_size=args.intermediate_size, num_attention_heads=args.num_attention_heads,
                         num_hidden_layers=args.num_hidden_layers, norm=args.norm).to(device)
            name = "BERT_{:}_{:}_{:}_{:}_{:}".format(args.hidden_dim, args.intermediate_size, args.num_attention_heads,
                                                     args.num_hidden_layers, int(args.norm))
        # TODO: change
        case 'mlp':
            model = MLP_ZXM(input_dim=args.input_dim, length=60, hidden_dim=args.hidden_dim,
                            output_dim=args.output_dim).to(device)
            name = "MLP"
        case _:
            print("No Such Model!")
            exit(-1)

    if len(cuda_devices) > 1 and not args.cpu:
        model = nn.DataParallel(model, device_ids=cuda_devices)

    parameters = set(model.parameters())
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print('total parameters:', total_params)
    optim = torch.optim.Adam(parameters, lr=args.lr, weight_decay=0.01) 
    loss_func = nn.CrossEntropyLoss()

    item_list = args.item_list
    test_loader = {}
    train_data = None
    for item in item_list:
        train_data_temp, test_data_temp = load_data(
            data_path=args.data_path + "/data_all_" + item + "_factors_and_label.csv",
            train_prop=1 - args.test_prop, normalization=args.normalization)
        if train_data is None:
            train_data = train_data_temp
        else:
            train_data = ConcatDataset([train_data, train_data_temp])
        test_loader[item] = DataLoader(test_data_temp, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    best_loss = 1e8
    best_acc = 0
    loss_epoch = 0
    acc_epoch = 0
    j = 0

    while True:
        j += 1
        acc_list = []
        loss_list = []

        loss, acc, _, _, _ = iteration(model, train_loader, optim, loss_func, device, train=True)
        log = "Epoch {} | Train Loss {}, Train Acc {:06f}".format(j, loss, acc)
        print(log)
        with open(store_dir + name + log_file_name, 'a') as file:
            file.write("\n" + log)

        for item in item_list:
            loss, acc, output, gt, label_list = iteration(model, test_loader[item], optim, loss_func, device,
                                                          train=False)
            acc_list.append(acc)
            loss_list.append(loss)
            log = " | " + item + " Test Loss {}, Test Acc {:06f}".format(loss, acc)
            print(log)
            with open(store_dir + name + log_file_name, 'a') as file:
                file.write(log)

            np.save(store_dir + item + output_file_name, output.cpu())
            np.save(store_dir + item + gt_file_name, gt.cpu())
            np.save(store_dir + item + label_file_name, label_list.cpu())

        acc, loss = np.mean(acc_list), np.mean(loss_list)
        if loss <= best_loss:
            # torch.save(model.state_dict(), store_dir + name + model_save_name)
            loss_epoch = 0
            best_loss = loss
        else:
            loss_epoch += 1
        if acc > best_acc:
            torch.save(model.state_dict(), store_dir + name + model_save_name)
            best_acc = acc
            acc_epoch = 00
        else:
            acc_epoch += 1
        print("Loss Epcoh {:}, Acc Epoch {:}".format(loss_epoch, acc_epoch))
        if loss_epoch > args.l_epoch and acc_epoch > args.a_epoch:
            break


if __name__ == '__main__':
    main()
