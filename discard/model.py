from transformers import MambaConfig,MambaModel,MambaForCausalLM,BertConfig,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Mamba(nn.Module):
    def __init__(self,hidden_dim=768, state_size=16, num_hidden_layers=32):
        super().__init__()
        self.config=MambaConfig(hidden_size=hidden_dim, state_size=state_size, num_hidden_layers=num_hidden_layers)
        self.model=MambaModel(self.config)
        self.model.embeddings=nn.Identity()

    def forward(self,x):
        y=self.model(inputs_embeds=x)
        return y.last_hidden_state

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class My_Mamba(nn.Module):
    def __init__(self,input_dim=13,output_dim=1,length=60,hidden_dim=768, state_size=16, num_hidden_layers=32, da=128, r=4, norm=False):
        super().__init__()
        self.linear_input=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,hidden_dim)
        )

        self.linear_output=nn.Sequential(
            nn.Linear(hidden_dim*r, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.attention = SelfAttention(input_dim=hidden_dim, da=da, r=r)
        self.mamba=Mamba(hidden_dim, state_size, num_hidden_layers)
        self.norm=norm

    def forward(self,x):
        if self.norm:
            mean=torch.mean(x,dim=-1,keepdim=True)
            std=torch.std(x,dim=-1,keepdim=True)
            x=(x-mean)/std

        x=self.linear_input(x)
        x=self.mamba(x)
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.linear_output(flatten)
        return res

class RNN(nn.Module):
    def __init__(self,class_num,input_dim=13, norm=False):
        super().__init__()
        self.rnn = nn.RNN(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, class_num)
        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        x, _ = self.rnn(x)
        y = self.fc(x[:,-1,:])
        return y

class LSTM(nn.Module):
    def __init__(self,class_num,input_dim=13, norm=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=4, batch_first=True)
        self.fc = nn.Linear(64, class_num)
        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        x, _ = self.lstm(x)
        y = self.fc(x[:,-1,:])
        return y

class BERT(nn.Module):
    def __init__(self,hidden_dim=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12):
        super().__init__()
        self.config=BertConfig(hidden_size=hidden_dim, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads,  num_hidden_layers=num_hidden_layers)
        self.model=BertModel(self.config)

    def forward(self,x):
        y=self.model(inputs_embeds=x,attention_mask=None)
        return y.last_hidden_state

class My_BERT(nn.Module):
    def __init__(self,input_dim=13,output_dim=1,length=60,hidden_dim=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12, da=128, r=4, norm=False):
        super().__init__()
        self.linear_input = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )

        self.linear_output = nn.Sequential(
            nn.Linear(hidden_dim * r, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.attention = SelfAttention(input_dim=hidden_dim, da=da, r=r)
        self.bert = BERT(hidden_dim, intermediate_size, num_attention_heads, num_hidden_layers)
        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std

        x = self.linear_input(x)
        x = self.bert(x)
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.linear_output(flatten)
        return res


# TCN: https://github.com/locuslab/TCN/tree/master
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x=torch.transpose(x,-1,-2)
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])

#test
if __name__ == '__main__':
    # model=Mamba(768)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total parameters:', total_params)
    #
    # x=torch.randn([5,128,768])
    # y=model(x)
    # print(y.shape)

    # model=My_Mamba(input_dim=12,output_dim=1,hidden_dim=768)
    # model=LSTM(class_num=1,input_dim=12)
    # model = RNN(class_num=1, input_dim=12)
    # model = My_BERT(input_dim=12, output_dim=1, hidden_dim=768)

    model = TCN(input_size=12,output_size=1,num_channels=[30]*8,kernel_size=7,dropout=0.1)

    x=torch.randn([5,60,12])
    y=model(x)
    print(y.shape)