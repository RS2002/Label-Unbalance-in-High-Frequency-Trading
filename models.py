from transformers import MambaConfig,MambaModel,BertConfig,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat

class Mamba(nn.Module):
    def __init__(self,input_dim=13,output_dim=1,hidden_dim=768, state_size=16, num_hidden_layers=32, da=128, r=4, norm=False):
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

        self.config=MambaConfig(hidden_size=hidden_dim, state_size=state_size, num_hidden_layers=num_hidden_layers)
        self.mamba=MambaModel(self.config)
        self.mamba.embeddings = nn.Identity()

        self.norm=norm

    def forward(self,x):
        if self.norm:
            mean=torch.mean(x,dim=-1,keepdim=True)
            std=torch.std(x,dim=-1,keepdim=True)
            x=(x-mean)/std

        x=self.linear_input(x)
        x=self.mamba(x)
        x=x.last_hidden_state

        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.linear_output(flatten)
        return res

class BERT(nn.Module):
    def __init__(self,input_dim=13,output_dim=1,hidden_dim=768, intermediate_size=3072, num_attention_heads=12, num_hidden_layers=12, da=128, r=4, norm=False):
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

        self.config=BertConfig(hidden_size=hidden_dim, intermediate_size=intermediate_size, num_attention_heads=num_attention_heads,  num_hidden_layers=num_hidden_layers)
        self.bert = BertModel(self.config)

        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std

        x = self.linear_input(x)
        x = self.bert(inputs_embeds=x)
        x = x.last_hidden_state

        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.linear_output(flatten)
        return res

class LSTM(nn.Module):
    def __init__(self, input_dim=13, output_dim=1, hidden_dim=256, da=128, r=4, norm=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=4, batch_first=True)

        self.attention = SelfAttention(input_dim=hidden_dim, da=da, r=r)

        self.linear_output = nn.Sequential(
            nn.Linear(hidden_dim * r, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        x, _ = self.lstm(x)

        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.linear_output(flatten)

        return res


if __name__ == '__main__':
    x=torch.randn([2,60,13])

    mamba = Mamba(input_dim=13,output_dim=1,hidden_dim=64, state_size=4, num_hidden_layers=4, da=128, r=4, norm=True)
    total_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    y = mamba(x)
    print(y.shape)

    bert = BERT(input_dim=13,output_dim=1,hidden_dim=64, intermediate_size=64, num_attention_heads=4, num_hidden_layers=4, da=128, r=4, norm=True)
    total_params = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    y = bert(x)
    print(y.shape)

    lstm = LSTM(input_dim=13, output_dim=1, hidden_dim=64, da=128, r=4, norm=False)
    total_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    y = lstm(x)
    print(y.shape)
