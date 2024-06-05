from transformers import MambaConfig,MambaModel,MambaForCausalLM,BertConfig,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F

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
    model = My_BERT(input_dim=12, output_dim=1, hidden_dim=768)

    x=torch.randn([5,60,12])
    y=model(x)
    print(y.shape)