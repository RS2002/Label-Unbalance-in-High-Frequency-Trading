from transformers import MambaConfig,MambaModel,BertConfig,BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

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


class MLP(nn.Module):
    def __init__(self, layer_sizes=[13*60,64,64,1], norm=False):
        super(MLP, self).__init__()
        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes 必须至少包含两个元素，分别代表输入层和输出层的大小。")
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        self.norm = norm

    def forward(self, x):
        if self.norm:
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x = (x - mean) / std
        x = x.reshape([-1,self.layer_sizes[0]])
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x)
        return x


class CSi_Net(nn.Module):
    def __init__(self, input_dims=64, hidden_dims=64, head=1, method="attention"):
        super().__init__()
        self.score=method
        if hidden_dims%head!=0:
            print("ERROR")
            exit(-1)
        self.q_linear=nn.Linear(input_dims,hidden_dims)
        self.k_linear=nn.Linear(input_dims,hidden_dims)
        self.sigmoid=nn.Sigmoid()
        self.head=head
        self.num=hidden_dims//head
        self.input_dims=input_dims


    def forward(self,q,k):
        if self.score=="attention":

            query=self.q_linear(q)
            key=self.k_linear(k)

            query=query.view(-1,self.head,self.num).transpose(0, 1)
            key=key.view(-1,self.head,self.num).transpose(0, 1)

            attn_matrix=torch.bmm(query,key.transpose(1, 2))
            attn_matrix=torch.sum(attn_matrix,dim=0)

            return self.sigmoid(attn_matrix)
        elif self.score=="distance":
            gaussian_dist = torch.cdist(q, k, p=2)
            return gaussian_dist
        elif self.score=="cosine":
            q_normalized = F.normalize(q, dim=1)
            k_normalized = F.normalize(k, dim=1)
            cos_sim = torch.mm(q_normalized, k_normalized.t())
            return (cos_sim+1)/2
        else:
            print("ERROR")
            exit(-1)


class Resnet(nn.Module):
    def __init__(self, output_dims=64, channel=1, pretrained=True, norm=False):
        super().__init__()
        self.model=models.resnet18(pretrained)
        self.model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, output_dims)
        self.norm=norm

    def forward(self,x):
        return self.model(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet5, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Weight_Net(nn.Module):
    def __init__(self,dim=100):
        super().__init__()
        self.model = MLP([dim*dim,4096,1024,256,dim])
        # self.model = MLP([dim*dim,256,64,dim])


    def forward(self,x):
        x = torch.squeeze(x,dim=1)
        return self.model(x)



if __name__ == '__main__':
    # x=torch.randn([2,60,13])
    #
    # # mamba = Mamba(input_dim=13,output_dim=1,hidden_dim=64, state_size=4, num_hidden_layers=4, da=128, r=4, norm=True)
    # # total_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    # # print('total parameters:', total_params)
    # # y = mamba(x)
    # # print(y.shape)
    # #
    # # bert = BERT(input_dim=13,output_dim=1,hidden_dim=64, intermediate_size=64, num_attention_heads=4, num_hidden_layers=4, da=128, r=4, norm=True)
    # # total_params = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    # # print('total parameters:', total_params)
    # # y = bert(x)
    # # print(y.shape)
    # #
    # # lstm = LSTM(input_dim=13, output_dim=1, hidden_dim=64, da=128, r=4, norm=False)
    # # total_params = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
    # # print('total parameters:', total_params)
    # # y = lstm(x)
    # # print(y.shape)
    #
    # mlp = MLP(layer_sizes=[13*60,64,64,1], norm=False)
    # total_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    # print('total parameters:', total_params)
    # y = mlp(x)
    # print(y.shape)

    x = torch.randn([1,1,100,100])
    # resnet = ResNet5(num_classes=100)
    # total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    # print('total parameters:', total_params)
    # y  = resnet(x)
    # print(y.shape)
    weightnet = Weight_Net(100)
    total_params = sum(p.numel() for p in weightnet.parameters() if p.requires_grad)
    print('total parameters:', total_params)
    y = weightnet(x)
    print(y.shape)

