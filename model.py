from transformers import MambaConfig,MambaModel,MambaForCausalLM
import torch.nn as nn
import torch

class Mamba(nn.Module):
    def __init__(self,hidden_dim=768, state_size=16, num_hidden_layers=32):
        super().__init__()
        self.config=MambaConfig(hidden_size=hidden_dim, state_size=state_size, num_hidden_layers=num_hidden_layers)
        self.model=MambaModel(self.config)
        self.model.embeddings=nn.Identity()

    def forward(self,x):
        y=self.model(inputs_embeds=x,output_hidden_states=True)
        return y.last_hidden_state

class My_Mamba(nn.Module):
    def __init__(self,input_dim=13,output_dim=1,length=60,hidden_dim=768, state_size=16, num_hidden_layers=32):
        super().__init__()
        self.linear_input=nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,hidden_dim)
        )

        self.linear_output=nn.Sequential(
            nn.Linear(hidden_dim*length, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        self.mamba=Mamba(hidden_dim, state_size, num_hidden_layers)

    def forward(self,x):
        x=self.linear_input(x)
        x=self.mamba(x)
        x=x.reshape(x.shape[0],-1)
        x=self.linear_output(x)
        return x

#test
if __name__ == '__main__':
    # model=Mamba(768)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total parameters:', total_params)
    #
    # x=torch.randn([5,128,768])
    # y=model(x)
    # print(y.shape)

    model=My_Mamba(input_dim=12,output_dim=1,hidden_dim=768)
    x=torch.randn([5,60,12])
    y=model(x)
    print(y.shape)