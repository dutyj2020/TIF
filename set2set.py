import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
class Set2Set(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_fn=nn.ReLU, num_layers=1):
        
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if hidden_dim <= input_dim:
            print('ERROR: Set2Set output_dim should be larger than input_dim')
        self.lstm_output_dim = hidden_dim - input_dim
        self.lstm = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)
        self.pred = nn.Linear(hidden_dim, input_dim)
        self.act = act_fn()
    def forward(self, embedding):
        
        batch_size = embedding.size()[0]
        n = embedding.size()[1]
        hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).cuda(),
                  torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).cuda())
        q_star = torch.zeros(batch_size, 1, self.hidden_dim).cuda()
        for i in range(n):
            q, hidden = self.lstm(q_star, hidden)
            e = embedding @ torch.transpose(q, 1, 2)
            a = nn.Softmax(dim=1)(e)
            r = torch.sum(a * embedding, dim=1, keepdim=True)
            q_star = torch.cat((q, r), dim=2)
        q_star = torch.squeeze(q_star, dim=1)
        out = self.act(self.pred(q_star))
        return out