import torch.nn as nn
import torch
import torch.nn.functional as F


from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,ARMAConv,global_mean_pool,GATConv,ChebConv,GCNConv)
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models.Bern import BernConv


class BernConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, hidden_size, layers, dropout, K=10):
        super(BernConvLayer, self).__init__()
        self.hidden_size = hidden_size        
        self.drop = dropout

        self.layers = layers
        self.head_dim = self.hidden_size // self.layers

        self.bern_0 = BernConv(self.hidden_size + self.head_dim * 0, self.head_dim)
        self.bern_1 = BernConv(self.hidden_size + self.head_dim * 1, self.head_dim)
        self.bern_2 = BernConv(self.hidden_size + self.head_dim * 2, self.head_dim)
        if self.layers > 3 :
            self.bern_3 = BernConv(self.hidden_size + self.head_dim * 3, self.head_dim)

        self.fc2 = torch.nn.Linear(360, 1)
        self.coe = Parameter(torch.Tensor(K+1))

        self.linear_output = nn.Linear(self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.coe.data.fill_(1)

    def DiagCat(self, mat):

        def DiagCat_Base(mat_1, mat_2):
            len_1 = mat_1.size(0)
            len_2 = mat_2.size(0)

            mat_1 = torch.cat( (mat_1, torch.zeros(len_1, len_2).to(mat.device)),  dim=1)
            mat_2 = torch.cat( (torch.zeros(len_2, len_1).to(mat.device), mat_2),  dim=1)

            return torch.cat( (mat_1, mat_2),  dim=0)


        tmp = DiagCat_Base(mat[0], mat[1])
        for i in range(1, mat.size(0) - 1):
            new = DiagCat_Base(tmp, mat[i+1])
            tmp = new ; del new

    
        return tmp

    def forward(self, adj, input):
        edge_index = ( self.DiagCat(adj) > 0).nonzero().t().contiguous()
        outputs = input_cat = torch.cat( [i for i in input] , dim=0)
        cache_list = [outputs]   
        output_list = []         


        outputs = F.relu(self.bern_0(outputs, edge_index, self.coe))
        output_list.append(self.drop(outputs))
        cache_list.append(outputs)
        outputs = torch.cat(cache_list, dim=-1)
        
        outputs = F.relu(self.bern_1(outputs, edge_index, self.coe))
        output_list.append(self.drop(outputs))
        cache_list.append(outputs)
        outputs = torch.cat(cache_list, dim=-1)
        

        outputs = F.relu(self.bern_2(outputs, edge_index, self.coe))
        output_list.append(self.drop(outputs))
        cache_list.append(outputs)
        outputs = torch.cat(cache_list, dim=-1)

        if self.layers > 3 :
            outputs = F.relu(self.bern_3(outputs, edge_index, self.coe))
            output_list.append(self.drop(outputs))
            cache_list.append(outputs)
            outputs = torch.cat(cache_list, dim=-1)


        bern_outputs = torch.cat(output_list, dim=-1)
        bern_outputs = bern_outputs + input_cat

        out = self.linear_output(bern_outputs)  
        



        N = []
        for i in input:
            N.append(i.size(0))
        N.insert(0, 0)

        for i in range(len(N) - 1):
            N[i+1] = N[i] + N[i+1]

        
        out = [ (out.clone().detach()[ N[i] : N[i+1] ]).tolist()  for i in range(len(N)-1) ]
        out = torch.tensor(out)

        return out.cuda()