import torch
from torch import nn
import torch.nn.functional as F

class MLPs(nn.Module):
    """ Build specific multi-layer perceptron for each aspect"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_aspects, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.output_dim=output_dim
        self.num_aspects = num_aspects
        self.MLP_list = nn.ModuleList()
        for _ in range(self.num_aspects):
            layer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim))
            self.MLP_list.append(layer)

    def forward(self, input_ids, aspect_id, **kwarg):
        x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        x = x.mean(dim=1) # [batch_size, hidden_dim]
        output = torch.zeros((input_ids.shape[0], self.output_dim), dtype=x.dtype, device=x.device)
        for i, aspect_idx in enumerate(aspect_id):
            layer = self.MLP_list[aspect_idx]
            output[i] = layer(x[i].unsqueeze(0))
        return output # [batch_size, output_dim]
    
class MLP(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) for all aspects"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, input_ids, aspect_token_id, **kwarg):
        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.cat((input_x, aspect_x.unsqueeze(1)), dim=1) # [batch_size, seq_length+1, hidden_dim]
        x = x.mean(dim=1) # [batch_size, hidden_dim]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < (self.num_layers - 1) else layer(x)
        return x # [batch_size, output_dim]
    
class MLPv2(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) for all aspects"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.linear_proj = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, input_ids, aspect_token_id, **kwarg):
        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.cat((input_x.mean(dim=1), aspect_x), dim=-1) # [batch_size, 2*hidden_dim]
        x = F.relu(self.linear_proj(x)) # [batch_size, hidden_dim]
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < (self.num_layers - 1) else layer(x)
        return x # [batch_size, output_dim]