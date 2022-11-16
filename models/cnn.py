import torch
from torch import nn
import torch.nn.functional as F

class TextCNNv1(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.cat((input_x, aspect_x.unsqueeze(1)), dim=1) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(x))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
class TextCNNv2(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.add(input_x, aspect_x.unsqueeze(1)) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(x))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

class TextCNNv3(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.mul(input_x, aspect_x.unsqueeze(1)) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(x))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
class TextCNNv1_maxpool(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool1 = nn.MaxPool1d(499)
        self.pool2 = nn.MaxPool1d(498)
        self.pool3 = nn.MaxPool1d(497)
        self.pools=[self.pool1,self.pool2,self.pool3]
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.cat((input_x, aspect_x.unsqueeze(1)), dim=1) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        i=0
        result=[]
        for conv in self.convs:
            ele = torch.squeeze(self.relu(self.pools[i](conv(x))), dim=-1)
            result.append(ele)
            i+=1
        encoding = torch.cat(result,dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
class TextCNNv2_maxpool(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool1 = nn.MaxPool1d(498)
        self.pool2 = nn.MaxPool1d(497)
        self.pool3 = nn.MaxPool1d(496)
        self.pools=[self.pool1,self.pool2,self.pool3]
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.add(input_x, aspect_x.unsqueeze(1)) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        i=0
        result=[]
        for conv in self.convs:
            ele = torch.squeeze(self.relu(self.pools[i](conv(x))), dim=-1)
            result.append(ele)
            i+=1
        encoding = torch.cat(result,dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
class TextCNNv3_maxpool(nn.Module):
    def __init__(self,embed_size, kernel_sizes, num_channels,embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 3)
        self.pool1 = nn.MaxPool1d(498)
        self.pool2 = nn.MaxPool1d(497)
        self.pool3 = nn.MaxPool1d(496)
        self.pools=[self.pool1,self.pool2,self.pool3]
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(embed_size, c, k))

    def forward(self, input_ids, aspect_token_id,**kwarg):

        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.mul(input_x, aspect_x.unsqueeze(1)) # [batch_size, seq_length+1, hidden_dim]
        x = x.permute(0,2,1)
        i=0
        result=[]
        for conv in self.convs:
            ele = torch.squeeze(self.relu(self.pools[i](conv(x))), dim=-1)
            result.append(ele)
            i+=1
        encoding = torch.cat(result,dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs