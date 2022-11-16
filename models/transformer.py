import torch
from torch import nn
import torch.nn.functional as F
import math

class TransformerEncoderV0(nn.Module):
    """ Transformer encoder"""
    def __init__(self, input_dim, output_dim, num_layers, embedding_matrix, seq_length, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.pos_encoder = PositionalEncoding(512, 0.1, seq_length+1)
        self.linear_proj = nn.Linear(input_dim, 512)
        
        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=num_layers)
        
        self.clf = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Dropout(p=drop),
            nn.Linear(512, output_dim),
        )

    def forward(self, input_ids, aspect_token_id, attention_mask, **kwarg):
        input_x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_token_id)  # [batch_size, hidden_dim]
        x = torch.cat((aspect_x.unsqueeze(1), input_x), dim=1) # [batch_size, seq_length+1, hidden_dim]
        x = self.linear_proj(x) # [batch_size, seq_length+1, 512]
        x = self.pos_encoder(x.permute(1,0,2)) # [seq_length+1, batch_size, 512]
        x = x.permute(1,0,2) # [batch_size, seq_length+1, 512]
        aspect_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device) # [batch_size, seq_length]
        attention_mask = torch.cat((aspect_mask, attention_mask), dim=1) # [batch_size, seq_length+1]
        x = self.encoder(x, src_key_padding_mask=attention_mask) # [batch_size, seq_length+1, 512]
        logits = self.clf(x.mean(dim=1)) # [batch_size, output_dim]
        return logits
    
class TransformerEncoderV1(nn.Module):
    """ Transformer encoder"""
    def __init__(self, input_dim, output_dim, num_layers, embedding_matrix, seq_length, num_aspect, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.aspect_embedding = nn.Embedding(num_aspect, 512)
        self.pos_encoder = PositionalEncoding(512, 0.1, seq_length+1)
        self.linear_proj = nn.Linear(input_dim, 512)
        
        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=num_layers)
        
        self.clf = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Dropout(p=drop),
            nn.Linear(1024, output_dim),
        )

    def forward(self, input_ids, aspect_id, attention_mask, **kwarg):
        x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        x = self.linear_proj(x) # [batch_size, seq_length, 512]
        aspect_x = self.aspect_embedding(aspect_id)  # [batch_size, 512]
        x = torch.cat((aspect_x.unsqueeze(1), x), dim=1) # [batch_size, seq_length+1, 512]
        x = self.pos_encoder(x.permute(1,0,2)) # [seq_length+1, batch_size, 512]
        x = x.permute(1,0,2) # [batch_size, seq_length+1, 512]
        aspect_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device) # [batch_size, seq_length]
        attention_mask = torch.cat((aspect_mask, attention_mask), dim=1) # [batch_size, seq_length+1]
        x = self.encoder(x, src_key_padding_mask=attention_mask) # [batch_size, seq_length+1, 512]
        logits = self.clf(x[:,0,:])
        return logits # [batch_size, output_dim]

class TransformerEncoderV2(nn.Module):
    """ Transformer encoder"""
    def __init__(self, input_dim, output_dim, num_layers, embedding_matrix, seq_length, num_aspect, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], 512)
        self.aspect_embedding = nn.Embedding(num_aspect, 512)
        self.pos_encoder = PositionalEncoding(512, 0.1, seq_length+1)
        
        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=num_layers)
        
        self.clf = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Dropout(p=drop),
            nn.Linear(1024, output_dim),
        )

    def forward(self, input_ids, aspect_id, attention_mask, **kwarg):
        x = self.embedding(input_ids) # [batch_size, seq_length, 512]
        aspect_x = self.aspect_embedding(aspect_id)  # [batch_size, 512]
        x = torch.cat((aspect_x.unsqueeze(1), x), dim=1) # [batch_size, seq_length+1, 512]
        x = self.pos_encoder(x.permute(1,0,2)) # [seq_length+1, batch_size, 512]
        x = x.permute(1,0,2) # [batch_size, seq_length+1, 512]
        aspect_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device) # [batch_size, seq_length]
        attention_mask = torch.cat((aspect_mask, attention_mask), dim=1) # [batch_size, seq_length+1]
        x = self.encoder(x, src_key_padding_mask=attention_mask) # [batch_size, seq_length+1, 512]
        logits = self.clf(x[:,0,:])
        return logits # [batch_size, output_dim]
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerEncoderV3(nn.Module):
    """ Transformer encoder"""
    def __init__(self, input_dim, output_dim, num_layers, embedding_matrix, seq_length, num_aspect, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.aspect_embedding = nn.Embedding(num_aspect, 512)
        self.pos_emb_layer = nn.Embedding(seq_length+1,
                                          512,
                                          padding_idx=0)
        self.linear_proj = nn.Linear(input_dim, 512)
        
        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=drop,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=num_layers)
        
        self.clf = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(p=drop),
            nn.Linear(128, output_dim),
        )

    def forward(self, input_ids, aspect_id, attention_mask, position_ids, **kwarg):
        x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.aspect_embedding(aspect_id)  # [batch_size, hidden_dim]
        pos_emb = self.pos_emb_layer(position_ids.type(torch.long))
        x = self.linear_proj(x) + pos_emb # [batch_size, seq_length, 512]
        x = torch.cat((aspect_x.unsqueeze(1), x), dim=1) # [batch_size, seq_length+1, 512]
        aspect_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device) 
        attention_mask = torch.cat((aspect_mask, attention_mask), dim=1) # [batch_size, seq_length+1]
        x = self.encoder(x, src_key_padding_mask=attention_mask) # [batch_size, seq_length+1, 512]
        logits = self.clf(x[:,0,:]) # [batch_size, output_dim]
        return logits