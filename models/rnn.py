import torch
from torch import nn
import torch.nn.functional as F

class RNNV0(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) for all aspects"""

    def __init__(self, input_dim, hidden_dim, output_dim,num_aspects, num_layers, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.output_dim=output_dim
        self.num_aspects = num_aspects
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,bidirectional=True)
        self.decoder = nn.Linear(4*hidden_dim, output_dim)

    def forward(self, input_ids, aspect_id, **kwarg):
        x = self.embedding(input_ids) # [batch_size, seq_length, hidden_dim]
        aspect_x = self.embedding(aspect_id) #[batch_size, seq_length, hidden_dim]
        x = torch.cat(( aspect_x.unsqueeze(1),x), dim=1) 
        x=x.permute(1,0,2)
      
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(x)#[batch_size, 2*hidden_dim]
            #print(outputs.shape)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
            #print(encoding.shape)
        output = self.decoder(encoding)
            #print(output[i].shape)
        return output # [seq_len, batch_size, output_dim]

class RNNV1(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) for all aspects"""

    def __init__(self, input_dim, hidden_dim, output_dim,num_aspects, num_layers, embedding_matrix,seq_length):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.output_dim=output_dim
        self.num_aspects = num_aspects
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,bidirectional=True)
        self.decoder = nn.Linear(4*hidden_dim, output_dim)
        self.pos_encoder = PositionalEncoding(300, 0.1, seq_length+1)
    def forward(self, input_ids, aspect_id, **kwarg):
      # first dimension of the input required by the LSMN is the time dimension
        x = self.embedding(input_ids) # [seq_length, batch_size, hidden_dim]
        aspect_x = self.embedding(aspect_id)  # [batch_size, hidden_dim]
        x = torch.cat((aspect_x.unsqueeze(1), x), dim=1) # [batch_size,seq_length+1,  300]
        #print(x.shape)
        x = self.pos_encoder(x.permute(1,0,2)) # [seq_length+1, batch_size, 300]
        #print(x.shape)
        #x=x.permute(1,0,2)
        
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(x)#[batch_size, 2*hidden_dim]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs # [batch_size, output_dim]

class RNNV2(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) for all aspects"""

    def __init__(self, input_dim, hidden_dim, output_dim,num_aspects, num_layers, embedding_matrix,seq_length):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.output_dim=output_dim
        self.num_aspects = num_aspects
        self.encoder1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,bidirectional=True)
        self.decoder = nn.Linear(4*hidden_dim, output_dim)
        self.pos_emb_layer = nn.Embedding(seq_length+1,
                                          300,
                                          padding_idx=0)
        self.linear_proj = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU()
        )
    def forward(self, input_ids, aspect_id,attention_mask,position_ids, **kwarg):
      # first dimension of the input required by the LSMN is the time dimension
        x = self.embedding(input_ids) # [seq_length, batch_size, hidden_dim]
        aspect_x = self.embedding(aspect_id)  # [batch_size, hidden_dim]
        #print(position_ids.shape)
        pos_emb = self.pos_emb_layer(position_ids.type(torch.long))
        x = self.linear_proj(x) + pos_emb # [batch_size, seq_length, 300]
        x = torch.cat((aspect_x.unsqueeze(1), x), dim=1) # [batch_size, seq_length+1, 300]
        #print(x.shape)
        x=x.permute(1,0,2)
        #aspect_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device) 
        #attention_mask = torch.cat((aspect_mask, attention_mask), dim=1) # [batch_size, seq_length+1]
        self.encoder1.flatten_parameters()
        outputs, _ = self.encoder1(x)#[batch_size, 2*hidden_dim]
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        #print(outputs.shape)
        outs = self.decoder(encoding)
        return outs # [batch_size, output_dim]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        #print(d_model)
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