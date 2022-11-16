import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np

class Bert(nn.Module):
    """ Transformer encoder"""
    def __init__(self, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.d_model = self.bert.config.hidden_size
        self.clf = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(self.d_model, output_dim),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, **kwarg):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids,
        ).last_hidden_state
        logits = self.clf(outputs[:,0,:])
        return logits

class AttentiveAggregation(nn.Module):
    """ Aggregation based on a single aspect query """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.k_proj = self.clf = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
        )

    def forward(self, q, v, mask):
        '''
        Args:
            q: [batch_size, dim],
            v: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len]
        Return:   
            values: [batch_size, dim]
        '''
        k = self.k_proj(v) # [batch_size, seq, dim]
        attn_logits = torch.matmul(k, q.unsqueeze(-1)) # [batch_size, seq, dim] @ [batch_size, dim, 1] = [batch_size, seq, 1]
        attn_logits = attn_logits.squeeze(-1)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -np.inf)
        attention = F.softmax(attn_logits, dim=-1) # [batch_size, seq]
        values = torch.matmul(attention.unsqueeze(1), v) # [batch_size,1, seq] @ [batch_size, seq, dim] = [batch_size,1, dim]
        return values.squeeze(1)
    
class BertForABSC(nn.Module):
    """ Transformer encoder"""
    def __init__(self, output_dim, num_aspect, num_aspect_query=None):
        super().__init__()
        self.num_aspect_query = num_aspect_query
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.d_model = self.bert.config.hidden_size
        self.aspect_embeddings = nn.ModuleList()
        for _ in range(self.num_aspect_query):
            self.aspect_embeddings.append(nn.Embedding(num_aspect, self.d_model))
        self.agg = AttentiveAggregation(self.d_model)
        if (not self.num_aspect_query) or self.num_aspect_query == 1:
            self.clf = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(self.d_model, output_dim),
            )
        else:
            self.clf = nn.Sequential(
                nn.Linear(self.d_model*self.num_aspect_query, self.d_model),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(self.d_model, self.d_model),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(self.d_model, output_dim),
            )

    def forward(self, input_ids, attention_mask, token_type_ids, aspect_id, **kwarg):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids,
        ).last_hidden_state
        doc_emb = []
        for aspect_embdding in self.aspect_embeddings:
            aspect_queries = aspect_embdding(aspect_id) # [batch_size, dim]
            doc_emb.append(self.agg(aspect_queries, outputs, attention_mask))
        outputs = torch.cat(doc_emb, dim=-1) # [batch_size, dim*self.num_aspect_query]
        logits = self.clf(outputs)
        return logits