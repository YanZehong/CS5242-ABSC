# for quick test only
import os
import math
import json
import pickle
import collections
import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
stopwords_list = stopwords.words('english')

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from data_utils import parse_dataset, build_tokenizer, build_glove_embeddings, ABSCProcessor, plot_ngram
from utils import Timer, Accumulator, Animator, try_all_gpus, accuracy, evaluate_loss_and_acc_gpu, test, test_on_checkpoint, read_and_calculate
from transformers import BertTokenizer, BertModel
from models import Bert, BertForABSC, MLPs, MLP, MLPv2, RNNV0, RNNV1, RNNV2, TextCNNv1_maxpool, TextCNNv2_maxpool, TextCNNv3_maxpool, TextCNNv1, TextCNNv2, TextCNNv3, TransformerEncoderV0, TransformerEncoderV1, TransformerEncoderV2

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class ABSCDataset(Dataset):
    """Build dataset and convert examples to features"""
    def __init__(self, name, examples, tokenizer):
        self.name = name
        self.examples = examples
        self.tokenizer = tokenizer
        self.convert_data_to_features()
    
    def convert_data_to_features(self):
        """convert examples to features"""
        label_map = {0: 0, 1: 1, 2: 2}
        aspect_map = {"value":0, "location":1, "service":2, "room":3, "clean":4, "sleep":5}

        output = []
        for (i, example) in enumerate(tqdm(self.examples)):
            input_ids = self.tokenizer.convert_tokens_to_ids(example.text)
            aspect_token_id = self.tokenizer.word2idx[example.aspect]
            aspect_id = aspect_map[example.aspect]
            label_id = label_map[example.label]
            attention_mask = np.zeros(input_ids.shape, dtype=bool)
            position_ids = np.zeros(input_ids.shape)
            for i in range(len(input_ids)):
                if input_ids[i] == 0: 
                    attention_mask[i] = True
                else:
                    position_ids[i] = i+1

            
            feature = {
                "input_ids": input_ids,
                "aspect_token_id": aspect_token_id,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "aspect_id": aspect_id,
                "label_id": label_id
            }
            output += [feature]
        self.data = output
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ABSCDatasetForBERT(Dataset):
    """Build dataset and convert examples to features"""
    def __init__(self, name, examples):
        self.name = name
        self.examples = examples
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.convert_data_to_features()
    
    def convert_data_to_features(self):
        """convert examples to features"""
        label_map = {0: 0, 1: 1, 2: 2}
        aspect_map = {"value":0, "location":1, "service":2, "room":3, "clean":4, "sleep":5}

        output = []
        for (i, example) in enumerate(tqdm(self.examples)):
            feature = {}
            inputs = self.tokenizer.encode_plus(
                text = example.aspect,
                text_pair = example.text,
                padding = 'max_length',
                max_length = 500,
                truncation = 'only_second',
                return_token_type_ids = True,
                return_tensors='pt',
            )
            
            input_ids = inputs["input_ids"][0]
            token_type_ids  = inputs["token_type_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            aspect_id = aspect_map[example.aspect]
            label_id = label_map[example.label]
            feature = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "aspect_id": aspect_id,
                "label_id": label_id
            }
            output += [feature]
        self.data = output
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def train_batch(net, batch, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs"""
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = batch[k].to(devices[0])
    else:
        batch = batch.to(devices[0])
    y = batch['label_id']
    net.train()
    trainer.zero_grad()
    pred = net(**batch)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    args.learning_rates.append(trainer.param_groups[0]["lr"]) # used to track scheduler
    return train_loss_sum, train_acc_sum

def show_acc_f1(dfs, names):
    output = collections.defaultdict(list)
    output['Model'] = names
    for i, df in enumerate(dfs):
        output['Acc'].append(df[names[i]+'_Acc'][0])
        output['F1'].append(df[names[i]+'_F1'][0])
    return pd.DataFrame(output)

def train(net, train_iter, valid_iter, num_epochs, optimizer, lr, wd, lr_period, lr_decay, devices):
    if optimizer == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif optimizer == 'adam':
        trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optimizer == 'adamw':
        trainer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        
    model_name = str(net.__class__.__name__)  
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    
    loss = nn.CrossEntropyLoss(reduction="none")
    timer = Timer()
    num_batches = len(train_iter)
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend += ['valid loss', 'valid acc']
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend, figsize=(7, 5))
    
    if args.monitor == 'loss':
        monitor_val = math.inf
    else:
        monitor_val = -math.inf
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(4)
        for i, batch in enumerate(train_iter):
            timer.start()
            labels = batch['label_id']
            output = train_batch(net, batch, loss, trainer, devices)
            metric.add(output[0], output[1], labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2], None, None))
        if valid_iter is not None:
            valid_loss, valid_acc = evaluate_loss_and_acc_gpu(net, valid_iter, loss, devices[0])
            animator.add(epoch + 1, (None, None, valid_loss, valid_acc))
            if args.monitor == 'loss':
                if valid_loss < monitor_val:
                    filename = model_name + f"-epoch{epoch}-val_loss{valid_loss :.2f}.pt"
                    best_model_state = copy.deepcopy(net.state_dict())
                    monitor_val = valid_loss
            else:
                if valid_acc > monitor_val:
                    filename = model_name + f"-epoch{epoch}-val_acc{valid_acc :.2f}.pt"
                    best_model_state = copy.deepcopy(net.state_dict())
                    monitor_val = valid_acc
        scheduler.step()
        
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    args.pt_path = os.path.join("checkpoints", filename)
    torch.save(best_model_state, args.pt_path)
    animator.save("images", model_name + f"-train_loss{metric[0] / metric[2]:.2f}.png")
    

def train_transformer(net, train_iter, valid_iter, num_epochs, optimizer, lr, wd, num_warmup_steps, devices):
    if optimizer == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    elif optimizer == 'adam':
        trainer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optimizer == 'adamw':
        trainer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        
    model_name = str(net.__class__.__name__)
    scheduler = get_cosine_schedule_with_warmup(trainer, num_warmup_steps=num_warmup_steps, num_training_steps=num_epochs*len(train_iter))
    
    loss = nn.CrossEntropyLoss(reduction="none")
    timer = Timer()
    num_batches = len(train_iter)
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend += ['valid loss', 'valid acc']
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], legend=legend, figsize=(7, 5))
    
    if args.monitor == 'loss':
        monitor_val = math.inf
    else:
        monitor_val = -math.inf
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        net.train()
        metric = Accumulator(4)
        for i, batch in enumerate(train_iter):
            timer.start()
            labels = batch['label_id']
            output = train_batch(net, batch, loss, trainer, devices)
            scheduler.step()
            metric.add(output[0], output[1], labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2], None, None))
            
            
        if valid_iter is not None:
            valid_loss, valid_acc = evaluate_loss_and_acc_gpu(net, valid_iter, loss, devices[0])
            animator.add(epoch + 1, (None, None, valid_loss, valid_acc))
            if args.monitor == 'loss':
                if valid_loss < monitor_val:
                    filename = model_name + f"-epoch{epoch}-val_loss{valid_loss :.2f}.pt"
                    best_model_state = copy.deepcopy(net.state_dict())
                    monitor_val = valid_loss
            else:
                if valid_acc > monitor_val:
                    filename = model_name + f"-epoch{epoch}-val_acc{valid_acc :.2f}.pt"
                    best_model_state = copy.deepcopy(net.state_dict())
                    monitor_val = valid_acc
        
        
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
    args.pt_path = os.path.join("checkpoints", filename)
    torch.save(best_model_state, args.pt_path)
    animator.save("images", model_name + f"-train_loss{metric[0] / metric[2]:.2f}.png")



class Args(object):
    def __init__(self):
        super().__init__()
        self.optimizer = 'adam'
        self.batch_size = 16
        self.num_epochs = 30
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.lr_period = 4
        self.lr_decay = 0.8
        self.dropout = 0.1
        self.embedding_dim = 300
        self.hidden_dim = 300
        self.num_class = 3
        self.num_aspect = 6
        self.max_seq_len = 500
        self.num_warmup_steps = 200
        self.pt_path = ''
        self.learning_rates = []
        self.monitor = 'loss'

args = Args()

def main():
    pass
    devices= try_all_gpus()
    devices = [devices[0]] # specify gpu id

    print("Processing the Dataset")
    data_dir="dataset"
    data_name = "TripAdvisor"
    datset_path = os.path.join(data_dir, data_name)
    processor = ABSCProcessor(data_dir, data_name)
    aspects = processor.get_aspects()
    num_classes = len(processor.get_labels())
    train_examples = processor.get_examples("train")
    dev_examples = processor.get_examples("dev")
    test_examples = processor.get_examples("test")

    max_seq_len = 500
    tokenizer = build_tokenizer(all_examples = [train_examples, dev_examples, test_examples], 
                                aspects = aspects, 
                                max_seq_len = max_seq_len,
                                dat_fname = datset_path+'/tokenizer_{0}.dat'.format(max_seq_len))

    embedding_matrix = build_glove_embeddings(
                    word2idx = tokenizer.word2idx,
                    dat_fname = datset_path+'/glove_300d_embeddings.dat')

    train_data = ABSCDataset(data_name, train_examples, tokenizer)
    dev_data = ABSCDataset(data_name, dev_examples, tokenizer)
    test_data = ABSCDataset(data_name, test_examples, tokenizer)

    train_iter = DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle=True)
    dev_iter = DataLoader(dataset=dev_data, batch_size = args.batch_size, shuffle=False)
    test_iter = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net = MLPv2(args.embedding_dim, args.hidden_dim, args.num_class, 2, embedding_matrix)
    test_on_checkpoint(net, "./checkpoints/MLPv2-epoch28-val_loss0.53.pt", test_iter, "test_output", device=devices[0])

    args.lr=1e-4
    net = RNNV0(args.embedding_dim, args.hidden_dim, args.num_class, args.num_aspect,2, embedding_matrix)
    test_on_checkpoint(net, "./checkpoints/RNNV0-epoch6-val_loss0.51.pt", test_iter, "test_output", device=devices[0])
    
    embed_size=300
    kernel_sizes=[3,4,5]
    num_channels=[300,300,3]
    net = TextCNNv2_maxpool(embed_size,kernel_sizes,num_channels,embedding_matrix)
    test_on_checkpoint(net, "./checkpoints/TextCNNv2_maxpool-epoch11-val_loss0.49.pt", test_iter, "test_output", device=devices[0])
    

    args.lr = 5e-5
    args.num_epochs=10
    net = TransformerEncoderV0(args.embedding_dim, args.num_class, 6, embedding_matrix, args.max_seq_len, 0.3)
    test_on_checkpoint(net, "./checkpoints/TransformerEncoderV0-epoch9-val_loss0.59.pt", test_iter, "test_output", device=devices[0])
    

    print("Processing the Dataset for BERT")
    data_dir="dataset"
    data_name = "TripAdvisor"
    datset_path = os.path.join(data_dir, data_name)
    processor = ABSCProcessor(data_dir, data_name)
    aspects = processor.get_aspects()
    num_classes = len(processor.get_labels())
    train_examples = processor.get_examples("train")
    dev_examples = processor.get_examples("dev")
    test_examples = processor.get_examples("test")
    train_data = ABSCDatasetForBERT(data_name, train_examples)
    dev_data = ABSCDatasetForBERT(data_name, dev_examples)
    test_data = ABSCDatasetForBERT(data_name, test_examples)
    args.lr = 3e-5
    args.num_epochs=5
    args.batch_size=16
    args.monitor = 'accuracy'
    train_iter = DataLoader(dataset=train_data, batch_size = args.batch_size, shuffle=True)
    dev_iter = DataLoader(dataset=dev_data, batch_size = args.batch_size, shuffle=False)
    test_iter = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net = Bert(args.num_class)
    test_on_checkpoint(net, "./checkpoints/Bert-epoch1-val_acc0.83.pt", test_iter, "test_output", device=devices[0])
    net = BertForABSC(args.num_class, args.num_aspect, 3)
    test_on_checkpoint(net, "./checkpoints/BertForABSCv1-epoch2-val_acc0.82.pt", test_iter, "test_output", device=devices[0])

    aspects = ["value", "location", "service", "rooms", "cleanliness", "sleep quality"]
    model_names = ["MLPs", "MLP", "MLPv2"]
    df_mlpv2 = read_and_calculate(aspects, ["./test_output/MLPv2_predictions.json"], model_names[2])

    model_names = ["RNNV0", "RNNV1", "RNNV2"]
    df_rnn0 = read_and_calculate(aspects, ["./test_output/RNNV0_predictions.json"], model_names[0])
    model_names = ["TextCNNv1_max", "TextCNNv2_max", "TextCNNv3_max", "TextCNNv2_mean"]
    df_cnn2 = read_and_calculate(aspects, ["./test_output/TextCNNv2_maxpool_predictions.json"], model_names[1])
    
    model_names = ["TransEncV0", "TransEncV1", "TransEncV2"]
    df_t0 = read_and_calculate(aspects, ["./test_output/TransformerEncoderV0_predictions.json"], model_names[0])
    
    model_names = ["BERT", "BertForABSC-1", "BertForABSC-2", "BertForABSC-3", "BertForABSC-4"]
    df_b0 = read_and_calculate(aspects, ["./test_output/Bert_predictions.json"], model_names[0])
    df_b3 = read_and_calculate(aspects, ["./test_output/BertForABSC_predictions.json"], model_names[3])
    

    cols = ["Aspects", "MLP", "RNN", "TextCNN", "Transformer", "BERT", "BertForABSC"]
    df_acc = pd.concat([df_mlpv2["Aspects"], df_mlpv2["MLPv2_Acc"], df_rnn0["RNNV0_Acc"], df_cnn2["TextCNNv2_max_Acc"],
                    df_t0["TransEncV0_Acc"], df_b0["BERT_Acc"], df_b3["BertForABSC-3_Acc"]], axis=1)
    df_acc.columns = cols
    print(df_acc)
    # df_acc.to_csv('results.csv', index = False)
if __name__ == '__main__':
    main()