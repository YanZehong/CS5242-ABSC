import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

class Tokenizer(object):
    """Construct a tokenizer."""
    def __init__(self, max_seq_len=math.inf, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text): # build word-id mapping
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def convert_tokens_to_ids(self, text, padding='post', truncating='post'): # convert input into ids
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
def build_tokenizer(all_examples, aspects, max_seq_len, dat_fname):
    """Build tokenizer (word-id pair) for dataset"""
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        length_dict = collections.defaultdict(list)
        corpus = ""
        for i, examples in enumerate(all_examples):
            for example in examples:
                words = nltk.word_tokenize(example.text)
                length_dict[i].append(len(words))
                for word in words:
                    corpus += (word + " ")
        for aspect in aspects:
            corpus += (aspect + " ")

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(corpus)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
        
        plt.figure()
        plt.hist([length_dict[0] + length_dict[1] + length_dict[2]], bins=range(0, 100, 10))
        print(f'max_length = {max(max(length_dict[0]), max(length_dict[1]), max(length_dict[2]))}')
        print(f'avg_length = {(sum(length_dict[0])+sum(length_dict[1])+sum(length_dict[2]))/(len(length_dict[0])+len(length_dict[1])+len(length_dict[2])) :.2f}')
    return tokenizer

def build_glove_embeddings(word2idx, dat_fname):
    """Build vocabulary for dataset"""
    if os.path.exists(dat_fname):
        print('Loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('Loading word vectors')
        embedding_matrix = np.zeros((len(word2idx) + 2, 300))  # idx 0 and len(word2idx)+1 are all-zeros
        embedding_dict={}
        with open('dataset/glove.42B.300d.txt','r') as f:
            for line in f:
                values=line.split()
                word = values[0]
                vectors=np.asarray(values[1:],'float32')
                embedding_dict[word]=vectors
                
        print('Building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = embedding_dict.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    """Pad or/and truncate the input sequence"""
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, unique_id, aspect, text, label):
        """Constructs a InputExample.
        Args:
            unique_id (string): 
                Unique id for the example.
            aspect (string): 
                Aspect category name.
            text (string): 
                Text or text span.
            label (string): 
                The label of the example. 
        Returns: 
        
        """
        self.unique_id = unique_id
        self.aspect = aspect
        self.text = text
        self.label = label

        
class ABSCProcessor(object):
    """Processor for the Aspect-based Sentiment Classification."""
    def __init__(self, data_dir, name):
        self.data_path = os.path.join(data_dir, name)
        self.name = name
        
    def get_examples(self, mode):
        """Read data from json file.
        Args:
            data_dir (string):
                Path.
            mode (string): 
                Indicate train/dev/test.
        Returns: 
            examples for different modes.
        """
        return self._create_examples(self._read_json(os.path.join(self.data_path, mode+".json")))
    
    def get_aspects(self):
        """Return aspects for specific dataset"""
        if self.name == "SemEval14":
            return ["food", "service", "price", "ambience", "miscellaneous"]
        elif self.name == "TripAdvisor":
            return ["value", "location", "service", "room", "clean", "sleep"]
        
    def get_labels(self):
        """Return labels for specific dataset"""
        return ["negative", "neutral", "positive"]
    
    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, key) in enumerate(lines):
            unique_id = key
            aspect = lines[key]['aspect']
            if self.name == "SemEval14":
                text = lines[key]['sentence']
            elif self.name == "TripAdvisor":
                text = lines[key]['text']
            label = lines[key]['polarity']
            examples.append(InputExample(unique_id=unique_id, aspect=aspect, text=text, label=label))
        return examples  
    
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

class ABSCDataset(Dataset):
    """Build dataset and convert examples to features"""
    def __init__(self, name, examples, tokenizer):
        self.name = name
        self.examples = examples
        self.tokenizer = tokenizer
        self.convert_data_to_features()
    
    def convert_data_to_features(self):
        """convert examples to features"""
        if self.name == "SemEval14":
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            aspect_map = {"food": 0,"service": 1, "price": 2, "ambience": 3, "miscellaneous": 4}
        elif self.name == "TripAdvisor":
            label_map = {0: 0, 1: 1, 2: 2}
            aspect_map = {"value":0, "location":1, "service":2, "room":3, "clean":4, "sleep":5}

        output = []
        for (i, example) in enumerate(tqdm(self.examples)):
            input_ids = self.tokenizer.convert_tokens_to_ids(example.text)
            aspect_token_id = self.tokenizer.word2idx[example.aspect]
            aspect_id = aspect_map[example.aspect]
            label_id = label_map[example.label]
            
            feature = {
                "input_ids": input_ids,
                "aspect_token_id": aspect_token_id,
                "aspect_id": aspect_id,
                "label_id": label_id
            }
            output += [feature]
        self.data = output
        
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def parse_Dataset(data_dir, mode):
    """read data from csv file"""
    df = pd.read_csv(os.path.join(data_dir, mode + '.csv'))
    corpus=[]
    label_cnt= np.zeros(len(polar_idx))
    for i, row in df.iterrows():
        text_id = row["id"]
        text = row["review"]
        for j, aspect in enumerate(aspects_names.keys()):
            label = int(row[aspect])
            if label == -1: continue
            corpus += [{"id": str(text_id) + '_' + str(j), "text": text, "aspect": aspects_names[aspect], "polarity": label}]
            label_cnt[label] += 1

    print(f"distribution of [negative, neutral, positive]: {label_cnt}")
    print(f"ratio: [{label_cnt[0]/sum(label_cnt)*100 :.1f}%, {label_cnt[1]/sum(label_cnt)*100 :.1f}%, {label_cnt[2]/sum(label_cnt)*100 :.1f}%]")
    print(f"#examples: {len(corpus)}")
    return corpus

def load_TripAdvisor():
    polar_idx={'negative': 0, 'neutral': 1, 'positive': 2}
    aspects_names = {"Value":"value", "Location": "location", "Service":"service", "Rooms": "room", "Cleanliness": "clean", "Sleep Quality": "sleep"}

    print("Load traning corpus:")
    train_corpus=parse_Dataset('dataset/TripAdvisor', 'train')
    with open("dataset/TripAdvisor/train.json", "w") as fw:
        json.dump({line["id"]: line for line in train_corpus}, fw, sort_keys=True, indent=4)
        
    print("Load development corpus:")
    dev_corpus=parse_Dataset("dataset/TripAdvisor", "dev")
    with open("dataset/TripAdvisor/dev.json", "w") as fw:
        json.dump({line["id"]: line for line in dev_corpus}, fw, sort_keys=True, indent=4)
        
    print("Load testing corpus:")
    test_corpus=parse_Dataset("dataset/TripAdvisor", "test")
    with open("dataset/TripAdvisor/test.json", "w") as fw:
        json.dump({line["id"]: line for line in test_corpus}, fw, sort_keys=True, indent=4)

def parse_SemEval14(fn):
    """read data from xml file"""
    root=ET.parse(fn).getroot()
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        opins=set()
        for opin in sent.iter('aspectCategory'):
            if opin.attrib['category']!="NULL":
                if opin.attrib['category'] in aspects_names.keys() and opin.attrib['polarity'] in polar_idx:
                    opins.add((aspects_names[opin.attrib['category']], opin.attrib['polarity']))
        for idx, opin in enumerate(opins):
            opin_cnt[polar_idx[opin[-1]]]+=1
            corpus.append({"id": sent.attrib['id']+"_"+str(idx), "sentence": sent.find('text').text, "aspect": opin[0], "polarity": opin[-1]})
    print(f"distribution of [negative, neutral, positive]: {opin_cnt}")
    print(f"#examples: {len(corpus)}")
    return corpus

def load_SemEval14(): 
    valid_split=150
    polar_idx={'negative': 0, 'neutral': 1, 'positive': 2}
    aspects_names = {"food": "food","service": "service", "price": "price", "ambience": "ambience", "anecdotes/miscellaneous": "miscellaneous"}
    print("Load traning corpus:")
    train_corpus=parse_SemEval14('dataset/SemEval14/Restaurants_Train_v2.xml')
    with open("dataset/SemEval14/train.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[:-valid_split] }, fw, sort_keys=True, indent=4)
    with open("dataset/SemEval14/dev.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in train_corpus[-valid_split:] }, fw, sort_keys=True, indent=4)
    print("Load testing corpus:")
    test_corpus=parse_SemEval14('dataset/SemEval14/Restaurants_Test_Gold.xml')
    with open("dataset/SemEval14/test.json", "w") as fw:
        json.dump({rec["id"]: rec for rec in test_corpus}, fw, sort_keys=True, indent=4)

