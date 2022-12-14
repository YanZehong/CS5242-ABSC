import os
import json
import pickle
import numpy as np
import pandas as pd
import math
import collections
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
        plt.hist([length_dict[0] + length_dict[1] + length_dict[2]], bins=range(0, max_seq_len, 10))
        plt.show()
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
            text = lines[key]['text']
            label = lines[key]['polarity']
            examples.append(InputExample(unique_id=unique_id, aspect=aspect, text=text, label=label))
        return examples  
    
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

def parse_dataset(data_dir, mode):
    """read data from csv file"""
    polar_idx={'negative': 0, 'neutral': 1, 'positive': 2}
    aspects_names = {"Value":"value", "Location": "location", "Service":"service", "Rooms": "room", "Cleanliness": "clean", "Sleep Quality": "sleep"}
    df = pd.read_csv(data_dir+ mode + '.csv')
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

def plot_ngram(df, aspect, k):
    # Splitting the corpus into positive and negative comments
    positive_comments = df[df[aspect] == 2]['review']
    negative_comments = df[df[aspect] == 1]['review']
    neutral_comments = df[df[aspect] == 0]['review']
    # Extracting the top k unigrams by sentiment
    trigrams_pos = ngrams_count(positive_comments, (3, 3), k)
    trigrams_neg = ngrams_count(negative_comments, (3, 3), k)
    trigrams_neu = ngrams_count(neutral_comments, (3, 3), k)

    # Joining everything in a python dictionary to make the plots easier
    ngram_dict_plot = {
        'Top Trigrams on Positive Comments': trigrams_pos,
        'Top Trigrams on Negative Comments': trigrams_neg,
        'Top Trigrams on Neutral Comments': trigrams_neu,
    }

    # Plotting the ngrams analysis
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 18))
    i, j = 0, 0
    colors = ['Blues_d', 'Reds_d', 'Greens_d']
    for title, ngram_data in ngram_dict_plot.items():
        # ax = axs[i, j]
        ax = axs[j]
        sns.barplot(x='count', y='ngram', data=ngram_data, ax=ax, palette=colors[j])
        
        # Customizing plots
        format_spines(ax, right_border=False)
        ax.set_title(title, size=14)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        # Incrementing the index
        j += 1
        # if j == 3:
        #     j = 0
        #     i += 1
    plt.tight_layout()
    plt.show()

def format_spines(ax, right_border=True):
    """docstring for format_spines:
    this function sets up borders from an axis and personalize colors
    input:
        ax: figure axis
        right_border: flag to determine if the right border will be visible or not"""
    
    # Setting up colors
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_color('#FFFFFF')
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')

def ngrams_count(corpus, ngram_range, n=-1, cached_stopwords=stopwords.words('portuguese')):
    """
    Args
    ----------
    corpus: text to be analysed [type: pd.DataFrame]
    ngram_range: type of n gram to be used on analysis [type: tuple]
    n: top limit of ngrams to be shown [type: int, default: -1]
    """
    
    # Using CountVectorizer to build a bag of words using the given corpus
    vectorizer = CountVectorizer(stop_words=cached_stopwords, ngram_range=ngram_range).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    total_list = words_freq[:n]
    
    # Returning a DataFrame with the ngrams count
    count_df = pd.DataFrame(total_list, columns=['ngram', 'count'])
    return count_df