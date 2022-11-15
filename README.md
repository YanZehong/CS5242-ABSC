# CS5242-ABSC
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.9-blue)](https://www.python.org/downloads/)

In this project, we focus on the aspect-based sentiment classification (ABSC). To be specific, the sentence “The restaurant was expensive, but the food was great” would be assigned with a negative polarity for the aspect PRICE, whereas the aspect FOOD has positive polarity. We implement MLP, CNNs, RNNs, ANNs*, and related improved versions on the dataset we collected on [Tripadvisor](https://www.tripadvisor.com.sg/).

## Table of Codes
- Data Collection: `DataScraper.ipynb`  
- EDA: `CS5242_GroupID26.ipynb`  
- DataModule: `CS5242_GroupID26.ipynb`  
    * ABSCDataset  
    * ABSCDatasetForBERT  
- ModelModule: `CS5242_GroupID26.ipynb`  
    * [MLP variants](./images/MLP.png): MLPs, MLP, MLPv2  
    * [CNN](./images/TextCNN.jpg): TextCNNv1, TextCNNv2, TextCNNv3 (mean/max)  
    * [RNN](./images/LSTM.png): RNNV0, RNNV1, RNNV2
    * [Transformer](./images/Transformer.png): TransformerEncoderV0, TransformerEncoderV1, TransformerEncoderV2, TransformerEncoderV3 
    * [BERT](./images/Bert.png): Bert, BertForABSC  

## Project Set Up

This is a list of all requirements used in this project.

```
git clone https://github.com/YanZehong/CS5242-ABSC.git
conda create -n cs5242 -y python=3.9
conda activate cs5242
# conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
cd CS5242-ABSC
mkdir checkpoints output
pip install jupyter
jupyter notebook
```

To install requirements, run `pip install -r requirements.txt`. Please ensure that you have met the prerequisites in [PyTorch](https://pytorch.org/) and install correspond version. 

### Download GloVe embedding
```
wget https://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip
```
Please unzip the txt file in the folder `dataset/`. For non-BERT-based models, GloVe pre-trained word vectors are required, please refer to data_utils.py for more detail. [GloVe](https://nlp.stanford.edu/projects/glove/)

### Runnig on Colab
Put the entire CS5242-ABSC folder into the root directory of google drive.
Put the following code in the first cell of Libraries and execute it.
```
!pip install matplotlib_inline
!pip install transformers
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import os
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
path = "/content/gdrive/MyDrive/CS5242-ABSC"
os.chdir(path)

#download glove embedding data.
if os.path.isfile("/content/gdrive/MyDrive/CS5242-ABSC/dataset/glove.42B.300d.zip") == False:
  !wget -P /content/gdrive/MyDrive/CS5242-ABSC/dataset/ https://nlp.stanford.edu/data/glove.42B.300d.zip
  print("download success.")
if os.path.isfile("/content/gdrive/MyDrive/CS5242-ABSC/dataset/glove.42B.300d.txt") == False:
  !unzip /content/gdrive/MyDrive/CS5242-ABSC/dataset/glove*.zip -d /content/gdrive/MyDrive/CS5242-ABSC/dataset/
  print("unzip success.")

!mkdir /content/gdrive/MyDrive/CS5242-ABSC/checkpoints
!mkdir /content/gdrive/MyDrive/CS5242-ABSC/output
```
Before 5.1, modify the following code to use colab's gpu.
```
devices= try_all_gpus()
devices = [devices[0]] # specify gpu id
```

## Framework Overview
![model-framework](https://github.com/YanZehong/CS5242-ABSC/blob/main/images/overview.png?raw=true)

## Milestones
![timelines](https://github.com/YanZehong/CS5242-ABSC/blob/main/images/milestones.png?raw=true)


## Note
Please check `images/` folder for models' figures. More details and result discussion can be found at `CS5242_GroupID26.ipynb`.

## Acknowledgement
We are grateful to NUS CS5242 for giving us such a valuable experience.  
Group26: Yan Zehong, Luo Jian, Zhao Yijing.  
[Project Github Link](https://github.com/YanZehong/CS5242-ABSC)  
<table>
  <tr>
    <td align="center"><a href="https://github.com/YanZehong"><img src="https://github.com/YanZehong.png" width="100px;" alt=""/><br /><sub><b>Yan Zehong</b></sub></a><br /><a href="https://github.com/YanZehong/CS5242-ABSC" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/LUOJIAN-GZ"><img src="https://avatars2.githubusercontent.com/u/36967362?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Luo Jian </b></sub></a><br /><a href="https://github.com/YanZehong/CS5242-ABSC" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/zhaoyijing24"><img src="https://avatars0.githubusercontent.com/u/37891032?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Zhao Yijing</b></sub></a><br /><a href="https://github.com/YanZehong/CS5242-ABSC" title="Code">💻</a></td>
  </tr>
</table>
