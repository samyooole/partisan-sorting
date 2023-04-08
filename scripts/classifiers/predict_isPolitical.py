"""
REALLY IMPORTANT STEPS TO SPEED UP INFERENCE
(1) use GPU
(2) use bettertransformers (goes from 0.03 sec / record --> 0.005 sec / record)
(3) use PIPELINE BATCHING (goes from the above --> 0.0007 sec / record) which can give you an entire year of North Carolina in 2-3 hours
"""


from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
from time import time
import pandas as pd
from datasets import Dataset
from optimum.bettertransformer import BetterTransformer
import os
from tqdm import tqdm
import geopandas as gpd


year = '2014'

from pyogrio import read_dataframe

gpdf = gpd.GeoDataFrame()

lof = os.listdir('bigdata/geotweets/container/' + str(year))

lof = [elem for elem in lof if elem.endswith('.shp')]

for file in tqdm(lof):
    mini_gpdf = read_dataframe('bigdata/geotweets/container/' + str(year) + '/' + file)

    gpdf = gpdf.append(mini_gpdf)

gpdf = gpdf.dropna()

gpdf = gpdf.reset_index(drop=True)

gts = pd.DataFrame(gpdf)

gts['text'] = gts.text.str.replace('[^\x00-\x7F]','', regex=True)
gts['text'] = gts.text.str.replace(r'http\S+', '', regex=True)
gts['text'] = gts.text.str.replace(r'&gt;', '', regex=True)
gts['text'] = gts.text.str.replace(r'&amp;', '', regex=True)

gts['text'] = gts.text.str.replace(r"^ +| +$", r"", regex=True)

gts = gts.dropna(subset='text')
gts = gts[gts['text'] != '']

## 

from datasets import Dataset
ds = Dataset.from_pandas(pd.DataFrame(gts['text']))


device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model_ft = AutoModelForSequenceClassification.from_pretrained('isPolitical_rd').to(device)

model_ft = BetterTransformer.transform(model_ft, keep_original_model=True)


classifier = pipeline("text-classification", model=model_ft, tokenizer = tokenizer, device=0 )

###
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

lol = []
for out in tqdm(classifier(KeyDataset(ds, 'text'), batch_size=256, truncation='only_first'), total = len(ds)):
    lol.append(out)

lol = pd.DataFrame(lol)

newgts = pd.concat([gts.reset_index(drop=True), lol], axis=1)

ispol = newgts[newgts['label'] == 'isPolitical']

ispol.to_csv('bigdata/political_geotweets/'  + str(year) + '.csv')