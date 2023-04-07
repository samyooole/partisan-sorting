import pandas as pd
import geopandas as gpd
import os
from tqdm import tqdm

from pyogrio import read_dataframe

gpdf = gpd.GeoDataFrame()

lof = os.listdir('bigdata/geotweets/container/2013')

lof = [elem for elem in lof if elem.endswith('.shp')]

for file in tqdm(lof):
    mini_gpdf = read_dataframe('bigdata/geotweets/container/2013/' + file)

    gpdf = gpdf.append(mini_gpdf)

gpdf = gpdf.dropna()

gpdf = gpdf.reset_index(drop=True)

gts = pd.DataFrame(gpdf)

gts['text'] = gts.text.str.replace('[^\x00-\x7F]','', regex=True)
gts['text'] = gts.text.str.replace(r'http\S+', '', regex=True)
gts['text'] = gts.text.str.replace(r'&gt;', '', regex=True)
gts['text'] = gts.text.str.replace(r'&amp;', '', regex=True)

## 

from datasets import Dataset
ds = Dataset.from_pandas(gts[0:10])
ds = Dataset.from_pandas(gts)




## load model

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

device = "cpu"


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model_ft = AutoModelForSequenceClassification.from_pretrained('isPolitical_ft').to(device)

classifier = pipeline("text-classification", model=model_ft, tokenizer = tokenizer )


start = time()
classifier(ds['text'])
elapsed = time() - start
elapsed

from optimum.bettertransformer import BetterTransformer

model_ft = BetterTransformer.transform(model_ft,keep_original_model=True)
classifier = pipeline("text-classification", model=model_ft, tokenizer=tokenizer )

start = time()
classifier(ds['text'])
elapsed = time() - start
elapsed