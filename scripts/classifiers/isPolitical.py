import pandas as pd
import requests
import json



# get political tweets (huggingface)

df1 = pd.read_parquet('scripts/classifiers/poltweets.parquet')
df1['isPolitical'] = 1

poltweets = df1[['text', 'isPolitical']]

topictweets = pd.read_json('scripts/classifiers/topictweets.json', lines=True)

ohe = pd.get_dummies(topictweets.label_name.explode()).groupby(level=0).sum()

topictweets = pd.concat([topictweets, ohe], axis=1)

nonpol = topictweets[topictweets['news_&_social_concern'] == 0]


# clean nonpol tweets
nonpol['text'] =nonpol.text.str.replace(r'{@.*@}', '', regex=True) #remove tags

nonpol['text'] =nonpol.text.str.replace(r'{{.*}}', '', regex=True)

nonpol['text'] = nonpol.text.str.replace('[^\x00-\x7F]','', regex=True)

# reorg nonpol

nonpol['isPolitical'] = 0
nonpol = nonpol[['text', 'isPolitical']]

# clean poliitcal tweets

poltweets['text'] = poltweets.text.str.replace(r'http\S+', '', regex=True)

######## to formalize at some point: a function for cleaning tweets

# get more nonpolitical tweets! need about 75000 more to make a balanced training data

gts = pd.read_csv('geotweetsamples.csv')

gts['text'] = gts.text.str.replace('[^\x00-\x7F]','', regex=True)
gts['text'] = gts.text.str.replace(r'http\S+', '', regex=True)
gts['text'] = gts.text.str.replace(r'&gt;', '', regex=True)
gts['text'] = gts.text.str.replace(r'&amp;', '', regex=True)

from random import sample

sample_text = sample(list(gts.text), 75000)

sample_text = pd.Series(sample_text)

sample_text = sample_text.dropna()

# drop familiar political topics

sample_text = sample_text[~sample_text.str.contains('obama')]
sample_text = sample_text[~sample_text.str.contains('Obama')]
sample_text = sample_text[~sample_text.str.contains('abortion')]
sample_text = sample_text[~sample_text.str.contains('gun')]
sample_text = sample_text[~sample_text.str.contains('guns')]
sample_text = sample_text[~sample_text.str.contains('shooting')]
sample_text = sample_text[~sample_text.str.contains('healthcare')]
sample_text = sample_text[~sample_text.str.contains('immigration')]
sample_text = sample_text[~sample_text.str.contains('migrants')]
sample_text = sample_text[~sample_text.str.contains('democrat')]
sample_text = sample_text[~sample_text.str.contains('republican')]
sample_text = sample_text[~sample_text.str.contains('shutdown')]

sample_text = pd.DataFrame(sample_text)
sample_text.columns = ['text']
sample_text['isPolitical'] = 0

"""
join and start training a classifier
"""

from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import evaluate
import numpy as np

fullset = pd.concat([poltweets, nonpol, sample_text] ,axis =0)

fullset.columns = ['text', 'label']

ds = Dataset.from_pandas(fullset)

ds = ds.remove_columns('__index_level_0__')

ds = ds.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "isNotPolitical", 1: "isPolitical"}
label2id = {"isNotPolitical": 0, "isPolitical": 1}


"""
trainer
"""

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


"""
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=1, id2label=id2label, label2id=label2id
)
"""

model = AutoModelForSequenceClassification.from_pretrained('isPolitical_ft')

training_args = TrainingArguments(
    output_dir="isPolitical",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('isPolitical_ft')


from transformers import pipeline, AutoModel

model_ft = AutoModelForSequenceClassification.from_pretrained('isPolitical_ft')

classifier = pipeline("text-classification", model=model_ft, tokenizer = tokenizer )
classifier("going to sleep")