import pandas as pd
import os

dirname = 'bigdata/NC_clean'

lof = os.listdir(dirname)

df=pd.DataFrame()
for file in lof:

    little_df = pd.read_csv(dirname + '/'+ file)
    df.concat(little_df)