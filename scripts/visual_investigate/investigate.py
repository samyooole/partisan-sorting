"""
do a visual investigation to sanity check the isolation function



"""

import scipy
import numpy as np
from tqdm import tqdm
import os
import time
import pandas as pd

# just for now
lof = os.listdir('bigdata/NC_analysisready')
#lof = [lof[0], lof[4], lof[8], lof[12]]

# apply function for isolation
def calc_isolation (long, lat, party, tree, df, k=101, c=0.0001, a = 1):
    
    #query 1001 and drop the first one which is probably itself or would only change the isolation number inconsequentially

    nearest_array = tree.query([long, lat], k = k) 
    
    # filter for same party peeps
    
    index_of_neighbors = nearest_array[1][1:]
    neigh = df.iloc[index_of_neighbors]
    #neigh = neigh.reset_index()

    my_party = party

    fellow_pals = neigh[neigh['Party'] == my_party]

    lol = []
    for index_no in fellow_pals.index:
        lol.append(np.where(nearest_array[1] == index_no)[0][0])
    
    fellow_distances = nearest_array[0][lol]
    num = sum(1 / (fellow_distances + c)**a )

    denom = sum(1 / (nearest_array[0][1:] + c)**a )
    isol = num/denom
    return isol


file = lof[-1]

df = pd.read_csv('bigdata/NC_analysisready/' + file)
#df=df[0:10000]
orig_length = len(df)
df = df[df['long'].notna()]

loss = (orig_length - len(df)) / orig_length

print(loss)

#df = df[(df['long'] > -84.1288299) & (df['long'] < -83.9940400)]
#df = df[(df['lat'] > 35.1577064) & (df['lat'] < 35.2190653)]

#df = df[(df['long'] > -79.809983) & (df['long'] < -79.7345157)]
#df = df[(df['lat'] > 36.0434061) & (df['lat'] < 36.0782571)]

# a nice example of a suburb going from slightly republican to dominantly republican

df = df[(df['long'] > -80.1907153) & (df['long'] < -80.1597057)]
df = df[(df['lat'] > 35.8671317) & (df['lat'] < 35.8813148)]

df = df.reset_index(drop=True)

df['isol'] = np.nan



tree = scipy.spatial.cKDTree(df[['long','lat']], leafsize=100) #LIFE SAVER THANK YOU STACKOVERFLOW

df['isol'] = df.apply(lambda x: calc_isolation(x['long'], x['lat'], party = x['Party'],df=df, tree = tree, k = 101), axis=1)

df.to_csv('sample_isol_2022.csv')



"""
"""

long = -84.052508
lat = 35.188104

