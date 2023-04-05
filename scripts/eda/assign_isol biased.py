
##### try to reduce a loop lol

"""
as of 21 march 2023, this is the most efficient algorithm (do one year of NC in around 12 minutes)
"""

import scipyS
import numpy as np
from tqdm import tqdm
import os
import time
import pandas as pd

# just for now
lof = os.listdir('bigdata/NC_analysisready')
#lof = [lof[0], lof[4], lof[8], lof[12]]

# apply function for isolation
def calc_isolation(long, lat, party, tree, dfarray, k=101, c=0.0001, a = 1):
    
    #query 1001 and drop the first one which is probably itself or would only change the isolation number inconsequentially

    nearest_array = tree.query([long, lat], k = k) 
    
    # filter for same party peeps
    
    index_of_neighbors = nearest_array[1][1:]
    neigh = dfarray[:,index_of_neighbors]
    #neigh = neigh.reset_index()

    my_party = party

    fellow_pals = neigh[1][np.where(neigh[0] == my_party)[0]]

    lol = []
    for index_no in fellow_pals:
        lol.append(np.where(nearest_array[1] == index_no)[0][0])
    
    fellow_distances = nearest_array[0][lol]
    num = sum(1 / (fellow_distances + c)**a )

    denom = sum(1 / (nearest_array[0][1:] + c)**a ) # we have to consistently exclude the first person
    isol = num/denom
    return isol





for file in tqdm(lof):
    print(file)
    df = pd.read_csv('bigdata/NC_analysisready/' + file)
    #df=df[0:10000]
    orig_length = len(df)
    df = df[df['long'].notna()]

    loss = (orig_length - len(df)) / orig_length

    print(loss)

    # get rid of all unaf voters (BIASED STEP)

    df = df[df['Party'] != 'UNA']

    df = df.reset_index(drop=True)

    df['isol'] = np.nan


    
    tree = scipy.spatial.cKDTree(df[['long','lat']], leafsize=100) #LIFE SAVER THANK YOU STACKOVERFLOW

    dfarray = np.array( [ df['Party'], df.index])

    df['isol'] = df.apply(lambda x: calc_isolation(x['long'], x['lat'], party = x['Party'],dfarray=dfarray, tree = tree, k = 101), axis=1)

    df = df.drop('Unnamed: 0', axis=1)

    df.to_csv('bigdata/NC_isol/' + file, index=False)

    





"""
test timings
"""
%timeit df.apply(lambda x: calc_isolation(long = x['long'], lat = x['lat'], party = x['Party'],dfarray=dfarray, tree = tree, k = 101, c= 0.00001, a = 1), axis=1)
%prun df.apply(lambda x: calc_isolation(long = x['long'], lat = x['lat'], party = x['Party'],dfarray=dfarray, tree = tree, k = 101, c= 0.00001, a = 1), axis=1)

























#########


from scripts.eda.isol import calc_isolation