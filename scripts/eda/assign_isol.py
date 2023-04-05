
##### try to reduce a loop lol



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

    denom = sum(1 / (nearest_array[0][1:] + c)**a ) # we have to consistently exclude the first person
    isol = num/denom
    return isol




import IPython
IPython.embed()

%load_ext Cython

%%cython
def calc_isolation_c(long: cython.float, lat: cython.float, party: cython.char, tree, df, k=101, c=0.00001, a = 1):
    nearest_array = tree.query([long, lat], k = k) 
    """
    index_of_neighbors = nearest_array[1][1:]
    neigh = df.iloc[index_of_neighbors]

    my_party = party

    fellow_pals = neigh[neigh['Party'] == my_party]

    lol = []
    for index_no in fellow_pals.index:
        lol.append(np.where(nearest_array[1] == index_no)[0][0])
    
    fellow_distances = nearest_array[0][lol]
    num = sum(1 / (fellow_distances + c)**a )

    denom = sum(1 / (nearest_array[0] + c)**a )
    isol = num/denom
    return isol
    """


## WHAT???

for file in tqdm(lof):
    print(file)
    df = pd.read_csv('bigdata/NC_analysisready/' + file)
    #df=df[0:10000]
    orig_length = len(df)
    df = df[df['long'].notna()]

    loss = (orig_length - len(df)) / orig_length

    print(loss)

    df = df.reset_index(drop=True)

    df['isol'] = np.nan


    
    tree = scipy.spatial.cKDTree(df[['long','lat']], leafsize=100) #LIFE SAVER THANK YOU STACKOVERFLOW

    df['isol'] = df.apply(lambda x: calc_isolation(x['long'], x['lat'], party = x['Party'],df=df, tree = tree, k = 101), axis=1)

    df = df.drop('Unnamed: 0', axis=1)

    df.to_csv('bigdata/NC_isol/' + file, index=False)

    



