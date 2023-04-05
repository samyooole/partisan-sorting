import os
import pandas as pd

lof = os.listdir('bigdata/NC_analysisready')


"""
summary registration numbers
"""

compildf = pd.DataFrame()

for file in lof:
    print(file)
    df = pd.read_csv('bigdata/NC_analysisready/' + file)

    rep = sum(df['Party'] == 'REP')
    dem = sum(df['Party'] == 'DEM')
    unaf = sum(df['Party'] == 'UNA')
    others = len(df) - rep - dem - unaf
    total = len(df)
    year = df.year[0]

    appendf = [['Year', 'Republican', 'Democrat', 'Unaffiliated', 'Others', 'Total'], [year, rep, dem, unaf, others, total]]
    appendf = pd.DataFrame(appendf)
    appendf.columns = appendf.iloc[0]
    appendf = appendf.drop(0, axis=0)

    compildf = compildf.append(appendf)

print(compildf.to_latex(index=False))





"""
brown and enos 2021 isolation measure
"""

import scipy
import numpy as np
from tqdm import tqdm
import os
import time

compildf = pd.DataFrame()

a = 1 
c = 0.00001

# just for now
lof = os.listdir('bigdata/NC_analysisready')
lof = [lof[0], lof[4], lof[8], lof[12]]

for file in lof:
    print(file)
    df = pd.read_csv('bigdata/NC_analysisready/' + file)
    orig_length = len(df)
    df = df[df['long'].notna()]

    loss = (orig_length - len(df)) / orig_length

    print(loss)

    df = df.reset_index(drop=False)

    df['isol'] = np.nan


    
    tree = scipy.spatial.cKDTree(df[['long','lat']], leafsize=10) #LIFE SAVER THANK YOU STACKOVERFLOW
    # create tree, reduce, then query
    #df = df[(df['Party'] == 'DEM') | (df['Party'] == 'REP')]
    start = time.time()
    query_storage = [tree.query([long, lat], k = 101) for long, lat in zip(df['long'], df['lat']) ]
    end = time.time()
    print('querying took ' + str(end-start) + ' seconds')
    #df['query_storage'] = query_storage

    # each person's isolation
    start = time.time()
    df['isol'] = [sum([  1 / (query_storage[n][0][np.where(query_storage[n][1] == index_no)[0][0]] + c )**a for index_no in  (df.iloc[query_storage[n][1]]['Party'] == df.iloc[0]['Party'])[(df.iloc[query_storage[n][1]]['Party'] == df.iloc[0]['Party'])].index[1:]  ]) / sum([  1 / (query_storage[n][0][np.where(query_storage[n][1] == index_no)[0][0]] + c )**a for index_no in  df.iloc[query_storage[n][1]].index[1:]  ]) for n in range(0, len(df))]
    end= time.time()
    print('isolation arithmetic took ' + str(end-start) + ' seconds')

    df = df.drop(['index', 'Unnamed: 0'], axis=1)

    df.to_csv('bigdata/NC_isol/' + file, index=False)

    # denominator


## just for proposal: summstats from 2010 to 2022
import pandas as pd
import matplotlib.pyplot as plt


before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')

len(before[before['Party'] == 'UNA']) / len(before)
len(after[after['Party'] == 'UNA']) / len(after)

isolbefore = before[before['Party'] == 'DEM'].isol.mean()
isolafter = after[after['Party'] == 'DEM'].isol.mean()

plt.hist(after[after['Party'] == 'DEM'].isol)
plt.show()
    

# check for individuals who have same ncid but different addresses across different years

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

before_new= before.merge(after, on='ncid', how='left')

before_new = before_new[before_new['lat_y'].notna()]

movers = before_new[before_new['lat_x'] != before_new['lat_y']]
from numpy import mean
mean(movers.isol_y - movers.isol_x)


# check for individuals with low isolation
"""
individuals with low isolation are more likely to move to places of higher isolation, whereas individuals with high isolation are more ok with moving to places of lower isolation
"""

before = before[['ncid', 'long', 'lat', 'isol']]
before = before[before['isol'] < 0.3]
after = after[['ncid', 'long', 'lat', 'isol']]

before_new= before.merge(after, on='ncid', how='left')

before_new = before_new[before_new['lat_y'].notna()]

movers = before_new[before_new['lat_x'] != before_new['lat_y']]
from numpy import mean
mean(movers.isol_y - movers.isol_x)

















"""
notes for iteration

"""

# numerator: sum for republican neighbors
sum([  1 / (query_storage[0][0][np.where(query_storage[0][1] == index_no)[0][0]] + c )**a for index_no in  (unitdf.iloc[query_storage[0][1]]['Party'] == unitdf.iloc[0]['Party'])[(unitdf.iloc[query_storage[0][1]]['Party'] == unitdf.iloc[0]['Party'])].index[1:]  ])


# denominator: sum for all neighbors

sum([  1 / (query_storage[0][0][np.where(query_storage[0][1] == index_no)[0][0]] + c )**a for index_no in  unitdf.iloc[query_storage[0][1]].index[1:]  ])


#logic


for row in tqdm(df.itertuples()):

    attempt = row
    i = row.Index
    attempt_resi = [attempt.long, attempt.lat]


    
    #query 1001 and drop the first one which is probably itself or would only change the isolation number inconsequentially

    nearest_array = tree.query(attempt_resi, k = 1001) 
    
    # filter for same party peeps
    
    index_of_neighbors = nearest_array[1]
    neigh = df.iloc[index_of_neighbors]
    neigh = neigh.reset_index()

    my_party = attempt.Party

    fellow_pals = neigh[neigh['Party'] == my_party]


    nearest_df = pd.DataFrame(nearest_array).T
    nearest_df.columns = ['distance', 'index']
    nearest_df['index'] = nearest_df['index'].astype(int)

    fellow_pals = fellow_pals.merge(nearest_df, on='index')
    universe_of_pals = neigh.merge(nearest_df, on='index')

    # calculate numbers

    num = sum(1 / (fellow_pals.distance + c)**a )
    denom = sum(1 / (universe_of_pals.distance + c)**a )

    isol = num/denom

    df.at[i, 'isol'] = isol
    

"""
unit test for list comprehension approach
"""
#unitdf = df[0:1000]
#
# tree = scipy.spatial.cKDTree(unitdf[['long','lat']], leafsize=10) #LIFE SAVER THANK YOU STACKOVERFLOW
# create tree, reduce, then query
#unitdf = unitdf[(unitdf['Party'] == 'DEM') | (unitdf['Party'] == 'REP')]
start = time.time()
query_storage = [tree.query([long, lat], k = 101) for long, lat in zip(unitdf['long'], unitdf['lat']) ]
end = time.time()
print(end - start)


#unitdf['query_storage'] = query_storage

# each person's isolation
start = time.time()
unitdf['isol'] = [sum([  1 / (query_storage[n][0][np.where(query_storage[n][1] == index_no)[0][0]] + c )**a for index_no in  (unitdf.iloc[query_storage[n][1]]['Party'] == unitdf.iloc[0]['Party'])[(unitdf.iloc[query_storage[n][1]]['Party'] == unitdf.iloc[0]['Party'])].index[1:]  ]) / sum([  1 / (query_storage[n][0][np.where(query_storage[n][1] == index_no)[0][0]] + c )**a for index_no in  unitdf.iloc[query_storage[n][1]].index[1:]  ]) for n in range(0, len(unitdf))]
end= time.time()
print(end - start)


















##### try to reduce a loop lol

import scipy
import numpy as np
from tqdm import tqdm
import os
import time
import pandas as pd

compildf = pd.DataFrame()

a = 1 
c = 0.00001

# just for now
lof = os.listdir('bigdata/NC_analysisready')
lof = [lof[0], lof[4], lof[8], lof[12]]

# apply function for isolation
def calc_isolation(long, lat, party, tree, df, k=101, c=0.00001, a = 1):
    
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

    denom = sum(1 / (nearest_array[0] + c)**a )
    isol = num/denom
    return isol


for file in lof:
    print(file)
    df = pd.read_csv('bigdata/NC_analysisready/' + file)
    df=df[0:1000]
    orig_length = len(df)
    df = df[df['long'].notna()]

    loss = (orig_length - len(df)) / orig_length

    print(loss)

    df = df.reset_index(drop=True)

    df['isol'] = np.nan


    
    tree = scipy.spatial.cKDTree(df[['long','lat']], leafsize=100) #LIFE SAVER THANK YOU STACKOVERFLOW

    df['isol'] = df.apply(lambda x: calc_isolation(x['long'], x['lat'], df=df, tree = tree, k = 101), axis=1)

    %timeit df['isol'] = df.apply(lambda x: calc_isolation(long = x['long'], lat = x['lat'], party = x['Party'], df=df, tree = tree, k = 101), axis=1)

    %prun -l 4 df['isol'] = df.apply(lambda x: calc_isolation(x['long'], x['lat'], tree = tree, k = 101), axis=1)



import IPython
IPython.embed()