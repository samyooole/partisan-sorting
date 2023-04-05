import numpy as np
from tqdm import tqdm
import os
import pandas as pd

compildf = pd.DataFrame()


# just for now
lof = os.listdir('bigdata/NC_isol')


for idx,file in tqdm(enumerate(lof)):
    print(file)
    before = pd.read_csv('bigdata/NC_isol/' + file)
    after = pd.read_csv('bigdata/NC_isol/' + lof[idx+1])
    if file == '2011.csv':
        before['year'] = 2011

    if lof[idx+1] == '2011.csv':
        after['year'] = 2011

    this_year = before['year'][0]
    next_year = after['year'][1]

    before = before[['ncid', 'long', 'lat', 'isol']]
    after = after[['ncid', 'long', 'lat', 'isol']]

    before_new= before.merge(after, on='ncid', how='left')
    
    before_new = before_new.drop_duplicates('ncid') # the same person could have moved to two separate locations ??

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] != before_new['lat_y']]


    # get an anti join which.. apparently has not been natively implemented in pandas!
    outer= before.merge(after, on='ncid', how='outer', indicator=True)
    dead_or_moved = outer[(outer._merge=='left_only')]

    dead_or_moved = dead_or_moved.drop_duplicates('ncid')

    isol = dead_or_moved[['ncid', 'isol_x']]

    isol = isol.append(movers[['ncid', 'isol_x']])

    newdf = isol

    newdf.columns = ['ncid', 'isol']

    newdf['moved'] = 1

    stayed = before_new[before_new['lat_x'] == before_new['lat_y']]

    stayed = stayed[['ncid', 'isol_x']]
    stayed.columns = ['ncid', 'isol']

    stayed['moved'] = 0

    newdf = newdf.append(stayed)

    newdf = newdf.reset_index(drop=True)

    newdf['year_pair'] = str(this_year) + '-' + str(next_year)

    compildf = compildf.append(newdf)


    
# regress move probability against pre move isolation w yearpair FE




    
import statsmodels.formula.api as smf

model = smf.logit(formula= 'moved ~ isol + C(year_pair)', data=compildf)

results = model.fit()

for table in results.summary().tables:
    print(table.as_latex_tabular())


"""
Construct a panel - columns = [move: {0,1}, pre-move isolation \in [0,1], year-pair: {eg. 2010-2011}]
"""

loy = pd.unique(compildf.year)

for year in loy:
    this_year = year
    next_year = year + 1

    this_df = compildf[compildf['year'] == this_year]
    next_df = compildf[compildf['year'] == next_year]