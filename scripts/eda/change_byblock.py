

"""
First 
"""

import numpy as np
from tqdm import tqdm
import os
import pandas as pd



# just for now
lof = os.listdir('bigdata/NC_isol')


before = pd.read_csv('bigdata/NC_isol/' + lof[0], dtype={'block_geoid10': str})

before = before.groupby(['block_geoid10', 'Party'])['ncid'].count()

before = before.reset_index()

before = before.pivot(index='block_geoid10', columns='Party', values='ncid')

before = before.fillna(0)

before['REP'] = before['REP'] + before['LIB']

before['demshare'] = before['DEM'] / (before['DEM'] + before['REP'])

before['repshare'] = 1 - before['demshare']



after = pd.read_csv('bigdata/NC_isol/' + lof[-1], dtype={'block_geoid10': str})

after = after.groupby(['block_geoid10', 'Party'])['ncid'].count()

after = after.reset_index()

after = after.pivot(index='block_geoid10', columns='Party', values='ncid')

after = after.fillna(0)

after['REP'] = after['REP'] + after['LIB']
after['DEM'] = after['DEM'] + after['GRE']

after['demshare'] = after['DEM'] / (after['DEM'] + after['REP'])

after['repshare'] = 1 - after['demshare']


before['homogeneity'] = abs(before['demshare'] - 0.5) * 2
after['homogeneity'] = abs(after['demshare'] - 0.5) * 2


##

after = after.reset_index()
before = before.reset_index()

before = before[['block_geoid10', 'demshare', 'repshare']]
before.columns = ['block_geoid10', 'before_demshare', 'before_repshare']

after = after[['block_geoid10', 'demshare', 'repshare']]
after.columns = ['block_geoid10', 'after_demshare', 'after_repshare']

before = before.merge(after, on='block_geoid10', how='left')

before['change_in_repshare'] = (before['after_repshare'] - before['before_repshare'])



import matplotlib.pyplot as plt

plt.scatter(before['before_demshare'], before['change_in_repshare'])
plt.xlabel('Democrat share in census block, 2010')
plt.ylabel('Change in Republican share in census block, 2010-2022')
plt.show()



























"""

tract"""




# just for now
lof = os.listdir('bigdata/NC_isol')


before = pd.read_csv('bigdata/NC_isol/' + lof[0], dtype={'tract_geoid10': str})

before = before.groupby(['tract_geoid10', 'Party'])['ncid'].count()

before = before.reset_index()

before = before.pivot(index='tract_geoid10', columns='Party', values='ncid')

before = before.fillna(0)

before['REP'] = before['REP'] + before['LIB']

before['demshare'] = before['DEM'] / (before['DEM'] + before['REP'])

before['repshare'] = 1 - before['demshare']



after = pd.read_csv('bigdata/NC_isol/' + lof[-1], dtype={'tract_geoid10': str})

after = after.groupby(['tract_geoid10', 'Party'])['ncid'].count()

after = after.reset_index()

after = after.pivot(index='tract_geoid10', columns='Party', values='ncid')

after = after.fillna(0)

after['REP'] = after['REP'] + after['LIB']
after['DEM'] = after['DEM'] + after['GRE']

after['demshare'] = after['DEM'] / (after['DEM'] + after['REP'])

after['repshare'] = 1 - after['demshare']



##

after = after.reset_index()
before = before.reset_index()

before = before[['tract_geoid10', 'demshare', 'repshare']]
before.columns = ['tract_geoid10', 'before_demshare', 'before_repshare']

after = after[['tract_geoid10', 'demshare', 'repshare']]
after.columns = ['tract_geoid10', 'after_demshare', 'after_repshare']

before = before.merge(after, on='tract_geoid10', how='left')

before['change_in_repshare'] = (before['after_repshare'] - before['before_repshare'])



import matplotlib.pyplot as plt

plt.scatter(before['before_demshare'], before['change_in_repshare'])
plt.xlabel('Democrat share in census tract, 2010')
plt.ylabel('Change in Republican share in census tract, 2010-2022')
plt.show()