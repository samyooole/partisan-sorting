

"""
Get a megafile first
"""

import numpy as np
from tqdm import tqdm
import os
import pandas as pd

compildf = pd.DataFrame()


# just for now
lof = os.listdir('bigdata/NC_isol')


for file in tqdm(lof):
    print(file)
    df = pd.read_csv('bigdata/NC_isol/' + file)
    if file == '2011.csv':
        df['year'] = 2011
    compildf = compildf.append(df)

del df

"""
Get the mean isolation trends for Republicans and Democrats
"""


isolmeans = compildf.groupby(['Party', 'year']).mean()




import matplotlib.pyplot as plt

f = plt.figure()
f.set_figwidth(12)
f.set_figheight(8)

plt.plot(isolmeans.loc['REP'].index, isolmeans.loc['REP'].isol, label='Republican', color='red')
plt.plot(isolmeans.loc['DEM'].index, isolmeans.loc['DEM'].isol, label='Democrat', color='blue')

plt.xlabel("Year")  # Latex commands can be used
plt.ylabel("Average isolation")
plt.legend()
plt.savefig('summstats/repvsdem_NC.png')
plt.close()

"""
Show shifting distributions of isolation, between Democrats and Republicans.

use kernel density estimation for smoothness
"""

dataDict = {}

lof = os.listdir('bigdata/NC_isol')
lof = [lof[0] , lof[-1]]
from random import sample

for file in tqdm(lof):
    print(file)
    df = pd.read_csv('bigdata/NC_isol/' + file)
    if file == '2011.csv':
        df['year'] = 2011
    
    df = df[df['Party'] == 'DEM']
    df=df.reset_index(False)
    year = df['year'][0]
    isol = list(df['isol'])

    isol = sample(isol, 100000)

    dataDict.update({year:isol})

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dataDict.items() ]))


f = plt.figure()
f.set_figwidth(12)
f.set_figheight(8)

df.plot.kde()
plt.savefig('summstats/dem_kde.png')
plt.close()


"""
Show pre and post movers isolation scores (2010 vs 2022)
"""

from numpy import mean, std

before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')


# check for individuals with low isolation
"""
individuals with low isolation are more likely to move to places of higher isolation, whereas individuals with high isolation are more ok with moving to places of lower isolation
"""

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

lon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for no in lon:

    subbefore = before[ (before['isol'] >= no ) & (before['isol'] < no+0.1 )]


    before_new= subbefore.merge(after, on='ncid', how='left')

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] != before_new['lat_y']]
    mean_increase = mean(movers.isol_y - movers.isol_x)
    std_increase = std(movers.isol_y - movers.isol_x)

    plt.errorbar(str(round(no,2)) + ' < x < ' + str(round(no + 0.1, 2)), mean_increase, std_increase, fmt='ok', lw=3)


plt.xlabel('x = pre-move partisan isolation')
plt.ylabel('change in partisan isolation after move')

plt.savefig('summstats/migrators.png')
plt.close()



"""
Dems
"""

before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')

before = before[before['Party'] == 'DEM']
after = after[after['Party'] == 'DEM']

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

lon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for no in lon:

    subbefore = before[ (before['isol'] >= no ) & (before['isol'] < no+0.1 )]


    before_new= subbefore.merge(after, on='ncid', how='left')

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] != before_new['lat_y']]
    mean_increase = np.mean(movers.isol_y)
    std_increase = np.std(movers.isol_y)

    plt.errorbar(str(round(no,2)) + ' < x < ' + str(round(no + 0.1, 2)), mean_increase, std_increase, fmt='ok', lw=3, color='blue')


plt.xlabel('x = pre-move partisan isolation')
plt.ylabel('level of partisan isolation after move')
plt.title('Post-move partisan isolation for Democrats, 2010-2022')

plt.savefig('summstats/migrators_dems.png')
plt.close()


"""
Reps
"""

before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')

before = before[before['Party'] == 'REP']
after = after[after['Party'] == 'REP']

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

lon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for no in lon:

    subbefore = before[ (before['isol'] >= no ) & (before['isol'] < no+0.1 )]


    before_new= subbefore.merge(after, on='ncid', how='left')

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] != before_new['lat_y']]
    mean_increase = np.mean(movers.isol_y)
    std_increase = np.std(movers.isol_y)

    plt.errorbar(str(round(no,2)) + ' < x < ' + str(round(no + 0.1, 2)), mean_increase, std_increase, fmt='ok', lw=3, color='red')


plt.xlabel('x = pre-move partisan isolation')
plt.ylabel('level of partisan isolation after move')
plt.title('Post-move partisan isolation for Republicans, 2010-2022')

plt.savefig('summstats/migrators_reps.png')
plt.close()


"""
Dems (stayers)
"""

before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')

before = before[before['Party'] == 'DEM']
after = after[after['Party'] == 'DEM']

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

lon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for no in lon:

    subbefore = before[ (before['isol'] >= no ) & (before['isol'] < no+0.1 )]


    before_new= subbefore.merge(after, on='ncid', how='left')

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] == before_new['lat_y']]
    mean_increase = np.mean(movers.isol_y)
    std_increase = np.std(movers.isol_y)

    plt.errorbar(str(round(no,2)) + ' < x < ' + str(round(no + 0.1, 2)), mean_increase, std_increase, fmt='ok', lw=3, color='blue')


plt.xlabel('x = pre partisan isolation in 2010')
plt.ylabel('level of partisan isolation in 2022')
plt.title('Post-move partisan isolation for Democrats, 2010-2022')

plt.savefig('summstats/migrators_dems.png')
plt.close()


"""
Reps (stayers)
"""

before = pd.read_csv('bigdata/NC_isol/2010.csv')
after = pd.read_csv('bigdata/NC_isol/2022.csv')

before = before[before['Party'] == 'REP']
after = after[after['Party'] == 'REP']

before = before[['ncid', 'long', 'lat', 'isol']]
after = after[['ncid', 'long', 'lat', 'isol']]

lon = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for no in lon:

    subbefore = before[ (before['isol'] >= no ) & (before['isol'] < no+0.1 )]


    before_new= subbefore.merge(after, on='ncid', how='left')

    before_new = before_new[before_new['lat_y'].notna()]

    movers = before_new[before_new['lat_x'] == before_new['lat_y']]
    mean_increase = np.mean(movers.isol_y)
    std_increase = np.std(movers.isol_y)

    plt.errorbar(str(round(no,2)) + ' < x < ' + str(round(no + 0.1, 2)), mean_increase, std_increase, fmt='ok', lw=3, color='red')


plt.xlabel('x = pre-move partisan isolation')
plt.ylabel('level of partisan isolation after move')
plt.title('Post-move partisan isolation for Republicans, 2010-2022')

plt.savefig('summstats/migrators_reps.png')
plt.close()


"""

??? so people who move in are causing high isolation scores??
"""

inmigrants = [item for item in after.ncid if item not in before.ncid]

after[after['ncid'] == inmigrants]

"""
Analyze change in isolation scores between movers and stayers
"""

lof = os.listdir('bigdata/NC_isol')


before = pd.read_csv('bigdata/NC_isol/' + lof[0])
after = pd.read_csv('bigdata/NC_isol/' + lof[-1])

this_year = before['year'][0]
next_year = after['year'][1]

before = before[['Party', 'ncid', 'long', 'lat', 'isol']]
after = after[['Party','ncid', 'long', 'lat', 'isol']]

before_new= before.merge(after, on='ncid', how='left')

before_new = before_new.drop_duplicates('ncid') # the same person could have moved to two separate locations ??

before_new = before_new[before_new['lat_y'].notna()]

movers = before_new[before_new['lat_x'] != before_new['lat_y']]
stayed = before_new[before_new['lat_x'] == before_new['lat_y']]

movers['change_isol'] = movers.isol_y - movers.isol_x
stayed['change_isol'] = stayed.isol_y - stayed.isol_x

stayed = stayed[['ncid', 'isol_y']]
movers = movers[['ncid', 'isol_y']]
stayed.columns = ['ncid', 'isol']
movers.columns = ['ncid', 'isol']


###

# for now, try to analyze those

stayed['moved'] = 0

newdf = newdf.append(stayed)

newdf = newdf.reset_index(drop=True)

newdf['year_pair'] = str(this_year) + '-' + str(next_year)

compildf = compildf.append(newdf)



"""
DESCRIPTIVE REGRESSIONS

post-move isolation score ~ pre-move isolation score --> expect a negative relationship - reveals that partisan uncomfortability is only salient when you are in the minority, which makes sense
"""




"""
bound into like a city or something to analyze specific differences
"""


"""

how stable are party affiliations?
"""

newdf = before[['ncid', 'Party']].merge(after[['ncid','Party']], on='ncid',how='inner')

sum(newdf.Party_y != newdf.Party_x)






"""
Analyze change in isolation scores for movers: how do they relate to distance?
"""

lof = os.listdir('bigdata/NC_isol')


before = pd.read_csv('bigdata/NC_isol/' + lof[0])
after = pd.read_csv('bigdata/NC_isol/' + lof[-1])

this_year = before['year'][0]
next_year = after['year'][1]

before = before[['Party', 'ncid', 'long', 'lat', 'isol']]
after = after[['Party','ncid', 'long', 'lat', 'isol']]

before_new= before.merge(after, on='ncid', how='left')

before_new = before_new.drop_duplicates('ncid') # the same person could have moved to two separate locations ??

before_new = before_new[before_new['lat_y'].notna()]

movers = before_new[before_new['lat_x'] != before_new['lat_y']]
stayed = before_new[before_new['lat_x'] == before_new['lat_y']]

movers['change_isol'] = movers.isol_y - movers.isol_x
stayed['change_isol'] = stayed.isol_y - stayed.isol_x

stayed = stayed[['ncid', 'isol_y']]
movers = movers[['ncid', 'isol_y']]
stayed.columns = ['ncid', 'isol']
movers.columns = ['ncid', 'isol']


###

nonswitchers = movers[movers['Party_x'] == movers['Party_y']]

##
import statsmodels.formula.api as smf

movers['distance'] = movers.apply(lambda x: np.sqrt( (x['long_x'] - x['long_y'])**2 + (x['lat_x'] - x['lat_y'])**2   ), axis=1 )

model = smf.ols(formula= 'change_isol ~ isol_x + distance', data=movers)

results = model.fit()



nonswitchers['distance'] = nonswitchers.apply(lambda x: np.sqrt( (x['long_x'] - x['long_y'])**2 + (x['lat_x'] - x['lat_y'])**2   ), axis=1 )

model = smf.ols(formula= 'change_isol ~ isol_x + distance', data=nonswitchers)

results = model.fit()