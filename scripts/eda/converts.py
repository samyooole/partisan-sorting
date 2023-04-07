import pandas as pd
import os
from tqdm import tqdm

lof = os.listdir('bigdata/NC_analysisready')
lof.sort()
compildf = pd.DataFrame()

for idx, file in enumerate(tqdm(lof)):

    before = pd.read_csv('bigdata/NC_analysisready/'  + file)
    after = pd.read_csv('bigdata/NC_analysisready/'  + lof[idx+1])

    df = before.merge(after, how='inner', on='ncid')

    dems = before[before['Party'] == 'DEM']
    reps = before[before['Party'] == 'REP']
    unaf = before[before['Party'] == 'UNA']

    switchers = df[df['Party_x'] != df['Party_y']]

    demtorep = switchers[ (switchers['Party_x'] == 'DEM') & (switchers['Party_y'] == 'REP') ]

    dtr_conversion = len(demtorep) / len(dems)

    reptodem = switchers[ (switchers['Party_x'] == 'REP') & (switchers['Party_y'] == 'DEM') ]

    rtd_conversion = len(reptodem) / len(reps)

    demtounaf = switchers[ (switchers['Party_x'] == 'DEM') & (switchers['Party_y'] == 'UNA') ]
    unaftodem = switchers[ (switchers['Party_x'] == 'UNA') & (switchers['Party_y'] == 'DEM') ]

    reptounaf = switchers[ (switchers['Party_x'] == 'REP') & (switchers['Party_y'] == 'UNA') ]
    unaftorep = switchers[ (switchers['Party_x'] == 'UNA') & (switchers['Party_y'] == 'REP') ]

    dtu_separation = len(demtounaf) / len(dems)
    utd_participation = len(unaftodem) / len(unaf)

    rtu_separation = len(reptounaf) / len(reps)
    utr_participation = len(unaftorep) / len(unaf)

    dem_ns = (len(demtounaf) - len(unaftodem)) / len(dems)
    rep_ns = (len(reptounaf) - len(unaftorep)) / len(reps)

    newdf = pd.DataFrame(columns = ['year', 'dtr', 'rtd', 'dtu', 'utd', 'rtu', 'utr', 'dem_ns', 'rep_ns'])
    newdf.loc[0] = [after.year[0], dtr_conversion, rtd_conversion, dtu_separation, utd_participation, rtu_separation, utr_participation, dem_ns, rep_ns]
    
    compildf = compildf.append(newdf)


compildf.to_csv('summstats/conversion_separation_participation.csv')
    

import matplotlib.pyplot as plt

plt.plot(compildf.year, compildf.dtr * 100, color = 'red', label='Democrat to Republican conversion')
plt.plot(compildf.year, compildf.rtd * 100, color= 'blue', label = 'Republican to Democrat conversion')
plt.ylabel(r'% of partisans that convert to the other party')
plt.xlabel('Year')
plt.legend()
plt.show()

plt.plot(compildf.year, compildf.dem_ns * 100, color = 'blue', label='Democrat to Unaffiliated net separation rate')
plt.plot(compildf.year, compildf.rep_ns * 100, color= 'red', label = 'Republican to Unaffiliated net separation rate')
plt.ylabel(r'% of partisans that become Unaffiliated, net')
plt.xlabel('Year')
plt.legend()
plt.show()

# net separation rate

compildf['net_dem_separation'] = compildf.dtu - compildf.utd
compildf['net_rep_separation'] = compildf.rtu - compildf.utr


