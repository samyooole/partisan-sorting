"""
With the universe of addresses, now join data from all North Carolinas and left join on addresses to get lat lon information
"""

import os
import pandas as pd

lof = os.listdir('bigdata/NC_clean')

universe = pd.read_csv('bigdata/NC_universe.csv')

# filter out unmatched addresses

universe = universe[universe['Status'] == 'M'] # we experience around an 18% loss in distinct voter addresses
universe['Address'] = universe['Address'].str.strip()


londf = pd.DataFrame()
for file in lof:
    print(file)
    
    df = pd.read_csv('bigdata/NC_clean/' + file)

    df['addr'] =  df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd 
    df['addr'] = df.addr.replace(r'\s+', ' ', regex=True) # remove double whitespaces: standardization
    df['addr'] = df['addr'].str.strip() # remove leading and trailing whitespaces: standardization
    df['city'] = df.res_city_desc
    df['state'] = 'NC'
    df['year'] = df.snapshot_dt.str[0:4]

    newdf = df[['year', 'county_desc', 'voter_reg_num', 'ncid', 'addr', 'city', 'state', 'party_cd']]
    newdf.columns = ['year', 'county', 'vrn', 'ncid', 'Address', 'City', 'State', 'Party']

    newdf = newdf.merge(universe, how='left', on = ['Address', 'City', 'State'])
    newdf = newdf.drop(['Unnamed: 0', 'Status'], axis=1)

    year = newdf.year[0]

    if file == '2012-01-01.csv':
        year = '2011'

    newdf.to_csv('bigdata/NC_analysisready/' + year + '.csv', index=False)