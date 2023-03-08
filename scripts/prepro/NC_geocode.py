import pandas as pd
import os


"""
Idea: read and deduplicate at each step - get a //universe// of voter addresses from North Carolina, geocode in one shot, then left_join simply
"""

lof = os.listdir('bigdata/NC_clean')
state_name = 'NC'

ss = pd.DataFrame()
for file in lof:

    df = pd.read_csv('bigdata/NC_clean/' + file)

    #df['addr'] = df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd + ', ' + df.res_city_desc + ', ' + state_name + ', ' + df.zip_code.astype('str')

    df['addr'] =  df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd 
    df['addr'] = df.addr.replace(r'\s+', ' ', regex=True)
    df['city'] = df.res_city_desc
    df['zip_code'] = df.zip_code
    df['state'] = 'NC'

    ss = pd.concat([ss, df[['addr', 'city', 'state', 'zip_code']] ], axis=0)

    ss = ss.drop_duplicates()





# 
ss.columns = ['Address', 'City', 'State', 'Zip Code']
ss.to_csv('bigdata/NC_addresses_table.csv', index=False)
ss.head(100).to_csv('bigdata/NC_addresses_sample_table.csv', index=False)

# sanity check: same address/city/state but different zip_code

check_dups = ss[ss.duplicated(['Address', 'City', 'State'])]


#chunking

#specify number of rows in each chunk
n=5
df=ss

import numpy as np
#split DataFrame into chunks
df_split = np.array_split(df, n)

for idx, dataf in enumerate(df_split):
    dataf.to_csv('bigdata/NC_' + str(idx) + '.csv')

# here on out, use arcgis pro to geocode (chunks cannot be too lage)
"""
later: write an arcpy script to pipeline this < this should be automated
"""










"""
take addresses, read them and create a universe csv for latlong
"""


"""
Load content of a DBF file into a Pandas data frame.

The iter() is required because Pandas doesn't detect that the DBF
object is iterable.
"""
from dbfread import DBF
from pandas import DataFrame
import os

lof = os.listdir('bigdata/NC_dbf')

universe = pd.DataFrame()
for file in lof:
    print(file)
    dbf = DBF('bigdata/NC_dbf/' + file)
    frame = DataFrame(iter(dbf))

    print(frame)

    frame = frame[['X', 'Y', 'USER_Addre', 'USER_City', 'USER_State', 'Status']]

    frame.columns = ['long', 'lat', 'Address', 'City', 'State', 'Status']

    universe = pd.concat([universe, frame], axis=0)


universe.to_csv('bigdata/NC_universe.csv')


"""
Diagnostic: understanding what explains unmatched addresses
"""

lost_sheep = frame[frame['Status'] == 'U']
















"""
junk: to get single-line addresses
"""

lof = os.listdir('bigdata/NC_clean')
state_name = 'NC'

ss = pd.Series()
for file in lof:

    df = pd.read_csv('bigdata/NC_clean/' + file)

    df['addr'] = df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd + ', ' + df.res_city_desc + ', ' + state_name + ', ' + df.zip_code.astype('str')

    #df['addr'] =  df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd 
    df['addr'] = df.addr.replace(r'\s+', ' ', regex=True)
    #df['city'] = df.res_city_desc
    #df['state'] = 'NC'

    ss = pd.concat([ss, df.addr ], axis=0)

    ss = ss.drop_duplicates()



