import pandas as pd
import os


"""
Idea: read and deduplicate at each step - get a universe of voter addresses from North Carolina, then geocode afterwards
"""

lof = os.listdir('bigdata/NC_clean')
state_name = 'North Carolina'

ss = pd.Series()
for file in lof:

    df = pd.read_csv('bigdata/NC_clean/' + file)

    df['addr'] = df.house_num.astype('str') + ' ' + df.street_dir + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd + ' ' + state_name + ' ' + df.zip_code.astype('str')

    ss = pd.concat([ss, df['addr']], axis=0)

    ss = pd.Series(pd.unique(ss))


# sanity check, strip extra whitespace

ss_stripped = ss.replace(r'\s+', ' ', regex=True)

ss_stripped = pd.Series(pd.unique(ss_stripped))

ss_stripped.to_csv('bigdata/NC_addresses.csv')

from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")

location = geolocator.geocode('220 N GURNEY ST NORTH CAROLINA 27215')

