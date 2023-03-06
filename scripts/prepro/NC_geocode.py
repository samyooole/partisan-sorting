import pandas as pd

df = pd.read_csv('bigdata/2016-11-08.csv')

df = df.drop(['Unnamed: 0', 'status_cd'],axis=1)

df['year'] = df.snapshot_dt.str[0:4]

df = df.drop(['snapshot_dt'],axis=1)

df['addr'] = df.house_num.astype('str') + ' ' + df.street_name + ' ' + df.street_type_cd + ' ' + df.street_sufx_cd

## import geolocation libraries

# Import the required library
from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")

location = geolocator.geocode("Hyderabad")

