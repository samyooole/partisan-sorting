
import paramiko

import gzip
import shutil
import pandas as pd
import geopandas as gpd
from shapely import wkt
from tqdm import tqdm
from shapely import Point


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

server = 'login.rc.fas.harvard.edu'
user='samho'
password = '525Thousand!'
twofa= '468934'


def handler(title, instructions, fields):
    return [password]

transport = paramiko.Transport(server) 
transport.connect(username=user)
transport.auth_interactive_dumb(user)

sftp = paramiko.SFTPClient.from_transport(transport)

# set year here

year = '2022'

filepath = '/n/holylabs/LABS/cga/Lab/data/geo-tweets/cga-sbg/' + year + '/'
localpath = 'bigdata/geotweets/'

sftp.chdir(filepath)

# cleaning step: remove files which were accidentally unzipped into the folder (.csv) and double .gz (.gz.gz) fo;ders

lof = sftp.listdir()

lof_clean = [elem for elem in lof if not elem.endswith('.csv')]
lof_clean = [elem for elem in lof_clean if not elem.endswith('.gz.gz')]

# read the NC shapefile

ncsf = gpd.read_file('bigdata/NC shapefile/North_Carolina_State_and_County_Boundary_Polygons.shp')

ncsf = ncsf.to_crs(crs='epsg:4326')

NC_poly = ncsf.geometry.unary_union
NC_polydf = gpd.GeoDataFrame({'state': ['NC'], 'geometry': [NC_poly]}, crs='EPSG:4326')



for idx, batch_i in tqdm(enumerate(batch(lof_clean, n=100))):

    compildf = pd.DataFrame()
    for file in tqdm(batch_i):
        sftp.get(filepath + file, localpath + 'input.gz')
        
        with gzip.open(localpath+ 'input.gz', 'rb') as f_in:
            with open(localpath + 'file.tsv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        try:
            df = pd.read_csv(localpath + 'file.tsv', '\t', lineterminator='\n')
        except:
            continue


        df = df[['date','text', 'latitude', 'longitude']]
        df = df.dropna()
        df['geometry'] = df.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)

        df = gpd.GeoDataFrame(df, crs="EPSG:4326")

        localtweets = gpd.sjoin(df, NC_polydf)

        localtweets = localtweets.drop(['index_right', 'state'], axis=1)

        # file management

        compildf = pd.concat([compildf, localtweets])

        localpath+file

    compildf.to_file('bigdata/geotweets/container/' + year + "_" + str(idx) + '.shp', index=False)







    

