import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from shapely import Point

df = pd.read_csv('bigdata/NC_universe.csv')

import geopandas as gpd

gdf = gpd.read_file('bigdata/NC census blocks/2010_Census_Blocks.shp')

gdf = gdf.to_crs({'init': 'epsg:4326'}) 


#unit test



unitdf = df

unitdf['geometry'] = unitdf.apply(lambda x: Point(x['long'], x['lat']), axis=1)

unitdf = gpd.GeoDataFrame(unitdf, crs="epsg:4326")

universe_with_blocks = gpd.sjoin(unitdf, gdf)

undf = universe_with_blocks[['long', 'lat', 'geoid10']]

undf.columns = ['long', 'lat', 'block_geoid10']

undf = undf.drop_duplicates()

"""
take isol stuff and assign blocks
"""

# just for now
lof = os.listdir('bigdata/NC_isol')


for file in tqdm(lof):
    print(file)
    df = pd.read_csv('bigdata/NC_isol/' + file)

    df = df.merge(undf, on=['long', 'lat'], how='left')
    if file == '2011.csv':
        df['year'] = 2011
    
    df.to_csv('bigdata/NC_isol_blocks/' + file, index=False)
    
