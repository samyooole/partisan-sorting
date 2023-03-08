# libraries
import zipfile_deflate64 as zipfile
import pandas as pd
import os

# prepare files that you want to join

paths = ['bigdata/NC/VR_Snapshot_20221206.zip', 'bigdata/NC/VR_Snapshot_20211102.zip', 'bigdata/NC/VR_Snapshot_20201103.zip', 'bigdata/NC/VR_Snapshot_20191105.zip','bigdata/NC/VR_Snapshot_20181106.zip', 'bigdata/NC/VR_Snapshot_20171107.zip', 'bigdata/NC/VR_Snapshot_20161108.zip', 'bigdata/NC/VR_Snapshot_20151103.zip', 'bigdata/NC/VR_Snapshot_20141104.zip', 'bigdata/NC/VR_Snapshot_20131105.zip', 'bigdata/NC/VR_Snapshot_20121106.zip', 'bigdata/NC/VR_Snapshot_20120101.zip', 'bigdata/NC/VR_Snapshot_20101102.zip']

#paths = ['bigdata/NC/VR_Snapshot_20120101.zip', 'bigdata/NC/VR_Snapshot_20101102.zip']


for path in paths:

    path_to_zip_file = path


    
    """
    Unzip file
    """
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall('bigdata/unzip_env')


    """
    Load file (big)
    """



    unzipped_fn = os.listdir('bigdata/unzip_env')[0]

    df = pd.DataFrame()

    chunks = pd.read_csv('bigdata/unzip_env/' + unzipped_fn, '\t', usecols= ['snapshot_dt', 'county_desc', 'voter_reg_num', 'ncid', 'status_cd', 'house_num', 'street_dir', 'street_name', 'street_type_cd', 'street_dir', 'street_sufx_cd', 'res_city_desc', 'zip_code', 'party_cd'], chunksize=500000, encoding='utf-16')


    for chunk in chunks:

        # internally remove any 'removed' flagged people, because they could have moved out of state etc

        chunk['status_cd'] = chunk.status_cd.str.strip()
        chunk = chunk[chunk['status_cd'] == 'A']
        
        df = pd.concat([df, chunk], ignore_index=True)
        print(len(df))


    """
    Insert processing step that reduces your data in some way or form
    """

    df.to_csv('bigdata/NC_clean/'+ df.snapshot_dt[0]+'.csv', index=False)




    """
    Delete unzipped file because it takes up space
    """

    os.remove('bigdata/unzip_env/' + unzipped_fn)