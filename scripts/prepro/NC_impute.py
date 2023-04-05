import pandas as pd

"""
Important note: voter history data only goes back to 2014 essentially, so we need to scope from there on out
"""


"""
Because there are a growing number of unaffiliated voters that we a priori claim to bias results, we want to get a better idea of what people's true party affiliations are by using their voter history, as to whether they participated in the most recent party primary.

Rules:
1. If the voter participated in a party primary (and have no declared party affiliation), they are affiliated with that party
2. If the voter participated in a third-party primary, they are manually reclassified. For NC:
3. Discard other unaffiliated voters (check numbers)

REP <- LIB, CST
DEM <- GRE

Sanity checks:
1. Check unaffiliated percentage before and after
2. Count number of people who participate in both Republican and Democrat primaries (tie breaking? exclude from analysis?)
"""

# start by chunk-loading the voter history file

df = pd.DataFrame()

chunks = pd.read_csv('bigdata/ncvhis.txt', '\t', usecols= ['election_lbl', 'election_desc', 'voted_party_cd', 'ncid'], chunksize=500000)


for chunk in chunks:

    # internally remove any 'removed' flagged people, because they could have moved out of state etc
    
    df = pd.concat([df, chunk], ignore_index=True)
    print(len(df))

"""
Broadly:
1. Filter elections for only thoe that include 'PRIMARY' in column election_desc
2. Deduplicate results upon ncid-year, keeping the result that is latest in the year
3. We get a voted_party_desc
4. Join with voter registration data: if NCID is non-REP/DEM, then override that with primary affiliation
"""

# get alphabetical contents of election_desc

df['election_desc'] = df['election_desc'].replace('(\d)', '', regex=True)
df['election_desc'] = df['election_desc'].replace(r'/', '', regex=True)
df['election_desc'] = df['election_desc'].str.strip()

df = df[df['election_desc'].str.contains('PRIMARY')]


# filter for only primary election participation to reduce space

df = df[['voted_party_cd', 'ncid', 'election_lbl']]

# sanity check percent of unaf

df[df['voted_party_cd'] == 'UNA'] # around 1.2% clerical errors?

# split date into month and year

df['year'] = df['election_lbl'].str.split('/', expand=True).iloc[:, 2]

df = df[['voted_party_cd', 'ncid', 'year']]

# lazy deduplication: take any primary affiliation in the year (doesn't change much, generally maximum of one primary per year)

df = df.drop_duplicates(['year', 'ncid'])

df = df[['voted_party_cd', 'ncid', 'year']]

# manually re-assign 
df['voted_party_cd'] = df['voted_party_cd'].replace({'LIB': 'REP', 'CST': 'REP', 'GRE': 'DEM'})

# drop clerical errors: UNA and nan
df = df[df['voted_party_cd'] != 'UNA']

df = df[df['voted_party_cd'].notna()] # very miniscule NAN errors

df.to_csv('bigdata/ncvhis_clean.csv')

"""
Now, load in by years, find most recent affiliation, then override
"""

import os

lof = os.listdir('bigdata/NC_analysisready')

# step one: primary vote imputation

for file in lof:
    cleandf = pd.read_csv('bigdata/NC_analysisready/' + file )

    # what is the initial unaf percentage
    sum(cleandf['Party'] == 'UNA') / len(cleandf)

    # now conduct the overriding

    year = file[0:4]

    subdf = df[df['year'] == year]

    cleandf = cleandf.merge(subdf[['ncid', 'voted_party_cd']], on = 'ncid', how='left')

    cleandf['Party'] = cleandf['Party'].replace({'LIB': 'REP'})

    cleandf['Party'] = cleandf.apply(lambda x: x['voted_party_cd'] if x['Party'] == 'UNA' else x['Party'], axis=1)

    sum(cleandf['Party'].isna()) / len(cleandf)


# step two: bayesian imputation