import numpy as np

def calc_isolation_other(long, lat, party, tree, df, k=101, c=0.0001, a = 1):
    
    #query 1001 and drop the first one which is probably itself or would only change the isolation number inconsequentially

    nearest_array = tree.query([long, lat], k = k) 
    
    # filter for same party peeps
    
    index_of_neighbors = nearest_array[1][1:]
    neigh = df.iloc[index_of_neighbors]
    #neigh = neigh.reset_index()

    my_party = party

    fellow_pals = neigh[neigh['Party'] == my_party]

    lol = []
    for index_no in fellow_pals.index:
        lol.append(np.where(nearest_array[1] == index_no)[0][0])
    
    fellow_distances = nearest_array[0][lol]
    num = sum(1 / (fellow_distances + c)**a )

    denom = sum(1 / (nearest_array[0][1:] + c)**a ) # we have to consistently exclude the first person
    isol = num/denom
    return isol





"""
BEST FUNCTION RIGHT NOW: around 1.2 seconds per 10000 calls --> around 12 minutes per 6m calls --> around 2.6 hours for the whole NC set

"""

def calc_isolation(long, lat, party, tree, dfarray, k=101, c=0.0001, a = 1):
    
    #query 1001 and drop the first one which is probably itself or would only change the isolation number inconsequentially

    nearest_array = tree.query([long, lat], k = k) 
    
    # filter for same party peeps
    
    index_of_neighbors = nearest_array[1][1:]
    neigh = dfarray[:,index_of_neighbors]
    #neigh = neigh.reset_index()

    my_party = party

    fellow_pals = neigh[1][np.where(neigh[0] == my_party)[0]]

    lol = []
    for index_no in fellow_pals:
        lol.append(np.where(nearest_array[1] == index_no)[0][0])
    
    fellow_distances = nearest_array[0][lol]
    num = sum(1 / (fellow_distances + c)**a )

    denom = sum(1 / (nearest_array[0][1:] + c)**a ) # we have to consistently exclude the first person
    isol = num/denom
    return isol