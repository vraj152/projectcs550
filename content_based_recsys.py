import pandas as pd
from scipy import spatial

def computeSimilarity(itemProfile, weight_list):
    item_profile = pd.read_csv(itemProfile)

    each_item_vector = item_profile.iloc[:,1:21].values
    each_item_id = item_profile.iloc[:,0].values
    cosine_distances = {}
    i = 0
    
    while (i != len(each_item_vector)):
        cosine_distance = spatial.distance.cosine(weight_list,each_item_vector[i])
        cosine_distances[each_item_id[i]] = (1-cosine_distance)
        i = i + 1
    cosine_distances = {k: v for k, v in sorted(cosine_distances.items(), key=lambda item: item[1], reverse=True)}
    return cosine_distances