"""
This is the main file.
For now, it creates an unsupervised ML k-means algorithm for survey->elapsed_time
data
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from helpers_aiders_and_conveniencers.file_names_and_prefixes import files_info

# Get all panda data in a list
panda_list = []
numpy_list = []
for survey_id in files_info.survey_ids:
    string = files_info.survey_file_path_from_root + files_info.survey_file_name_prefix + survey_id
    
    temp_data_panda = pd.read_csv(string + '.csv')
    temp_numpy = np.array(temp_data_panda)
    panda_list.append(temp_data_panda) 
    numpy_list.append(temp_numpy)   
    

no_of_clusters = 4
# Plot initial clusters for all data
for i in range(len(numpy_list)):
    
    Kmean = KMeans(no_of_clusters)
    X = numpy_list[i][:,6]
    X = X.reshape(-1, 1)
    Kmean.fit(X)  
    
    print ('for index ', i, ' we have ', Kmean.cluster_centers_)
    plt.figure(i)
    plt.plot(Kmean.cluster_centers_, '*')
    plt.plot(X)
    plt.show()