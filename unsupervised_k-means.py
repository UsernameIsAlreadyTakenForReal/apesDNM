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

# Get all survey->elapsed_time data in a list
data = pd.read_csv(files_info.all_survey_file_from_root_complete)
data_numpy = np.array(data)

elapsed_time_array = data_numpy[:,6]
elapsed_time_array = elapsed_time_array.reshape(-1, 1)

no_of_clusters = 7
Kmean = KMeans(no_of_clusters)
Kmean.fit(elapsed_time_array)

print('Cluster centers: ', Kmean.cluster_centers_)

plt.figure(1)
plt.plot(Kmean.cluster_centers_, '*')
plt.plot(elapsed_time_array)
plt.show()


# # Plot initial clusters for all data
# for i in range(len(numpy_list)):
    
#     Kmean = KMeans(no_of_clusters)
#     X = numpy_list[i][:,6]
#     X = X.reshape(-1, 1)
#     Kmean.fit(X)  
    
#     print ('for index ', i, ' we have ', Kmean.cluster_centers_)
#     plt.figure(i)
#     plt.plot(Kmean.cluster_centers_, '*')
#     plt.plot(X)
#     plt.show()