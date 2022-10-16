#!/usr/bin/env python
# coding: utf-8

# 
# # Clustering Project
# 
# ## This algorithm is aimed to use K-means in order to divide students in a dormitory base on their similarities in cultures, personalities, and habits.

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sample = pd.read_csv("file:///C:/Users/Ali%20Fard/Desktop/clustering%20project/Clustering-Project_BNUT/sample2.csv")
# take a look at the dataset
sample.head()


# In[3]:


sample1=sample.drop(['day/ night person','student code','smoke'],axis=1)

sample1.head()


# In[4]:


from sklearn.preprocessing import StandardScaler
X=sample1
X = sample1.values[:,:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[5]:


clusterNum = 7
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 50)
k_means.fit(Clus_dataSet)
labels = k_means.labels_
print(labels)


# In[6]:


sample1["Clus_km"] = labels
sample1.head(10)


# In[7]:


sample1.groupby('Clus_km').mean()


# In[8]:


#sample_list=sample1.values.tolist()


# In[9]:


#c0=[]
#for i in range (195):
#    if sample_list[i][2]==0:
#         c0.append(sample_list[i])

# df = pd.DataFrame(c0)
# writer = pd.ExcelWriter('test0.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='c0', index=False)
# writer.save()
# writer.close()


# In[ ]:





# In[10]:


k_means_labels=labels
k_means_cluster_centers = k_means.cluster_centers_

# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(10, 10))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len(k_means_cluster_centers)), colors):
    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    #print(my_members)
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'b', markerfacecolor=col, marker='o')
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
#ax.set_xticks(())

# Remove y-axis ticks
#ax.set_yticks(())

# Show the plot
plt.show()


# In[12]:


for k in range(len(k_means_cluster_centers)):
    my_members = (k_means_labels == k)

    a=X[my_members]
    print (len(a))

print (a)


# In[ ]:





# In[ ]:




