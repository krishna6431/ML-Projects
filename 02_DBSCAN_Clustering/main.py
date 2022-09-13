#!/usr/bin/env python
# coding: utf-8

# **Assignment: 2 (DBSCAN Clustering Algorithm Implementation)<br>
# Name: Krishna Kant Verma<br>
# Roll No: 2211cs19<br>
# M.Tech CSE**

# **Importing all the libraries**

# In[1]:


#importing all the required library for this Assignment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
#kmeans clustering
from sklearn.cluster import KMeans
#dbscan clustering
from sklearn.cluster import DBSCAN


# **Loading Datasets**

# In[2]:


#loading datasets into pandas dataframe 
dataframeBlobs=pd.read_csv("datasets/cluster_blobs.csv")
dataframeCircles=pd.read_csv("datasets/cluster_circles.csv")
dataframeMoons=pd.read_csv("datasets/cluster_moons.csv")


# **Describing Datasets**

# In[3]:


#datablobs dataframe description
dataframeBlobs.describe()


# In[4]:


#datacircles datasets description
dataframeCircles.describe()


# In[5]:


#dataframe datasets description
dataframeMoons.describe()


# **Checking that whether there is missing data or not**

# In[6]:


imputer = SimpleImputer(missing_values = np.nan,strategy ='mean')
dataBlobs= imputer.fit_transform(dataframeBlobs)
dataCircles= imputer.fit_transform(dataframeCircles)
dataMoons= imputer.fit_transform(dataframeMoons)


# **Scaling Each data sets**

# In[7]:


scaler=StandardScaler()
dataBlobs=scaler.fit_transform(dataBlobs)
dataCircles=scaler.fit_transform(dataCircles)
dataMoons=scaler.fit_transform(dataMoons)
print("Data Blobs: \n",dataBlobs)
print("Data Circles: \n",dataCircles)
print("Data Moons: \n",dataMoons)


# **Plotting the scatter plot for each dataset**

# In[8]:


#for blob
plt.scatter(dataBlobs[:,[0]],dataBlobs[:,[1]],c='orange', s=10)
plt.title("Blobs Dataset")
plt.show


# In[9]:


#for circle
plt.scatter(dataCircles[:,[0]],dataCircles[:,[1]],c='violet', s=10)
plt.title("Circles Dataset")
plt.show


# In[10]:


#for moons
plt.scatter(dataMoons[:,[0]],dataMoons[:,[1]],c='green', s=10)
plt.title("Moons Dataset")
plt.show


# **Applying Kmeans Clustering on each Dataset (3-Means)**

# In[11]:


#Kmeans for blob (3-means)
kmeansBlobs= KMeans(n_clusters=3, random_state= 100,max_iter=75)
kmeansBlobs.fit(dataBlobs)
print(f'Blobs Dataset lables using KMeans: {kmeansBlobs.labels_}')


# In[12]:


#Kmeans for Circles (3-means)
kmeansCircles= KMeans(n_clusters=3, random_state= 100,max_iter=75)
kmeansCircles.fit(dataCircles)
print(f'Circles Dataset lables using KMeans: {kmeansCircles.labels_}')


# In[13]:


#Kmeans for Moons (3-means)
kmeansMoons= KMeans(n_clusters=3, random_state= 100,max_iter=75)
kmeansMoons.fit(dataMoons)
print(f'Moons Dataset lables using KMeans: {kmeansMoons.labels_}')


# **Silhouette Score for each clustered datasets**

# In[14]:


print("Silhouette Score:")
silhouetteKmeansBlobs= silhouette_score(dataBlobs, kmeansBlobs.labels_)
print(f'Blobs Dataset: {round(silhouetteKmeansBlobs,2)}')

silhouetteKmeansCircles= silhouette_score(dataCircles, kmeansCircles.labels_)
print(f'Circles Dataset: {round(silhouetteKmeansCircles,2)}')

silhouetteKmeansMoons= silhouette_score(dataMoons, kmeansMoons.labels_)
print(f'Moons Dataset: {round(silhouetteKmeansMoons,2)}')


# **Plotting Graph for each clustered datasets**

# In[15]:


#for blobs datasets
cluster1=dataframeBlobs
cluster1['label']=kmeansBlobs.labels_
print("Cluster1:\n",cluster1)
dataframe_0 = cluster1[cluster1['label'] == 0]
dataframe_1 = cluster1[cluster1['label'] == 1]
dataframe_2 = cluster1[cluster1['label'] == 2]


# In[16]:


plt.scatter(dataframe_0['X1'], dataframe_0['X2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X1'], dataframe_1['X2'], c='violet', s=10, label='Cluster B')
plt.scatter(dataframe_2['X1'], dataframe_2['X2'], c='green', s=10, label='Cluster C')
plt.title("Blobs Dataset")
plt.legend()


# In[17]:


#for circles dataset
cluster2=dataframeCircles
cluster2['label']=kmeansCircles.labels_
print("Cluster2:\n",cluster2)
dataframe_0 = cluster2[cluster2['label'] == 0]
dataframe_1 = cluster2[cluster2['label'] == 1]
dataframe_2 = cluster2[cluster2['label'] == 2]


# In[18]:


plt.scatter(dataframe_0['X1'], dataframe_0['X2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X1'], dataframe_1['X2'], c='violet', s=10, label='Cluster B')
plt.scatter(dataframe_2['X1'], dataframe_2['X2'], c='green', s=10, label='Cluster C')
plt.title("Circles Dataset")
plt.legend()


# In[19]:


#for moon datasets
cluster3=dataframeMoons
cluster3['label']=kmeansMoons.labels_
print("Cluster3:\n",cluster3)
dataframe_0 = cluster3[cluster3['label'] == 0]
dataframe_1 = cluster3[cluster3['label'] == 1]
dataframe_2 = cluster3[cluster3['label'] == 2]


# In[20]:


plt.scatter(dataframe_0['X_1'], dataframe_0['X_2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X_1'], dataframe_1['X_2'], c='violet', s=10, label='Cluster B')
plt.scatter(dataframe_2['X_1'], dataframe_2['X_2'], c='green', s=10, label='Cluster C')
plt.title("Moons Dataset")
plt.legend()


# **Applying DBSCAN Clustering on each datsets + Silhoutte Score + Plot**

# In[21]:


#for blob dataset
dbscanBlobs = DBSCAN(eps=1.293, min_samples=6)
labels_blobs=dbscanBlobs.fit_predict(dataBlobs)
print(f'Labels for Blobs dataset using DBSCAN: {labels_blobs}')
print(f'No.of clusters on circles dataset using DBSCAN:{np.unique(labels_blobs)}')


# In[22]:


silhouetteDbscanBlobs = silhouette_score(dataBlobs, labels_blobs)
print(f'Silhouette Score for DBSCAN on Blobs Dataset: {round(silhouetteDbscanBlobs,2)}')


# In[23]:


cluster1=dataframeBlobs
cluster1['label']=labels_blobs
print("Cluster1:\n",cluster1)
dataframe_0 = cluster1[cluster1['label'] == 0]
dataframe_1 = cluster1[cluster1['label'] == 1]
dataframe_2 = cluster1[cluster1['label'] == 2]


# In[24]:


plt.scatter(dataframe_0['X1'], dataframe_0['X2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X1'], dataframe_1['X2'], c='violet', s=10, label='Cluster B')
plt.scatter(dataframe_2['X1'], dataframe_2['X2'], c='green', s=10, label='Cluster C')
plt.title("Blobs Dataset")
silhouetteDbscanBlobs = silhouette_score(dataBlobs, labels_blobs)
print(f'Silhouette Score for DBSCAN on Blobs Dataset: {round(silhouetteDbscanBlobs,2)}')
plt.legend()


# In[25]:


#for circle datasets
dbscanCircles = DBSCAN(eps=0.298, min_samples=5)
labels_circles=dbscanCircles.fit_predict(dataCircles)
print(f'Labels for Circles dataset using DBSCAN: {labels_circles}')
print(f'No.of clusters on circles dataset using DBSCAN:{np.unique(labels_circles)}')


# In[26]:


silhouetteDbscanCircles = silhouette_score(dataCircles, labels_circles)
print(f'Silhouette Score for DBSCAN on Circles Dataset: {round(silhouetteDbscanCircles,2)}')


# In[27]:


cluster2=dataframeCircles
cluster2['label']=labels_circles
print("Cluster2:\n",cluster2)
dataframe_0 = cluster2[cluster2['label'] == 0]
dataframe_1 = cluster2[cluster2['label'] == 1]


# In[28]:


plt.scatter(dataframe_0['X1'], dataframe_0['X2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X1'], dataframe_1['X2'], c='violet', s=10, label='Cluster B')
plt.title("Circles Dataset")
plt.legend()


# In[29]:


#for moon dataset
dbscanMoons = DBSCAN(eps=0.298, min_samples=5)
labels_moons=dbscanMoons.fit_predict(dataMoons)
print(f'Labels for Moons dataset using DBSCAN: {labels_moons}')
print(f'No.of clusters on moons dataset using DBSCAN:{np.unique(labels_moons)}')


# In[30]:


silhouetteDbscanMoons = silhouette_score(dataMoons, labels_moons)
print(f'Silhouette Score for DBSCAN on Circles Dataset: {round(silhouetteDbscanMoons,2)}')


# In[31]:


cluster3=dataframeMoons
cluster3['label']=labels_moons
print("Cluster3:\n",cluster3)
dataframe_0 = cluster3[cluster3['label'] == 0]
dataframe_1 = cluster3[cluster3['label'] == 1]


# In[32]:


plt.scatter(dataframe_0['X_1'], dataframe_0['X_2'], c='orange', s=10, label='Cluster A')
plt.scatter(dataframe_1['X_1'], dataframe_1['X_2'], c='violet', s=10, label='Cluster B')
plt.title("Moons Dataset")
plt.legend()


# Thank You So Much
# ©krishna_2211cs19@iitp.ac.in
