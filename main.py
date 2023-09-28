from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics.cluster import normalized_mutual_info_score

# Prepping Dataset
df = pd.read_csv("Mall_customers.csv")
# Show Dataset Info
print("Info:")
print(df.info())
print("Describe")
print(df.describe())

# Selecting Columns to Cluster
df = df.iloc[:, [3, 4]].values
# Make a Scatter Plot
plt.scatter(df[:, 0], df[:, 1], s=10, c="black")
plt.show()

# K-means Cluster
startTimeKmeans = time()
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    Kmeans.fit(df)
    wcss.append(Kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()
endTimeKmeans = time()

# DBSCAN Algorithm
startTimeDBSCAN = time()
dbscan = DBSCAN(eps=5, min_samples=5)
# Fit the data (Training)
label = dbscan.fit(df)
# Predict the same data
labels = dbscan.fit_predict(df)
# Visualising the clusters
plt.scatter(df[labels == -1, 0], df[labels == -1, 1], s = 10, c = 'black')
plt.scatter(df[labels == 0, 0], df[labels == 0, 1], s = 10, c = 'blue')
plt.scatter(df[labels == 1, 0], df[labels == 1, 1], s = 10, c = 'red')
plt.scatter(df[labels == 2, 0], df[labels == 2, 1], s = 10, c = 'green')
plt.scatter(df[labels == 3, 0], df[labels == 3, 1], s = 10, c = 'brown')
plt.scatter(df[labels == 4, 0], df[labels == 4, 1], s = 10, c = 'pink')
plt.scatter(df[labels == 5, 0], df[labels == 5, 1], s = 10, c = 'yellow')
plt.scatter(df[labels == 6, 0], df[labels == 6, 1], s = 10, c = 'silver')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
endTimeDBSCAN = time()


# BIRCH Algorithm
startTimeBirch = time()
model = Birch(branching_factor=50, n_clusters=None, threshold=1.5)
# Fit the data (Training)
model.fit(df)
# Predict the same data
prediction = model.predict(df)
# Creating a scatter plot
plt.scatter(df[:, 0], df[:, 1], c=prediction, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.show()
endTimeBirch = time()

# Algorithm Evaluation
kmeansTime = endTimeKmeans - startTimeKmeans
dbscanTime = endTimeDBSCAN - startTimeDBSCAN
birchTime = endTimeBirch - startTimeBirch
print("K-Means Time: %.3f" % kmeansTime)
print("DBSCAN Time: %.3f" % dbscanTime)
print("BIRCH Time: %.3f" % birchTime)

visualizer = SilhouetteVisualizer(Kmeans, colors='yellowbrick')
visualizer.fit(df)

score = silhouette_score(df, labels, metric='euclidean')
scoreB = silhouette_score(df, prediction, metric='euclidean')

print('DBSCAN Silhouetter Score: %.3f' % score)
print('BIRCH Silhouetter Score: %.3f' % scoreB)



