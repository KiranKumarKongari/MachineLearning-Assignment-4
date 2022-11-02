# 2. Apply K means clustering in the dataset provided:
#     • Remove any null values by the mean.
#     • Use the elbow method to find a good number of clusters with the K-Means algorithm
#     • Calculate the silhouette score for the above clustering


# Importing required libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")

# customer is a dataframe that we load the csv data.
customer = pd.read_csv("C:/Users/Kiran Kumar Kongari/PycharmProjects/ML-Assignment-4/Datasets/K-Mean_Dataset.csv")
print("\nThe Dataframe is : \n", customer)

# Checking the columns having null values and displaying the resultant columns.
columnsWithNullValues = customer.isna().any()

# a. Replacing the null values with the mean
customer['CREDIT_LIMIT'] = customer['CREDIT_LIMIT'].fillna(customer['CREDIT_LIMIT'].mean())
customer['MINIMUM_PAYMENTS'] = customer['MINIMUM_PAYMENTS'].fillna(customer['MINIMUM_PAYMENTS'].mean())

# Verifying the dataframe again for null values
f = customer[customer.isna().any(axis=1)]
print('\nVerifying customer dataframe for null values again : ', f)

# dropping the CUST_ID column
customerDf = customer.drop(['CUST_ID'], axis='columns')

# Use elbow method to find optimal number of clusters
wcss = []  # WCSS is the sum of squared distance between each point and the centroid in a cluster
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(customerDf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the above plot we can observe a sharp edge at k=2 (Number of clusters).
# Hence k=2 can be considered a good number of the cluster to cluster this data.

km = KMeans(n_clusters=2)
km.fit(customerDf)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(customerDf)

# Calculating the silhouette score for the above clustering
score = metrics.silhouette_score(customerDf, y_cluster_kmeans)
print("\nSilhouette Score for the above cluster is : ", score)

# --------------------------------------------------------------------------------------------------------------------------------

# 3.Try feature scaling and then apply K-Means on the scaled features. Did that improve the Silhouette score? If
#   Yes, can you justify why?

# Applying the feature scaling and using resultant dataset for applying K-Means
scaler = preprocessing.StandardScaler()
scaler.fit(customerDf)
X_scaled_array = scaler.transform(customerDf)
X_scaled = pd.DataFrame(X_scaled_array, columns=customerDf.columns)
print('\n', X_scaled)

# Use elbow method to find optimal number of clusters
wcss = []  # WCSS is the sum of squared distance between each point and the centroid in a cluster
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the above plot we can observe a sharp edge at k=4 but the silhouette_score is high at k=3(Number of clusters).
# Hence k=3 can be considered a good number of the cluster to cluster this data.

km = KMeans(n_clusters=3)
km.fit(X_scaled)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)

# Calculating the silhouette score for the above clustering
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print("\nSilhouette Score after applying feature scaling is : ", score)

# In my case, after applying the feature scaling the Silhouette didn't increase but decreased.


