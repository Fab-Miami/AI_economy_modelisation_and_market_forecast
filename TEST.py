# # from sklearn.cluster import KMeans
# # from sklearn.metrics import silhouette_score
# # import matplotlib.pyplot as plt
# # import numpy as np




# # cluster_1 = np.random.normal(loc=[0.2, 0.2], scale=[0.1, 0.1], size=(50, 2))
# # cluster_2 = np.random.normal(loc=[0.7, 0.7], scale=[0.1, 0.1], size=(30, 2))
# # cluster_3 = np.random.normal(loc=[0.8, 0.1], scale=[0.05, 0.05], size=(20, 2))

# # data = np.vstack([cluster_1, cluster_2, cluster_3])


# # kmeans = KMeans(n_clusters=3)
# # kmeans.fit(data)
# # centers = kmeans.cluster_centers_
# # labels = kmeans.labels_
# # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
# # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
# # plt.xlabel('Feature 1')
# # plt.ylabel('Feature 2')
# # plt.title('KMeans Clustering')
# # plt.grid(True)
# # plt.show()


# # # SILHOUETTE SCORE METHOD
# # sil_scores = []
# # for i in range(2, 11):
# #     kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
# #     label = kmeans.labels_
# #     sil_score = silhouette_score(data, label, metric='euclidean')
# #     sil_scores.append(sil_score)
# # plt.figure()
# # plt.plot(range(2, 11), sil_scores, marker='o')
# # plt.title('Silhouette Score Method')
# # plt.xlabel('Number of clusters')
# # plt.ylabel('Silhouette Score')
# # plt.grid(True)
# # plt.show()

# # ----------------------------------------------------------------


# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import numpy as np
# import matplotlib.pyplot as plt

# # Create some random clusters
# cluster_1 = np.random.normal(5, 2, 20)
# cluster_2 = np.random.normal(15, 2, 20)
# cluster_3 = np.random.normal(25, 2, 20)

# # Combine these clusters to create the dataset
# data = np.hstack((cluster_1, cluster_2, cluster_3)).reshape(-1, 1)

# # Finding the optimal number of clusters using silhouette score
# scores = []
# for n_clusters in range(2, 10):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(data)
#     labels = kmeans.labels_
#     scores.append(silhouette_score(data, labels))

# # Plotting the silhouette scores
# plt.figure()
# plt.bar(range(2, 10), scores)
# plt.title('Silhouette Scores for Different Numbers of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.show()

# # Finding the number of clusters with the highest silhouette score
# optimal_clusters = scores.index(max(scores)) + 2

# # Applying KMeans with the optimal number of clusters
# kmeans = KMeans(n_clusters=optimal_clusters)
# kmeans.fit(data)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_

# # Visualizing the clusters along with the cluster centers
# plt.figure()
# plt.scatter(data, [1] * len(data), c=labels, cmap='viridis')
# plt.scatter(centers, [1] * len(centers), c='red', marker='X')
# plt.yticks([])
# plt.title('KMeans Clustering with Optimal Number of Clusters')
# plt.xlabel('Data Points')
# plt.show()

# # Print optimal number of clusters
# print(f'The optimal number of clusters is: {optimal_clusters}')



print(1724146722/10000000000)
print(1724146722/1E9)
