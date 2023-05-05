import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.cluster import KMeans

def kmeans_clustering(input_file_path, output_file_path):
    # 导入Excel数据
    data = pd.read_excel(input_file_path, sheet_name='sheet1', usecols=[0, 1, 2], header=None)

    # 执行k-means聚类算法
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)

    # 获取聚类结果
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 将聚类结果保存到Excel文件中
    output_data = pd.DataFrame({'Light sleep duration': data.iloc[:, 0], 'Deep sleep time': data.iloc[:, 1], 'REM sleep time': data.iloc[:, 2], 'Cluster label': labels})
    output_data.to_excel(output_file_path, sheet_name='sheet1', index=False)

    return centroids


# 导入Excel数据
data = pd.read_excel('output.xlsx', sheet_name='sheet1', usecols=[0, 1, 2], header=None)

# 执行k-means聚类算法
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 将聚类结果保存到Excel文件中
#centroids = kmeans_clustering('output.xlsx', 'outputdata.xlsx')

# 定义三种颜色
colors = ['r', 'g', 'b']

# 绘制3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(data)):
    ax.scatter(data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2], c=colors[labels[i]])
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='k')
ax.set_xlabel('Light sleep time')
ax.set_ylabel('Deep sleep time')
ax.set_zlabel('REM sleep time')
plt.show()



