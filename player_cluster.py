import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import mplcursors

# Read in the data
#Per 36 minutes was introduced in 1952
df = pd.read_csv('Per 36 Minutes.csv')


df = df[['player_id', 'player', 'g', 'trb_per_36_min', 'ast_per_36_min',  'pts_per_36_min', 'fg_percent']]
#Rows get dropped after df gets shortened due to nulls in unneeded columns
df = df.dropna()
#print(df.isnull().sum())

#Create player career per 36 minute averages
final_df = df.groupby(['player_id', 'player']).mean()

#Puts minimum games played at an average of 10 games per season
final_df = final_df[final_df['g'] >= 10]


final_df = final_df.reset_index()

#Visualize the data with fg% as the color
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(final_df['trb_per_36_min'], final_df['ast_per_36_min'], final_df['pts_per_36_min'], c=final_df['fg_percent'], marker='o',cmap='RdYlGn')
# ax.set_xlabel('Total Rebounds')
# ax.set_ylabel('Assists')
# ax.set_zlabel('Points')

# mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(final_df['player'].iloc[sel.target.index]))

# fig.colorbar(img)
# plt.show()
##############################################################################################################
#Determines the number of clusters by using the elbow method
# wcss = []
# for k in range(1, 21):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(final_df[['trb_per_36_min', 'ast_per_36_min', 'pts_per_36_min', 'fg_percent']])
#     wcss.append(kmeans.inertia_)

# plt.plot(range(1, 21), wcss)
# plt.title('Elbow Method')
# plt.show()
##############################################################################################################
#Create the clusters
kmeans = KMeans(n_clusters=7)
kmeans.fit(final_df[['trb_per_36_min', 'ast_per_36_min', 'pts_per_36_min', 'fg_percent']])
final_df['cluster'] = kmeans.predict(final_df[['trb_per_36_min', 'ast_per_36_min', 'pts_per_36_min', 'fg_percent']])

#Visualize the clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(final_df['trb_per_36_min'], final_df['ast_per_36_min'], final_df['pts_per_36_min'], c=final_df['cluster'], marker='o')
ax.set_xlabel('Total Rebounds')
ax.set_ylabel('Assists')
ax.set_zlabel('Points')

mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(final_df['player'].iloc[sel.target.index]))

fig.colorbar(img)
plt.show()





