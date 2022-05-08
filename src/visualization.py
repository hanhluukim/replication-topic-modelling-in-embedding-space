import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
#%matplotlib inline
import umap.umap_ as umap
import time
import plotly.express as px
from sklearn import cluster
from sklearn import metrics
import torch


def show_embedding_with_kmeans_umap(
  id2word, embedding_data, num_topics, etm_model_topic_embedding_alphas_weights,
  figures_path):

  we = torch.from_numpy(np.array(embedding_data)).float()
  print(we.shape)
  te = etm_model_topic_embedding_alphas_weights
  print(te.shape)

  # kmeans only for words
  kmeans = cluster.KMeans(n_clusters=10)
  kmeans.fit(we)
  labels = kmeans.labels_
  labels = ['cluster ' + str(c) for c in labels]
  centroids = kmeans.cluster_centers_

  # update information for umap
  topics = [f'topic' for i in range(num_topics)]
  words_topics = list(id2word.values()).extend(topics)

  # update labels als clustering
  for tp in topics:
    labels.append(tp)

  for t in te:
    embedding_data.append(t.detach().cpu().numpy())

  # dimension reduction with umap
  start = time.time()
  reducer = umap.UMAP(random_state=42,n_components=2)
  embedding = reducer.fit_transform(embedding_data)
  print('Duration: {} seconds'.format(time.time() - start))

  # show samples after dim-reduction in dataframe
  wb = pd.DataFrame(embedding, columns=['x', 'y'])
  wb['word-topic'] = words_topics #list(id2word.values())
  wb['cluster'] = labels

  # visualize
  fig = px.scatter(wb, 
                  #text = wb['word'],
                  x='x', y='y',
                  color = wb['cluster'],
                  title ="word-embedding-samples")
  fig.write_image(f'{figures_path}/wb_clustering_with_kmeans.png')
  fig.show()