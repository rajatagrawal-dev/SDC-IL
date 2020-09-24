from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from utils import to_numpy
import numpy as np

from utils.meters import AverageMeter
import pdb
from scipy import spatial

def extract_features(model, data_loader, print_freq=1, metric=None):
    model=model.cuda()
    model.eval()
  
    features = []
    labels = []
 
    for i, data in enumerate(data_loader,0):
        imgs, pids=data
      
        inputs = imgs.cuda()
        with torch.no_grad():
            _,outputs = model(inputs)
            outputs = outputs.cpu().numpy()
     
        if features==[]:
            features=outputs
            labels=pids
        else:
            features=np.vstack((features,outputs))
            labels = np.hstack((labels,pids))

    return features, labels

def get_mean_example(class_embeddings, class_data, sigma=0.2):
    #class_data = []
    #for i, data in enumerate(data_loader,0):
    #    inputs, pids = data
    #    if class_data == []:
    #        class_data = inputs.numpy()
    #    else:
    #        class_data = np.vstack((class_data, inputs.numpy()))

    #class_embeddings = all_embeddings[ind_cl]
    #class_data = class_data[ind_cl]

    mean_embedding = np.mean(class_embeddings, axis=0)
    distances = np.linalg.norm(class_embeddings-mean_embedding,axis=1)
    weights = np.e**(-distances**2/(2*sigma*sigma))
    weighted_average = np.average(class_data,weights=weights,axis=0)
    return weighted_average

def get_mean_example2(Y1, Y2, embedding_old, sigma=0.2):
    DY = Y2
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
    W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
    W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    #displacement = np.sum(np.tile(W_norm[:, :, None], [
    #                      1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    #displacement = np.average(DY, weights=W_norm[0], axis=0)
    #print("The W_norm is:", W_norm[0])
    print("W shape:",W_norm.shape)
    k = 200
    idx = np.argpartition(W_norm[0], k)
    exemplars = DY[idx[:k]]#try increasing storage to 10 exemplars per class
    #print("examplar shape:", exemplars.shape)
    return exemplars

def get_close_embeddings(embeddings_tmp, k=10):
    mean_embedding = np.mean(embeddings_tmp, axis=0)
    done = [False]*len(embeddings_tmp)
    closest_points = []
    closest_dist = []

    for i in range(k):
	curr_best_idx = -1
	curr_best_val = float('inf')
	#curr_mean = np.mean(closest_points, axis=0)
	for j in range(len(done)):
	    if done[j]:
		continue
	    new_mean = np.mean(closest_points+[embeddings_tmp[j]],axis=0)
	    error = np.linalg.norm(mean_embedding-new_mean)
	    if error < curr_best_val:
		curr_best_idx = j

	closest_points.append(embeddings_tmp[curr_best_idx])
	closest_dist.append(curr_best_val)
	done[curr_best_idx] = True

    #tree = spatial.KDTree(embeddings_tmp, leafsize=k-1)
    #closest_dist, closest_points = tree.query(mean_embedding, k-1)
    #closest = embeddings_tmp[closest_points]
    #closest_dist = np.append(closest_dist, 0)
    #closest = np.append(closest,[mean_embedding],axis=0)
    return closest_dist, closest_points


def pairwise_distance(features, metric=None):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True)
    dist = dist.expand(n, n)
    dist = dist + dist.t()
    dist = dist - 2 * torch.mm(x, x.t()) + 1e5 * torch.eye(n)
    dist = torch.sqrt(dist)
    return dist


def pairwise_similarity(features):
    n = len(features)
    x = torch.cat(features)
    x = x.view(n, -1)
    similarity = torch.mm(x, x.t()) - 1e5 * torch.eye(n)
    return similarity


