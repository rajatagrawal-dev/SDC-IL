# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance, get_mean_example, get_mean_example2,get_close_embeddings
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
import random
from CIFAR100 import CIFAR100
import pdb
from embed_transformation import train_mapping
from scipy import spatial

#from tensorboardX import SummaryWriter
#writer = SummaryWriter('logs')

def displacement(Y1, Y2, embedding_old, sigma):
    DY = Y2-Y1
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
    W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
    W = weight_update(W, K=len(W[0]))
    W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
                          1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return displacement

def weight_update(W, K):
    bottomK = np.argpartition(W, -K)[:,:-K]
    for i in range(len(W)):
        W[i][bottomK[i]] = 0

    return W

def knn_displacement(Y1, Y2, embedding_old, K=100):
    K = Y1.shape[0]
    DY = Y2-Y1
    tree = spatial.KDTree(Y1, leafsize=K)
    closest_points = tree.query(embedding_old,K)[1]
    displacement = np.mean(DY[closest_points],axis=1)
    return displacement

def get_mapping_function(Y1, Y2, embedding_old):
    DY = Y2-Y1
    displacement = train_mapping(Y1, DY, embedding_old)
    #new_embeddings = mapping_model(embedding_old)
    print(displacement.size())
    return displacement.numpy()

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cub')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument("-method", type=str, default='no', help='Choose FT or SC')
parser.add_argument('-mapping_test', help='Print more data',
                    action='store_true')
parser.add_argument('-sigma_test', default=0, type=float, help='sigma_test')
parser.add_argument('-real_mean', help='Print more data', action='store_true')
parser.add_argument('-epochs', default=600, type=int,
                    metavar='N', help='epochs for training process')
parser.add_argument('-exp', type=str, default='exp1',
                    help="learning rate of new parameters")
parser.add_argument('-task', default=1, type=int, help='task')
parser.add_argument('-base', default=50, type=int, help='task')


args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []

for i in os.listdir(args.r):
    if i.endswith("%d_model.pkl" % (args.epochs-1)):   # 500_model.pkl
        models.append(os.path.join(args.r, i))

models.sort()
if args.task > 10:
    models.append(models[1])
    del models[1]

if args.data == 'cub':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = 'DataSet/CUB_200_2011'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')

    num_classes = 200

if args.data == 'car':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = 'DataSet/Car196'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')

    num_classes = 196


if args.data == 'flower':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values),
    ])
    root = 'DataSet/flowers'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')
    num_classes = 102

if args.data == "cifar100":

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    root = 'DataSet'
    traindir = root + '/cifar'
    testdir = root + '/cifar'
    num_classes = 100

if args.data == 'imagenet_sub' or args.data == 'imagenet_full':
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        # transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    transform_test = transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_values,
                             std=std_values)
    ])
    root = '/datatmp/datasets/ILSVRC12_256'
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'val')
    num_classes = 100


num_task = args.task
num_class_per_task = (num_classes-args.base)/(num_task-1)
np.random.seed(args.seed)
random_perm = np.random.permutation(num_classes)


print('Test starting -->\t')

class_mean = []
class_std = []
class_label = []
class_mean_mapping = []
class_mean_example = []
class_embed_dist = []

for task_id in range(num_task):

    index = random_perm[:args.base+task_id*num_class_per_task]
    if task_id == 0:
        index_train = random_perm[:args.base]
    else:
        index_train = random_perm[args.base +
                                  (task_id-1)*num_class_per_task:args.base+task_id*num_class_per_task]
	#index_train = random_perm[:args.base+task_id*num_class_per_task]

    if args.data == 'cifar100':
        trainfolder = CIFAR100(root=traindir, train=True, download=True,
                               transform=transform_train, index=index_train)
        testfolder = CIFAR100(root=traindir, train=False,
                              download=True, transform=transform_test, index=index)
    else:
        trainfolder = ImageFolder(traindir, transform_train, index=index_train)
        testfolder = ImageFolder(testdir, transform_test, index=index)

    train_loader = torch.utils.data.DataLoader(
        trainfolder, batch_size=128, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        testfolder, batch_size=128, shuffle=False, drop_last=False)
    print('Test %d\t' % task_id)

    model = torch.load(models[task_id])

    train_embeddings_cl, train_labels_cl = extract_features(
        model, train_loader, print_freq=32, metric=None)
    val_embeddings_cl, val_labels_cl = extract_features(
        model, test_loader, print_freq=32, metric=None)

    class_data = []
    for i, data in enumerate(train_loader,0):
        inputs, pids = data
        if class_data == []:
            class_data = inputs.numpy()
        else:
            class_data = np.vstack((class_data, inputs.numpy()))
    
    #print("class data shape is: ", class_data.shape)
    #exit(1)

    #class_data = class_data[ind_cl]

    # Test for each task
    #class_label = []
    #class_mean = []
    k = 200
    for i in index_train:
        ind_cl = np.where(i == train_labels_cl)[0]
        embeddings_tmp = train_embeddings_cl[ind_cl]
        data_tmp = class_data[ind_cl]
	#mean_example = np.mean(class_data, axis=0)
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
	#close_to_mean_dist, close_to_mean_embeddings = get_close_embeddings(embeddings_tmp,k)
##	
	mean_example = get_mean_example2(embeddings_tmp, data_tmp, class_mean[-1].reshape(1,-1))
	#class_mean_example.append(mean_example)
	#class_embed_dist = np.append(class_embed_dist, close_to_mean_dist)
	if class_mean_example == []:
	    #class_mean_example = close_to_mean_embeddings
	    class_mean_example = mean_example
	else:
	    #class_mean_example = np.vstack((class_mean_example, close_to_mean_embeddings))
	    class_mean_example = np.vstack((class_mean_example, mean_example))

    if task_id > 0 and args.mapping_test:
	model_old = torch.load(models[task_id-1])
	mean_X = torch.Tensor(class_mean_example[:(args.base+(task_id-1)*num_class_per_task)*k])
	mean_Y = torch.Tensor(np.repeat(class_label[:args.base+(task_id-1)*num_class_per_task],k))
	#print("X shape:", mean_X.shape)
	#print("Y shape:", mean_Y.shape)
	mean_ds = torch.utils.data.TensorDataset(mean_X, mean_Y)
	mean_dl = torch.utils.data.DataLoader(mean_ds,shuffle=False, drop_last=False)
	mean_output, mean_output_label = extract_features(model, mean_dl, print_freq=32, metric=None)
	class_wise_mean = []
	for i in range(args.base+(task_id-1)*num_class_per_task):
	    curr_mean = np.average(mean_output[i*k:(i+1)*k],axis=0)
	    class_wise_mean.append(curr_mean)
	#print(mean_output.shape)
	#print(mean_output)
	class_mean[:args.base+(task_id-1)*num_class_per_task] = class_wise_mean

    #if task_id > 0 and args.mapping_test:
    #    model_old = torch.load(models[task_id-1])
    #    train_embeddings_cl_old, train_labels_cl_old = extract_features(
    #        model_old, train_loader, print_freq=32, metric=None)

    #    MU = np.asarray(class_mean[:args.base+(task_id-1)*num_class_per_task])
    #    #if task_id==1:
    #    #    np.savetxt('task_1_old_embed.txt',train_embeddings_cl_old)
    #    #    np.savetxt('task_1_new_embed.txt',train_embeddings_cl)
    #    #    np.savetxt('task_1_MU.txt',MU)

    #    #gap = get_mapping_function(train_embeddings_cl_old,
    #    #                   train_embeddings_cl, MU)
    #    gap = displacement(train_embeddings_cl_old,
    #                       train_embeddings_cl, class_mean_example, args.sigma_test)
    #    
    #    class_mean_example += gap
    #    new_means = []
    #    weights = np.exp(-class_embed_dist**2/(2*args.sigma_test*args.sigma_test))
    #    for i in range(args.base+(task_id-1)*num_class_per_task):
    #        curr_weights = weights[i*k:(i+1)*k]
    #        #curr_weights[-1]=1
    #        curr_mean = np.average(class_mean_example[i*k:(i+1)*k],axis=0)
    #        new_means.append(curr_mean)
    #    new_means = np.asarray(new_means)
    #    #gap = displacement(train_embeddings_cl_old,
    #    #                   train_embeddings_cl, MU, args.sigma_test)
    #    #MU += gap
    #    class_mean[:args.base+(task_id-1)*num_class_per_task] = new_means

    embedding_mean_old = []
    embedding_std_old = []
    gt_all = []
    estimate_all = []

    acc_ave = 0
    for k in range(task_id+1):
        if k == 0:
            tmp = random_perm[:args.base]
        else:
            tmp = random_perm[args.base +
                              (k-1)*num_class_per_task:args.base+k*num_class_per_task]
        gt = np.isin(val_labels_cl, tmp)

        pairwise_distance = euclidean_distances(
            val_embeddings_cl, np.asarray(class_mean))
        estimate = np.argmin(pairwise_distance, axis=1)
        estimate_label = [index[j] for j in estimate]
        estimate_tmp = np.asarray(estimate_label)[gt]
        if task_id == num_task-1:
            if estimate_all == []:
                estimate_all = estimate_tmp
                gt_all = val_labels_cl[gt]
            else:
                estimate_all = np.hstack((estimate_all, estimate_tmp))
                gt_all = np.hstack((gt_all, val_labels_cl[gt]))

        acc = np.sum(estimate_tmp ==
                     val_labels_cl[gt])/float(len(estimate_tmp))
        if k == 0:
            acc_ave += acc*(float(args.base) /
                            (args.base+task_id*num_class_per_task))
        else:
            acc_ave += acc*(float(num_class_per_task) /
                            (args.base+task_id*num_class_per_task))
        print("Accuracy of Model %d on Task %d is %.3f" % (task_id, k, acc))
    print("Weighted Accuracy of Model %d is %.3f" % (task_id, acc_ave))
