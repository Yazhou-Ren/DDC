# -*- coding: utf-8 -*-‘
from time import time
import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import metrics
from time import time

def load_data(InputFileName):
    #-------X and Y in 2 Files----------
    x = np.loadtxt(InputFileName+".txt")
    y = np.loadtxt(InputFileName+"_y.txt")

    # xy = np.concatenate((x,y),axis=1)
    #-------X and Y in 1 File-----------
    # xy = np.loadtxt(InputFileName +  ".txt")
    # x = xy[:, 0:xy.shape[1]-1]# get just x without y
    # y = xy[:,-1]
    # print("x.shape:", x.shape, "y.shape:", y.shape)
    return x,y

def ChooseCenter_N_Cluster(dc_percent,X):
    #calculate points distance
    dis_matrix = euclidean_distances(X, X)
    avg_dis = np.average(dis_matrix)
    #choose dc
    dc = dc_percent * avg_dis

    #calculate rho
    dis_matrix1 = dis_matrix / dc
    dis_matrix1 = np.multiply(dis_matrix1, dis_matrix1)
    dis_matrix1 = np.exp(-dis_matrix1)
    rho_ = dis_matrix1.sum(axis=1)
    rho = rho_ - 1
    avg_rho = np.average(rho)

    rho_list = [(rho[i], i) for i in range(rho.shape[0])]
    rho_sorted = sorted(rho_list, reverse=1)

    #calculate delta
    delta = [rho_sorted[0][0] for i in range(X.shape[0])]# avg_dis 存疑
    nneigh = [-1 for i in range(X.shape[0])]

    #如果rho不排序的话，就只能遍历整个矩阵找topK
    for ii in range(1, X.shape[0]):
        for jj in range(0, ii):
            id_p1 = rho_sorted[ii][1]  # get point1's id
            id_p2 = rho_sorted[jj][1]  # get point2's id
            if (dis_matrix[id_p1, id_p2] < delta[id_p1]):
                delta[id_p1] = dis_matrix[id_p1, id_p2]
                nneigh[id_p1] = id_p2
    # assignment
    cl = [-1 for i in range(X.shape[0])]
    center_num = 0
    each_rho_set=[]# every cluster's points rho
    each_cluster_num=[]# every cluster's points num
    centers_idx=[]
    centers_label=[]
    cluster_set=[]# every cluster state
    dis_m=dis_matrix
    for i in range(X.shape[0]):
        idx = rho_sorted[i][1]
        each_cluster = []
        if rho[idx] > avg_rho and delta[idx] > dc:
            cl[idx] = center_num
            each_cluster.append(idx)
            cluster_set.append(each_cluster)
            centers_idx.append(idx)
            centers_label.append(center_num)
            each_rho_set.append(rho[idx])
            each_cluster_num.append(1)
            center_num += 1

        else:
            if (cl[idx] == -1 and cl[nneigh[idx]] != -1):
                cl[idx] = cl[nneigh[idx]]
                each_rho_set[cl[idx]] +=rho[idx]
                each_cluster_num[cl[idx]] +=1
                # for k in range(len(cluster_set[cl[idx]])):
                #     id=cluster_set[cl[idx]][k]
                #     dis_m[id][idx]=0
                #     dis_m[idx][id]=0
                cluster_set[cl[idx]].append(idx)

    each_avg_rho=[]
    for j in range(len(each_rho_set)):
        each_avg_rho.append(each_rho_set[j]/each_cluster_num[j]) #every cluster's average rho
    print("inital cluster num:", len(cluster_set))
    return dc,cl,rho,delta,cluster_set,each_avg_rho,dis_m

def Corepoints_N_Merge(dc,cl,rho,cluster_set,each_avg_rho,dis_m):
    core_point=[]
    cluster_num=len(cluster_set)
    cl_borader=cl.copy()
    merged_cluster_num=cluster_num
    for i in range(cluster_num):
        temp_set=cluster_set[i]
        for j in range(len(cluster_set[i])):
            id = temp_set[j]
            if rho[id] <= each_avg_rho[i]:#boarder point
                cl_borader[id]= -1
                for k in range(len(cl)):
                    dis_m[id][k] = 0
                    dis_m[k][id] = 0
            else:
                core_point.append(id)

    for i in range(len(core_point)):
        idx=core_point[i]
        for j in range(len(cl)):
            if 0.0 <dis_m[idx][j] < dc and cl[idx] != cl[j]:
                cl = merge(idx,j,cluster_set,cl)
                merged_cluster_num -=1
    print("merged cluster num:",merged_cluster_num)
    return cl,cl_borader,cluster_num,merged_cluster_num

def merge(i,j,cluster_set,cl):
    label1 = cl[i]
    label2 = cl[j]
    if(cl[i]>cl[j]):
        label1=cl[j]
        label2=cl[i]
    print("merge:",label1,label2)
    for k in range(len(cluster_set[label2])):
        id = cluster_set[label2][k]
        cl[id]=label1
    cluster_set[label1].extend(cluster_set[label2])
    return cl


def match(y,cl):
    cl=np.array(cl)
    y=np.array(y)
    acc = np.round(metrics.acc(y, cl), 5)
    nmi = np.round(metrics.nmi(y, cl), 5)
    ari = np.round(metrics.ari(y, cl), 5)
    return acc,nmi,ari

def drawOriginGraph(pl, points, cl,cl_boarder, colorNum):
    x = points[:,0]
    y = points[:,1]
    cm = pl.get_cmap("RdYlGn")
    for i in range(len(points)):
        if (cl_boarder[i] != -1):
            pl.plot(x[i], y[i], '.', color=cm(cl[i] * 1.0 / colorNum))
        else:
            pl.plot(x[i], y[i], '.', color='k')

def drawDecisionGraph(pl, rho, delta, cl, cl_boarder, colorNum):
    cm = pl.get_cmap("RdYlGn")
    for i in range(len(rho)):
        if(cl_boarder[i]!=-1):
            pl.plot(rho[i], delta[i], '.', color=cm(cl[i] * 1.0 / colorNum))
        else:
            pl.plot(rho[i], delta[i], '.', color='k')
    pl.xlabel(r'$\rho$')
    pl.ylabel(r'$\delta$')

def DenPeakCluster(x):
    # InputFileName="flame"
    # InputFileName="Compund"
    # InputFileName = "D31"
    # InputFileName = "Spiral"
    # InputFileName = "Jain"
    # InputFileName = "R15"
    # InputFileName = "usps"
    # x,y=load_data(InputFileName)

    #--------------------------coefficient：dc---------------------------------(avg_delta*dc_percent)
    dc_percent = 0.1
    # print("dc_percent:",dc_percent)
    t1=time()
    dc, cl, rho, delta, cluster_set, each_avg_rho, dis_m=ChooseCenter_N_Cluster(dc_percent,x)
    t2=time()
    print("ChooseCenter_N_Cluster time:",t2-t1)

    # import pylab as pl
    # fig1 = pl.figure(1)
    # pl.subplot(321)
    # drawOriginGraph(pl, x, cl, cl,len(cluster_set))
    # pl.subplot(322)
    # drawDecisionGraph(pl, rho, delta, cl, cl,len(cluster_set))

    cl,cl_boarder,cluster_num,merged_cluster_num = Corepoints_N_Merge(dc,cl,rho,cluster_set,each_avg_rho,dis_m)
    t3=time()
    print("Corepoints_N_Merge time:",t3-t2)
    # pl.subplot(323)
    # drawOriginGraph(pl, x, cl, cl, cluster_num)
    # pl.subplot(324)
    # drawDecisionGraph(pl, rho, delta, cl, cl, cluster_num)
    #
    # pl.subplot(325)
    # drawOriginGraph(pl, x, cl, cl_boarder, cluster_num)
    # pl.subplot(326)
    # drawDecisionGraph(pl, rho, delta, cl, cl_boarder, cluster_num)
    # pl.savefig('./'+InputFileName+'.png')
    # pl.show()
    cl=np.array(cl)
    return cl,cl_boarder,merged_cluster_num,dc_percent,dc