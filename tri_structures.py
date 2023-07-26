#
# 1. triangles on original graph
# 2. triangles on graph output of the operation 

import pandas as pd
import itertools
import networkx as nx
import numpy as np
import time
import sys
sys.path.insert(1, 'graphtempo')
from graphtempo import *


edges_df = pd.read_csv('datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])

# remove self loop found
idx_to_keep = []
for i in edges_df.index.tolist():
    if i[0] != i[1]:
        idx_to_keep.append(i)
edges_df = edges_df.loc[idx_to_keep,:]

for i in range(2000,2021):
    x = edges_df.loc[:,str(i)][edges_df.loc[:,str(i)]!=0].index.tolist()
    G = nx.Graph()
    G.add_edges_from(x)
    G0 = G.subgraph(max(nx.connected_components(G), key=len))
    nodes_G0 = list(G0.nodes)
    edges_G0 = list(G0.edges)
    print(i,': nodes in lcc', len(nodes_G0), 'edges in lcc', len(edges_G0))

nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/nodes.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)

time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_variant_attr.csv', sep=' ', index_col=0)

######################

# time for triangles creation (nodes, edges, attributes)

edges_dicts_lst = []
time_duration = [['nodes','edges','static','varying']]
for i in edges_df.columns:
    # nodes-tri
    s1 = time.time()
    edges = edges_df.loc[:,i][edges_df.loc[:,i]!=0].index.tolist()
    edges = [tuple(sorted(e)) for e in edges]
    edges_set = set(edges)
    nodes = sorted(list(set([n for e in edges for n in e])))
    
    nodes_tri = []
    for edge in edges:
        for node in nodes:
            if tuple(sorted((edge[1],node))) in edges_set and tuple(sorted((node,edge[0]))) in edges_set:
                t = tuple(sorted((edge[0],edge[1],node)))
                if t not in set(nodes_tri):
                    nodes_tri.append(t)
    
    nodes_tri = sorted(nodes_tri)
    e1 = time.time()
    
    # edges-tri
    s2=time.time()
    edges_dict = {}
    for ind,triangle in enumerate(nodes_tri):
        for node in triangle:
            edges_dict.setdefault(node,[]).append(tuple(triangle))

    edges_tri = []
    for node,triangle_list in edges_dict.items():
        if len(triangle_list) > 1:
            edges_tri.extend(itertools.combinations(triangle_list, 2))
    
    edges_tri = sorted(list(set(edges_tri)))
    e2 = time.time()
    edges_dicts_lst.append(edges_dict)
    
    nodes_tri = sorted(set([n for e in edges_tri for n in e]))
    # static
    s3 = time.time()
    static_tri = []
    for n in nodes_tri:
        tmp = [time_invariant_attr.loc[n[0],'gender'],
               time_invariant_attr.loc[n[1],'gender'],
               time_invariant_attr.loc[n[2],'gender']]
        tmp = ''.join(tmp)
        static_tri.append(tmp)
    e3 = time.time()
    # varying
    s4 = time.time()
    varying_tri = []
    for n in nodes_tri:
        tmp = [str(time_variant_attr.loc[n[0],i]),
               str(time_variant_attr.loc[n[1],i]),
               str(time_variant_attr.loc[n[2],i])]
        tmp = ''.join(tmp)
        varying_tri.append(tmp)
    e4 = time.time()

    time_duration.append([e1-s1, e2-s2, e3-s3, e4-s4])
    print(i, ': edges', len(edges_tri), ', nodes', len(nodes_tri), 
          ', static', len(static_tri), ', varying', len(varying_tri))

df = pd.DataFrame(time_duration).T


# time for triangles creation on operation result

intvl = list(edges_df.columns)

intervals = []
i = 1
for i in range(1,len(edges_df.columns)):
    intervals.append(intvl[:i+1])
    i += 1

intervals_diff = []
i = 1
for i in range(1,len(edges_df.columns)):
    intervals_diff.append(intvl[:i])
    i += 1


def triangles(operation,attr_operation,attrtype,ivl):
    # nodes-tri
    edges = operation[1].index.tolist()
    edges = [tuple(sorted(e)) for e in edges]
    edges_set = set(edges)
    nodes = sorted(list(set([n for e in edges for n in e])))
    nodes_tri = []
    for edge in edges:
        for node in nodes:
            if tuple(sorted((edge[1],node))) in edges_set and tuple(sorted((node,edge[0]))) in edges_set:
                t = tuple(sorted((edge[0],edge[1],node)))
                if t not in set(nodes_tri):
                    nodes_tri.append(t)
    nodes_tri = sorted(nodes_tri)
    # edges-tri
    edges_dict = {}
    for ind,triangle in enumerate(nodes_tri):
        for node in triangle:
            edges_dict.setdefault(node,[]).append(tuple(triangle))
    edges_tri = []
    for node,triangle_list in edges_dict.items():
        if len(triangle_list) > 1:
            edges_tri.extend(itertools.combinations(triangle_list, 2))
    edges_tri = sorted(list(set(edges_tri)))
    
    nodes_tri = sorted(set([n for e in edges_tri for n in e]))
    attr_tri = []
    if attrtype == 'static':
        # static
        for n in nodes_tri:
            tmp = [attr_operation.loc[n[0],:][0],
                   attr_operation.loc[n[1],:][0],
                   attr_operation.loc[n[2],:][0]]
            tmp = ''.join(tmp)
            attr_tri.append(tmp)
    else:
        # varying
        for i in ivl:
            tmp2 = []
            for n in nodes_tri:
                tmp = [str(attr_operation.loc[n[0],i]),
                       str(attr_operation.loc[n[1],i]),
                       str(attr_operation.loc[n[2],i])]
                tmp = ''.join(tmp)
                tmp2.append(tmp)
            attr_tri.append(tmp2)
    return(nodes_tri,edges_tri,attr_tri)


######## INTERSECTION

# G_DIST_I
result = []
for j in range(10):
    start_end_aggr = []
    for interval in intervals:
        inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        trin,trie,tria = triangles(inx,tia_inx,'static',interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res_gi = pd.DataFrame(result).T
res_gi['avg'] = res_gi.mean(axis=1)


# P_DIST_I
result = []
for j in range(10):
    start_end_aggr = []
    for interval in intervals:
        inx,tva_inx = Intersection_Variant(nodes_df,edges_df,time_variant_attr,interval)
        start = time.time()
        trin,trie,tria = triangles(inx,tva_inx,'varying',interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res_pi = pd.DataFrame(result).T
res_pi['avg'] = res_pi.mean(axis=1)


######## UNION
# G_ALL_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        trin,trie,tria = triangles(un,tia_un,'static',interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res_gu_all = pd.DataFrame(result).T
res_gu_all['avg'] = res_gu_all.mean(axis=1)


# G_DIST_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        trin,trie,tria = triangles(un,tia_un,'static',interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res_gu = pd.DataFrame(result).T
res_gu['avg'] = res_gu.mean(axis=1)

