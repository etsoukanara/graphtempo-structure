
import pandas as pd
import itertools
import networkx as nx
import numpy as np
import time
import sys
sys.path.insert(1, 'graphtempo')
from graphtempo import *


edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
#edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/school_dataset/edges.csv', sep=' ', index_col=[0,1])
#####
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

######################

# =============================================================================
# # create nodes
# nodes_undirected_triangles_dict = {}
# for i in range(len(edges_df.columns)):
#     edges = edges_df.iloc[:,i][edges_df.iloc[:,i]!=0].index.tolist()
#     edges = [tuple(sorted(e)) for e in edges]
#     edges_set = set(edges)
#     nodes = sorted(list(set([i for e in edges for i in e])))
#     
#     triangles_un_new = []
#     for edge in edges:
#         for node in nodes:
#             if tuple(sorted((edge[1],node))) in edges_set and tuple(sorted((node,edge[0]))) in edges_set:
#                 t = tuple(sorted((edge[0],edge[1],node)))
#                 if t not in set(triangles_un_new):
#                     triangles_un_new.append(t)
# 
#     ones = [0]*len(edges_df.columns)
#     ones[i] = 1
#     for t in triangles_un_new:
#         if t not in nodes_undirected_triangles_dict.keys():
#             nodes_undirected_triangles_dict[t] = ones
#         else:
#             nodes_undirected_triangles_dict[t][i] = 1
# 
# 
# tr_nodes_df = pd.DataFrame(nodes_undirected_triangles_dict.values(), index=nodes_undirected_triangles_dict.keys())
# tr_nodes_df = tr_nodes_df.sort_index()
# tr_nodes_df.index.names = ['id1','id2','id3']
# 
# tr_nodes_df2 = tr_nodes_df.copy()
# tr_nodes_df2.index = tr_nodes_df.index.tolist()
# #tr_nodes_df2.reset_index().to_csv('triangles_datasets/undirected/dblp/nodes_triangles.csv', sep=',', index=None)
# 
# tr_nodes_df = pd.read_csv('triangles_datasets/undirected/dblp/nodes_triangles.csv', sep=',', index_col=[0])
# =============================================================================

############################

# NEW nodes
# create nodes
nodes_undirected_triangles_dict = {}
nodes_undirected_triangles_list = []
for i in range(len(edges_df.columns)):
    edges = edges_df.iloc[:,i][edges_df.iloc[:,i]!=0].index.tolist()
    edges = [tuple(sorted(e)) for e in edges]
    edges_set = set(edges)
    nodes = sorted(list(set([i for e in edges for i in e])))
    
    triangles_un_new = []
    for edge in edges:
        for node in nodes:
            if tuple(sorted((edge[1],node))) in edges_set and tuple(sorted((node,edge[0]))) in edges_set:
                t = tuple(sorted((edge[0],edge[1],node)))
                if t not in set(triangles_un_new):
                    triangles_un_new.append(t)
                    
    nodes_triangles_undirected_df = pd.DataFrame([], index=triangles_un_new)

    for j in range(len(edges_df.columns)):
        if j == i :
            tmp = [1]*len(nodes_triangles_undirected_df)
            nodes_triangles_undirected_df['col'+str(j)] = tmp
        else:
            tmp = [0]*len(nodes_triangles_undirected_df)
            nodes_triangles_undirected_df['col'+str(j)] = tmp
        del tmp
    nodes_triangles_undirected_df.index.name = 'id'
    nodes_undirected_triangles_list.append(nodes_triangles_undirected_df)

tr_nodes_df = pd.concat(nodes_undirected_triangles_list).groupby(['id']).sum()

cols = [str(i) for i in range(2000,2021)]
tr_nodes_df.columns = cols
tr_nodes_df.sort_index(inplace=True)

tr_nodes_df.to_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/nodes.csv', sep=',')

#tr_nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/nodes.csv', sep=',', index_col=0)



# create edges on each time point

for i in range(len(edges_df.columns)):
    print(i)
    triangles = tr_nodes_df.iloc[:,i][tr_nodes_df.iloc[:,i]!=0].index.tolist()
    #triangles = sorted([eval(triangle) for triangle in triangles])
    triangles = sorted(triangles)
    nodes_in_triangles = sorted(list(set([j for triangle in triangles for j in triangle])))
    
    edges_dict = {}
    for idx,node in enumerate(nodes_in_triangles):
        for ind,triangle in enumerate(triangles):
            if node in set(triangle):
                edges_dict.setdefault(node,[]).append(tuple(triangle))
    
    result = []
    for node,triangle_list in edges_dict.items():
        if len(triangle_list) > 1:
            result.extend(itertools.combinations(triangle_list, 2))
    
    result = sorted(list(set(result)))
    print(len(result))
    
    edges_triangles_undirected_df = pd.DataFrame(result)
    result = []
    edges_triangles_undirected_df.to_csv('triangles_datasets/undirected/dblp/prepro/data'+str(i)+'.csv', sep=',', index=None)


# add 0s and 1s for the dataframe of each time point

for i in range(len(edges_df.columns)):
    print('i: ',i)
    tria_edges_undirected = pd.read_csv('triangles_datasets/undirected/dblp/prepro/data'+str(i)+'.csv', sep=',')
    tria_edges_undirected.columns = ['Left','Right']
    tria_edges_undirected = tria_edges_undirected.set_index(['Left','Right'])
    for j in range(len(edges_df.columns)):
        print(j)
        if j == i :
            tmp = [1]*len(tria_edges_undirected)
            tria_edges_undirected['col'+str(j)] = tmp
        else:
            tmp = [0]*len(tria_edges_undirected)
            tria_edges_undirected['col'+str(j)] = tmp
        del tmp
    tria_edges_undirected.to_csv('triangles_datasets/undirected/dblp/prepro_2/data'+str(i)+'.csv', sep=',')
    

list_of_dfs = []  
for i in range(len(edges_df.columns)):
    print('i: ',i)
    list_of_dfs.append(
        pd.read_csv(
            'triangles_datasets/undirected/dblp/prepro_2/data'+str(i)+'.csv', 
            sep=',', 
            index_col=['Left','Right'],
            dtype='category'
            )
        )
    list_of_dfs[-1].sort_index(inplace=True)

list_of_dfs = [df.astype('int8') for df in list_of_dfs]

tr_edges_df = pd.concat(list_of_dfs).groupby(['Left','Right']).sum()
cols = [str(i) for i in range(2000,2021)]
tr_edges_df.columns = cols
tr_edges_df.sort_index(inplace=True)

# save to csv
tr_edges_df.to_csv('triangles_datasets/undirected/dblp/edges_triangles.csv', sep=',')

# read csv
tr_edges_df = pd.read_csv(
    'triangles_datasets/undirected/dblp/edges.csv', sep=',', index_col=['Left','Right'],)

#############################


# create gender attributes | static

time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
#time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/school_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)

idx = tr_nodes_df.index.tolist()
#idx = [eval(i) for i in idx]
multi_idx = pd.MultiIndex.from_tuples(idx)
tr_nodes_df.index = multi_idx
tr_nodes_df.columns = [str(i) for i in range(2000,2021)]

# triangles to df
triangles_undirected_df = pd.DataFrame(tr_nodes_df.reset_index().iloc[:,:3])
attr_triangles = []
for i in range(3):
    tmp = time_invariant_attr.loc[:,'gender'].to_frame().loc[triangles_undirected_df.iloc[:,i],:]['gender'].tolist()
    attr_triangles.append(tmp)
attr_triangles = list(zip(*attr_triangles))
attr_triangles = [''.join(sorted(i)) for i in attr_triangles]
attr_triangles = pd.DataFrame(attr_triangles)
#attr_triangles.index = tr_nodes_df.index
# or
attr_triangles.index = tr_nodes_df.index.tolist()
attr_triangles.columns = ['gender']
attr_triangles.to_csv('triangles_datasets/undirected/dblp/time_invariant_attr.csv', sep=',')

############################

# create #publications attributes | time-varying

time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_tempoGRAPHer_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_variant_attr.csv', sep=' ', index_col=0)

idx = tr_nodes_df.index.tolist()
idx = [eval(i) for i in idx]
multi_idx = pd.MultiIndex.from_tuples(idx)
tr_nodes_df.index = multi_idx
tr_nodes_df.columns = [str(i) for i in range(2000,2021)]

# merge # of publications ('121')
triangles_undirected_df = pd.DataFrame(tr_nodes_df.reset_index().iloc[:,:3])
attr_triangles_var = []
for j in time_variant_attr.columns:
    attr_tr_var = []
    for i in range(3):
        tmp = time_variant_attr.loc[:,j].to_frame().loc[triangles_undirected_df.iloc[:,i],:][j].tolist()
        tmp = [str(i) for i in tmp]
        attr_tr_var.append(tmp)
    attr_tr_var = list(zip(*attr_tr_var))
    attr_tr_var = ['_'.join(sorted(i)) for i in attr_tr_var]
    attr_tr_var = [0 if i[0]=='0' else i for i in attr_tr_var]
    attr_triangles_var.append(attr_tr_var)

attr_triangles_var = pd.DataFrame(attr_triangles_var).T
attr_triangles_var.index = tr_nodes_df.index.tolist()
attr_triangles_var.columns = tr_nodes_df.columns
attr_triangles_var = tr_nodes_df.where(tr_nodes_df==0, other=attr_triangles_var)
attr_triangles_var.index = attr_triangles_var.index.tolist()
attr_triangles_var.to_csv('triangles_datasets/undirected/dblp/time_variant_attr_str.csv', sep=',')

# sum # of publications ('121' --> 4)
triangles_undirected_df = pd.DataFrame(tr_nodes_df.reset_index().iloc[:,:3])
attr_triangles_var = []
for j in time_variant_attr.columns:
    attr_tr_var = []
    for i in range(3):
        tmp = time_variant_attr.loc[:,j].to_frame().loc[triangles_undirected_df.iloc[:,i],:][j].tolist()
        attr_tr_var.append(tmp)
    attr_tr_var = list(zip(*attr_tr_var))
    attr_tr_var = [sum(i) if i[0]!=0 and i[1]!=0 and i[2]!=0 else 0 for i in attr_tr_var]
    attr_triangles_var.append(attr_tr_var)

attr_triangles_var = pd.DataFrame(attr_triangles_var).T
attr_triangles_var.index = tr_nodes_df.index.tolist()
attr_triangles_var.columns = tr_nodes_df.columns
attr_triangles_var = tr_nodes_df.where(tr_nodes_df==0, other=attr_triangles_var)
attr_triangles_var.index = tr_nodes_df.index.tolist()
attr_triangles_var.to_csv('triangles_datasets/undirected/dblp/time_variant_attr.csv', sep=',')

###########################

# =============================================================================
# # test on aggregation algorithms
# # read files
# tr_edges_df = pd.read_csv('triangles_datasets/undirected/dblp/edges_triangles.csv', sep=',', index_col=[0,1])
# tr_nodes_df = pd.read_csv('triangles_datasets/undirected/dblp/nodes_triangles.csv', sep=',', index_col=0)
# tr_time_invariant_attr = pd.read_csv('triangles_datasets/undirected/dblp/attr_triangles.csv', sep=',', index_col=0)
# tr_time_invariant_attr.columns = ['gender']
# #tr_nodes_df = tr_nodes_df.iloc[:,:15]
# tr_nodes_df.columns = [str(i) for i in range(2000,2021)]
# 
# interval = tr_edges_df.columns
# res, tia = Union_Static(tr_nodes_df,tr_edges_df,tr_time_invariant_attr,interval)
# agg = Aggregate_Static_Dist(res,tia,stc_attrs=['gender'])
# # difference
# res, tia = Diff_Static(tr_nodes_df,tr_edges_df,tr_time_invariant_attr,['0'],['1'])
# agg_tmp = Aggregate_Static_All(res,tia,['gender'])
# agg = Diff_Post_Agg_Static(agg_tmp,['gender'])
# =============================================================================

