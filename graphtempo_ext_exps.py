
import pandas as pd
import networkx as nx
import itertools
import copy
import time
from pyvis.network import Network
import pathlib
from functools import reduce
import numpy as np
from graphtempo import *


#dblp
# READ edges, nodes, static and variant attributes from csv
edges_df = pd.read_csv('datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
nodes_df = pd.read_csv('datasets/dblp_dataset/nodes.csv', sep=' ', index_col=0)
time_variant_attr = pd.read_csv('datasets/dblp_dataset/time_variant_attr.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_invariant_attr.rename(columns={'0': 'gender'}, inplace=True)
nodes_df.index.names = ['userID']

# replace notation for gender attribute
time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)

# =============================================================================
# # school
# edges_df = pd.read_csv('school_dataset/edges.csv', sep=' ')
# edges_df.set_index(['Left', 'Right'], inplace=True)
# nodes_df = pd.read_csv('school_dataset/nodes.csv', sep=' ', index_col=0)
# time_invariant_attr = pd.read_csv('school_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
# =============================================================================


# =============================================================================
# # dblp triangles
# edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/edges.csv', sep=',', dtype='category')
# edges_df.set_index(['Left', 'Right'], inplace=True)
# edges_df = edges_df.astype('int8')
# nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/nodes.csv', sep=',', index_col=0)
# nodes_df = nodes_df.astype('int8')
# time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/time_invariant_attr.csv', sep=',', index_col=0, dtype='str')
# time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/dblp/time_variant_attr_str.csv', sep=',', index_col=0, dtype='str')
# time_variant_attr.replace('0', 0, inplace=True)
# =============================================================================

# =============================================================================
# # school triangles
# edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/school/edges.csv', sep=',')
# edges_df.set_index(['Left', 'Right'], inplace=True)
# nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/school/nodes.csv', sep=',', index_col=0)
# time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/triangles_datasets/undirected/school/time_invariant_attr.csv', sep=',', index_col=0)
# =============================================================================


stc_attrs = list(time_invariant_attr.columns)
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


#################

# Evolution for Primary School triangles

#inx with inx 
inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,['2','3'])

diff_no,tia_d_no = Diff_Static(nodes_df,edges_df,time_invariant_attr,['3'],['2'])
diff_on,tia_d_on = Diff_Static(nodes_df,edges_df,time_invariant_attr,['2'],['3'])
# intersection aggregation
gdi = Aggregate_Static_Dist(inx,tia_inx,['gender'])
# diff aggregation
# fst-scd
ddi_no = Aggregate_Static_Dist(diff_no,tia_d_no,['gender'])
diff_agg_no = Diff_Post_Agg_Stc(ddi_no,['gender'])
# scd-fst
ddi_on = Aggregate_Static_Dist(diff_on,tia_d_on,['gender'])
diff_agg_on = Diff_Post_Agg_Stc(ddi_on,['gender'])
# evolution nodes
evol_nodes = pd.concat([gdi[0],diff_agg_on[0],diff_agg_no[0]], axis=1)
evol_nodes.columns = ['stable', 'deleted', 'new']
evol_nodes.index.names = ['gender']
evol_edges = pd.concat([gdi[1],diff_agg_on[1],diff_agg_no[1]], axis=1)
evol_edges.columns = ['stable', 'deleted', 'new']

# remove unspecified gender: U
# nodes
nodes_idx = []
for i in evol_nodes.index:
    if 'U' not in i:
        nodes_idx.append(i)
evol_nodes = evol_nodes.loc[nodes_idx,:]
nodes_percent = (evol_nodes.T/evol_nodes.T.sum()).T
nodes_percent = nodes_percent.applymap("{0:.1%}".format)
nodes_weight = evol_nodes.T.sum()
# edges
edges_idx = []
for i in evol_edges.index:
    if 'U' not in i[0] and 'U' not in i[1]:
        edges_idx.append(i)
evol_edges = evol_edges.loc[edges_idx,:]

edges_percent = (evol_edges.T/evol_edges.T.sum()).T
edges_percent = edges_percent.applymap("{0:.1%}".format)
edges_weight = evol_edges.T.sum()



##############################################################################

# SCHOOL
# EXPERIMENTS
# TPs static
result = []
for j in range(20):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        ne = [n,e]
        agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['gender'])
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


result = []
for j in range(20):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        ne = [n,e]
        agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['class'])
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


result = []
for j in range(20):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        ne = [n,e]
        agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['gender','class'])
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Union Static
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        un,tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Intersection Static
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# OLD - NEW
# Difference Static
result = []
for j in range(3):
    start_end_proj = []
    for i in range(len(intervals_diff)):
        intvl_fst = intervals_diff[i]
        intvl_scd = [intvl[i+1]]
        start = time.time()
        diff,tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# NEW - OLD
# Difference Static
result = []
for j in range(3):
    start_end_proj = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        start = time.time()
        diff,tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



######## UNION
# G_ALL_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_All(un,tia_un,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# G_DIST_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(un,tia_un,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# C_ALL_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_All(un,tia_un,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# C_DIST_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(un,tia_un,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



######## INTERSECTION

# G_DIST_I
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# P_DIST_I
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



######## DIFFERENCE

# OLD - NEW
# G_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = intervals_diff[i]
        intvl_scd = [intvl[i+1]]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# G_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = intervals_diff[i]
        intvl_scd = [intvl[i+1]]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# G_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# G_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# OLD - NEW
# G_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = intervals_diff[i]
        intvl_scd = [intvl[i+1]]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['class'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# G_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = intervals_diff[i]
        intvl_scd = [intvl[i+1]]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['class'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# G_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['class'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# G_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['class'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



######## EFFICIENT
# EFFICIENT Union
# G
agg_G_tps = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
    nodes_edges_tp = [nodes_tp,edges_tp]
    agg = Aggregate_Static_All(nodes_edges_tp,tia_tp,stc_attrs=['gender'])
    agg_G_tps.setdefault(i, []).append(agg)

result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        start = time.time()
        un_agg_eff = Union_Eff(agg_G_tps,interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# C
agg_C_tps = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
    nodes_edges_tp = [nodes_tp,edges_tp]
    agg = Aggregate_Static_All(nodes_edges_tp,tia_tp,stc_attrs=['class'])
    agg_C_tps.setdefault(i, []).append(agg)

result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        start = time.time()
        un_agg_eff = Union_Eff(agg_C_tps,interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# EFFICIENT DIMS | give dimension(s) as subset of the standard aggregation used

# GC
agg_GC_tp = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
    nodes_edges_tp = [nodes_tp,edges_tp] 
    agg = Aggregate_Static_Dist(nodes_edges_tp,tia_tp,stc_attrs=['gender','class'])
    agg_GC_tp.setdefault(i, []).append(agg)

# G (GC)
result = []
dims = ['gender']
for j in range(3):
    start_end_aggr = []
    for k,v in agg_GC_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)

# C (GC)
result = []
dims = ['class']
for j in range(3):
    start_end_aggr = []
    for k,v in agg_GC_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


##############################################################################


# DBLP TRIANGLES
# EXPERIMENTS
# TPs static
result = []
for j in range(3):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        ne = [n,e]
        del e
        del n
        agg = Aggregate_Static_Dist(ne,tia,stc_attrs)
        del ne
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# TPs variant
result = []
for j in range(3):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tva = time_variant_attr[y][time_variant_attr[y].index.isin(n.index)].to_frame()
        ne = [n,e]
        del e
        del n
        agg = Aggregate_Variant_All(ne,tva,[y])
        del ne
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# TPs mix
result = []
for j in range(3):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        tva = time_variant_attr[y][time_variant_attr[y].index.isin(n.index)].to_frame()
        ne = [n,e]
        del e
        del n
        agg = Aggregate_Mix_All(ne,tva,tia,stc_attrs,[y])
        del ne
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# Intersection Static
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Intersection Variant
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        un,tva_un = Intersection_Variant(nodes_df,edges_df,time_variant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# NEW - OLD
# Difference Static
result = []
for j in range(3):
    start_end_proj = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        start = time.time()
        diff,tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Difference Var
result = []
for j in range(3):
    start_end_proj = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        start = time.time()
        diff, tva_d = Diff_Variant(nodes_df,edges_df,time_variant_attr,intvl_fst,intvl_scd)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


######## INTERSECTION

# G_DIST_I
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# P_DIST_I
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        inx,tva_inx = Intersection_Variant(nodes_df,edges_df,time_variant_attr,interval)
        start = time.time()
        agg = Aggregate_Variant_Dist(inx,tva_inx,interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# DIFF

# NEW - OLD
# G_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# G_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['gender'])
        diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# P_DIST_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tva_d = Diff_Variant(nodes_df,edges_df,time_variant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Variant_Dist(diff,tva_d,intvl_fst)
        diff_agg = Diff_Post_Agg_Var(diff_agg)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# NEW - OLD
# P_ALL_D
result = []
for j in range(3):
    start_end_aggr = []
    for i in range(len(intervals_diff)):
        intvl_fst = [intvl[i+1]]
        intvl_scd = intervals_diff[i]
        diff, tva_d = Diff_Variant(nodes_df,edges_df,time_variant_attr,intvl_fst,intvl_scd)
        if diff[1].empty:
            start_end_aggr.append(0)
            continue
        start = time.time()
        agg = Aggregate_Variant_Dist(diff,tva_d,intvl_fst)
        diff_agg = Diff_Post_Agg_Var(diff_agg)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


######## EFFICIENT

# EFFICIENT DIMS | give dimension(s) as subset of the standard aggregation used

# GP
agg_GP_tp = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tva_tp = time_variant_attr[i][time_variant_attr[i]!=0].to_frame()
    tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
    nodes_edges_tp = [nodes_tp,edges_tp]
    agg = Aggregate_Mix_All(nodes_edges_tp,tva_tp,tia_tp,stc_attrs,[i])
    agg_GP_tp.setdefault(i, []).append(agg)

# G
# from GP All aggregation to G
result = []
dims = ['gender']
for j in range(20):
    start_end_aggr = []
    for k,v in agg_GP_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)

# P
# from GP All aggregation to P
result = []
dims = ['variant']
for j in range(20):
    start_end_aggr = []
    for k,v in agg_GP_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Union Static
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        un,tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# Union Variant
result = []
for j in range(3):
    start_end_proj = []
    for interval in intervals:
        start = time.time()
        un,tva_un = Union_Variant(nodes_df,edges_df,time_variant_attr,interval)
        end = time.time()
        start_end_proj.append(end-start)
    result.append(start_end_proj)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



######## UNION
# G_ALL_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_All(un,tia_un,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# G_DIST_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
        start = time.time()
        agg = Aggregate_Static_Dist(un,tia_un,stc_attrs=['gender'])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# P_ALL_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un,tva_un = Union_Variant(nodes_df,edges_df,time_variant_attr,interval)
        start = time.time()
        agg = Aggregate_Variant_All(un,tva_un,interval)
        end = time.time()
        del un
        del tva_un
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# P_DIST_U
result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        un,tva_un = Union_Variant(nodes_df,edges_df,time_variant_attr,interval)
        start = time.time()
        agg = Aggregate_Variant_Dist(un,tva_un,interval)
        end = time.time()
        del un
        del tva_un
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)



# EFFICIENT Union
# G
agg_G_tps = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
    nodes_edges_tp = [nodes_tp,edges_tp]
    agg = Aggregate_Static_All(nodes_edges_tp,tia_tp,stc_attrs=['gender'])
    agg_G_tps.setdefault(i, []).append(agg)

result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        start = time.time()
        un_agg_eff = Union_Eff(agg_G_tps,interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# P
agg_P_tps = {}
for i in intvl:
    nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
    edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
    tva_tp = time_variant_attr[i][time_variant_attr[i].index.isin(nodes_tp.index)].to_frame()
    nodes_edges_tp = [nodes_tp,edges_tp]
    agg = Aggregate_Variant_All(nodes_edges_tp,tva_tp,[i])
    agg_P_tps.setdefault(i, []).append(agg)

result = []
for j in range(3):
    start_end_aggr = []
    for interval in intervals:
        start = time.time()
        un_agg_eff = Union_Eff(agg_P_tps,interval)
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)


# DBLP EXPERIMENS

# TPs static
result = []
for j in range(3):
    start_end_agg = []
    for y in intvl:
        start = time.time()
        n = nodes_df[y][nodes_df[y]!=0].to_frame()
        e = edges_df[y][edges_df[y]!=0].to_frame()
        tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
        ne = [n,e]
        agg = Aggregate_Static_Dist(ne,tia,stc_attrs)
        end = time.time()
        start_end_agg.append(end-start)
    result.append(start_end_agg)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)

