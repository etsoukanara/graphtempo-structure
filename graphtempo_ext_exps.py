
import pandas as pd
import networkx as nx
import itertools
import copy
import time
from pyvis.network import Network
import pathlib
from functools import reduce
import numpy as np

# school
edges_df = pd.read_csv('school_dataset/edges.csv', sep=' ')
edges_df.set_index(['Left', 'Right'], inplace=True)
nodes_df = pd.read_csv('school_dataset/nodes.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('school_dataset/time_invariant_attr.csv', sep=' ', index_col=0)

stc_attrs = list(time_invariant_attr.columns)
intvl = list(edges_df.columns)

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

######## UNION STATIC
def Union_Static(nodesdf,edgesdf,tia,intvl):
    # get union of nodes and edges on interval
    nodes_u = nodesdf[intvl][nodesdf[intvl].any(axis=1)]
    edges_u = edgesdf[intvl][edgesdf[intvl].any(axis=1)]
    tia_u = tia[tia.index.isin(nodes_u.index)]
    un = [nodes_u,edges_u]
    return(un,tia_u)

######## UNION VARIANT
def Union_Variant(nodesdf,edgesdf,tva,intvl):
    # get union of nodes and edges on interval
    nodes_u = nodesdf[intvl][nodesdf[intvl].any(axis=1)]
    edges_u = edgesdf[intvl][edgesdf[intvl].any(axis=1)]
    tva_u = tva[intvl][tva[intvl].index.isin(nodes_u.index)]
    un = [nodes_u,edges_u]
    return(un,tva_u)

######## UNION STATIC-VARIANT
def Union_StcVar(nodesdf,edgesdf,tia,tva,intvl):
    # get union of nodes and edges on interval
    nodes_u = nodesdf[intvl][nodesdf[intvl].any(axis=1)]
    edges_u = edgesdf[intvl][edgesdf[intvl].any(axis=1)]
    tia_u = tia[tia.index.isin(nodes_u.index)]
    tva_u = tva[intvl][tva[intvl].index.isin(nodes_u.index)]
    un = [nodes_u,edges_u]
    return(un,tia_u,tva_u)

######## INTERSECTION STATIC
def Intersection_Static(nodesdf,edgesdf,tia,intvl):
    # get intersection of nodes and edges on interval
    nodes_inx = nodesdf[intvl][nodesdf[intvl].all(axis=1)]
    edges_inx = edgesdf[intvl][edgesdf[intvl].all(axis=1)]
    tia_inx = tia[tia.index.isin(nodes_inx.index)]
    inx = [nodes_inx,edges_inx]
    return(inx,tia_inx)


def Intersection_Static_New(nodesdf,edgesdf,tia,intvls):
    nodes_total = []
    edges_total = []
    for i in intvls:
        un, tia_un = Union_Static(nodesdf,edgesdf,tia,i)
        nodes_total.append(un[0])
        edges_total.append(un[1])
    nodes = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),nodes_total)
    edges = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),edges_total)
    tia_inx = tia_un[tia_un.index.isin(nodes.index)]
    inx = [nodes, edges]
    return(inx,tia_inx)


######## INTERSECTION VARIANT
def Intersection_Variant(nodesdf,edgesdf,tva,intvl):
    # get union of nodes and edges on interval
    nodes_inx = nodesdf[intvl][nodesdf[intvl].all(axis=1)]
    edges_inx = edgesdf[intvl][edgesdf[intvl].all(axis=1)]
    tva_inx = tva[intvl][tva[intvl].index.isin(nodes_inx.index)]
    inx = [nodes_inx,edges_inx]
    return(inx,tva_inx)


def Intersection_Variant_New(nodesdf,edgesdf,tva,intvls):
    nodes_total = []
    edges_total = []
    tva_total = []
    for i in intvls:
        un, tva_un = Union_Variant(nodesdf,edgesdf,tva,i)
        nodes_total.append(un[0])
        edges_total.append(un[1])
        tva_total.append(tva_un)
    nodes = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),nodes_total)
    edges = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),edges_total)
    tva = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),tva_total)
    tva_inx = tva[tva.index.isin(nodes.index)]
    inx = [nodes, edges]
    return(inx,tva_inx)


######## INTERSECTION STATIC-VARIANT
def Intersection_StcVar(nodesdf,edgesdf,tia,tva,intvl):
    # get union of nodes and edges on interval
    nodes_inx = nodesdf[intvl][nodesdf[intvl].all(axis=1)]
    edges_inx = edgesdf[intvl][edgesdf[intvl].all(axis=1)]
    tia_inx = tia[tia.index.isin(nodes_inx.index)]
    tva_inx = tva[intvl][tva[intvl].index.isin(nodes_inx.index)]
    inx = [nodes_inx,edges_inx]
    return(inx,tia_inx,tva_inx)


def Intersection_StcVar_New(nodesdf,edgesdf,tia,tva,intvls):
    nodes_total = []
    edges_total = []
    tva_total = []
    for i in intvls:
        un, tia, tva_un = Union_StcVar(nodesdf,edgesdf,tia,tva,i)
        nodes_total.append(un[0])
        edges_total.append(un[1])
        tva_total.append(tva_un)
    nodes = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),nodes_total)
    edges = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),edges_total)
    tva = reduce(lambda df_left,df_right: pd.merge(df_left,df_right,left_index=True,right_index=True),tva_total)
    tia_inx = tia[tia.index.isin(nodes.index)]
    tva_inx = tva[tva.index.isin(nodes.index)]
    inx = [nodes, edges]
    return(inx,tia_inx,tva_inx)


######## DIFFERENCE STATIC
# difference is the nodes and edges that are present in I and not present in I'
# DIFF = t_i - UNION(I') / UNION(I') - t_i

def Diff_Static(nodesdf,edgesdf,tia,intvl_fst,intvl_scd):
    un_init, tia_init = Union_Static(nodesdf,edgesdf,tia,intvl_fst)
    un_to_rm, tia_to_rm = Union_Static(nodesdf,edgesdf,tia,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    ndiff_idx = set(nodes.index.values.tolist())
    diff_idx = {*ediff_idx,*ndiff_idx}
    tia_d = tia_init[tia_init.index.isin(diff_idx)]
    diff = [nodes,edges]
    return(diff,tia_d)

def Diff_Variant(nodesdf,edgesdf,tva,intvl_fst,intvl_scd):
    un_init, tva_init = Union_Variant(nodesdf,edgesdf,tva,intvl_fst)
    un_to_rm, tva_to_rm = Union_Variant(nodesdf,edgesdf,tva,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    ndiff_idx = set(nodes.index.values.tolist())
    diff_idx = {*ediff_idx,*ndiff_idx}
    tva_d = tva_init[tva_init.index.isin(diff_idx)]
    diff = [nodes,edges]
    return(diff,tva_d)

def Diff_StcVar(nodesdf,edgesdf,tia,tva,intvl_fst,intvl_scd):
    un_init, tia_init, tva_init = Union_StcVar(nodesdf,edgesdf,tia,tva,intvl_fst)
    un_to_rm, tia_to_rm, tva_to_rm = Union_StcVar(nodesdf,edgesdf,tia,tva,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    ndiff_idx = set(nodes.index.values.tolist())
    diff_idx = {*ediff_idx,*ndiff_idx}
    tia_d = tia_init[tia_init.index.isin(diff_idx)]
    tva_d = tva_init[tva_init.index.isin(diff_idx)]
    diff = [nodes,edges]
    return(diff,tia_d,tva_d)

def Diff_Post_Agg_Stc(agg,stc_attrs):
    n_df = agg[0]
    e_df = agg[1]
    e_dfnew = e_df.reset_index().drop('count', axis=1)
    eL_df = e_dfnew.iloc[:,:len(stc_attrs)]
    eR_df = e_dfnew.iloc[:,len(stc_attrs):]    
    eLR_df = pd.DataFrame(eL_df.values.tolist() + eR_df.values.tolist()).drop_duplicates()
    eLR_df = eLR_df.set_index(eLR_df.columns.values.tolist())
    n_df = pd.concat([n_df, eLR_df], axis=1).fillna(0)
    diff_agg = [n_df,e_df]
    return(diff_agg)

def Diff_Post_Agg_Var(agg):
    n_df = agg[0]
    e_df = agg[1]
    e_dfnew = e_df.reset_index().drop('count', axis=1)
    eL_df = e_dfnew.iloc[:,0]
    eR_df = e_dfnew.iloc[:,1]    
    eLR_df = pd.DataFrame(eL_df.values.tolist() + eR_df.values.tolist()).drop_duplicates()
    eLR_df = eLR_df.set_index(eLR_df.columns.values.tolist())
    n_df = pd.concat([n_df, eLR_df], axis=1).fillna(0)
    diff_agg = [n_df,e_df]
    return(diff_agg)

def Diff_Post_Agg_Mix(agg,stc_attrs):
    n_df = agg[0]
    e_df = agg[1]
    e_dfnew = e_df.reset_index().drop('count', axis=1)
    eL_df = e_dfnew.iloc[:,:len(stc_attrs)+1]
    eR_df = e_dfnew.iloc[:,len(stc_attrs)+1:]    
    eLR_df = pd.DataFrame(eL_df.values.tolist() + eR_df.values.tolist()).drop_duplicates()
    eLR_df = eLR_df.set_index(eLR_df.columns.values.tolist())
    n_df = pd.concat([n_df, eLR_df], axis=1).fillna(0)
    diff_agg = [n_df,e_df]
    return(diff_agg)

################ AGGREGATION

def Aggregate_Static_All(oper_output,tia,stc_attrs):
    # nodes
    nodes = pd.DataFrame(oper_output[0].sum(axis=1), columns=['count'])
    for attr in stc_attrs:
        nodes[attr] = tia.loc[nodes.index,attr].values
    nodes = nodes.set_index(nodes.columns.values[1:].tolist())
    nodes = nodes.groupby(nodes.index.names).sum()
    # edges
    edges = pd.DataFrame(oper_output[1].sum(axis=1), columns=['count'])
    for attr in stc_attrs:
        edges[attr+'L'] = tia.loc[edges.index.get_level_values('Left'),attr].values
    for attr in stc_attrs:
        edges[attr+'R'] = tia.loc[edges.index.get_level_values('Right'),attr].values
    edges = edges.set_index(edges.columns.values[1:].tolist())
    edges = edges.groupby(edges.index.names).sum()
    agg = [nodes, edges]
    return(agg)

def Aggregate_Static_Dist(oper_output,tia,stc_attrs):
    # nodes
    if oper_output[0].index.equals(tia.index):
        nodes = tia[stc_attrs].set_index(tia[stc_attrs].columns.values.tolist())
    else:#difference output produces different indexes for nodes and attributes
        nodes = pd.DataFrame(index=oper_output[0].index)
        for attr in stc_attrs:
            nodes[attr] = tia.loc[nodes.index,attr].values
        nodes = nodes.set_index(nodes.columns.values.tolist())
    nodes = nodes.groupby(nodes.index.names).size().to_frame('count')
    # edges
    edges = pd.DataFrame(index=oper_output[1].index)
    for attr in stc_attrs:
        edges[attr+'L'] = tia.loc[edges.index.get_level_values('Left'),attr].values
    for attr in stc_attrs:
        edges[attr+'R'] = tia.loc[edges.index.get_level_values('Right'),attr].values
    edges = edges.set_index(edges.columns.values.tolist())
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    agg = [nodes, edges]
    return(agg)


def Aggregate_Variant_All(oper_output,tva,intvl):
    # nodes
    if oper_output[0].index.equals(tva.index):
        nodes = pd.melt(tva, value_name='variant', ignore_index=False).drop('variable', axis=1)
    else:
        nodes = pd.DataFrame(index=oper_output[0].index)
        for i in intvl:
            nodes[i] = tva.loc[nodes.index,i].values
        nodes = pd.melt(nodes, value_name='variant', ignore_index=False).drop('variable', axis=1)
    nodes = nodes[nodes.variant!=0]
    nodes = nodes.set_index(nodes.columns.values.tolist())
    nodes = nodes.groupby(nodes.index.names).size().to_frame('count')
    nodes = nodes[nodes['count'] != 0]
    # edges
    edges = pd.DataFrame(index=oper_output[1].index)
    for i in intvl:
        edges[i+'L'] = tva.loc[edges.index.get_level_values('Left'),i].values
        edges[i+'R'] = tva.loc[edges.index.get_level_values('Right'),i].values
    colnames = edges.columns.values.tolist()
    lefts = [colnames[i] for i in range(0,len(colnames),2)]
    rights = [colnames[i] for i in range(1,len(colnames),2)]
    edges_lefts = edges[lefts]
    edges_rights = edges[rights]
    edges_lefts = pd.melt(edges_lefts, value_name='variantL', ignore_index=False).drop('variable', axis=1)
    edges_rights = pd.melt(edges_rights, value_name='variantR', ignore_index=False).drop('variable', axis=1)
    edgelr = pd.concat([edges_lefts,edges_rights], axis=1)
    edges = edgelr.loc[~(edgelr==0).any(axis=1)]
    edges = edges.set_index(edges.columns.values.tolist())
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    edges = edges[edges['count'] != 0]
    agg = [nodes, edges]
    return(agg)


def Aggregate_Variant_Dist(oper_output,tva,intvl):
    # nodes
    if oper_output[0].index.equals(tva.index):
        nodes = pd.melt(tva, value_name='variant', ignore_index=False).drop('variable', axis=1)
    else:
        nodes = pd.DataFrame(index=oper_output[0].index)
        for i in intvl:
            nodes[i] = tva.loc[nodes.index,i].values
        nodes = pd.melt(nodes, value_name='variant', ignore_index=False).drop('variable', axis=1)
    nodes = nodes[nodes.variant!=0]
    nodes = nodes.reset_index()
    nodes.columns = ['userID', 'variant']
    nodes = nodes.drop_duplicates()
    nodes = nodes.set_index('userID')
    nodes = nodes.set_index(nodes.columns.values.tolist())
    nodes = nodes.groupby(nodes.index.names).size().to_frame('count')
    nodes = nodes[nodes['count'] != 0]
    # edges
    edges = pd.DataFrame(index=oper_output[1].index)
    for i in intvl:
        edges[i+'L'] = tva.loc[edges.index.get_level_values('Left'),i].values
        edges[i+'R'] = tva.loc[edges.index.get_level_values('Right'),i].values
    colnames = edges.columns.values.tolist()
    lefts = [colnames[i] for i in range(0,len(colnames),2)]
    rights = [colnames[i] for i in range(1,len(colnames),2)]
    edges_lefts = edges[lefts]
    edges_rights = edges[rights]
    edges_lefts = pd.melt(edges_lefts, value_name='variantL', ignore_index=False).drop('variable', axis=1)
    edges_rights = pd.melt(edges_rights, value_name='variantR', ignore_index=False).drop('variable', axis=1)
    edgelr = pd.concat([edges_lefts,edges_rights], axis=1)
    edges = edgelr.loc[~(edgelr==0).any(axis=1)]
    edges = edges.reset_index()
    edges = edges.drop_duplicates()
    edges = edges.drop(['Left', 'Right'], axis=1)
    edges = edges.set_index(edges.columns.values.tolist())
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    edges = edges[edges['count'] != 0]
    agg = [nodes, edges]
    return(agg)

def Aggregate_Mix_All(oper_output,tva,tia,stc_attrs,intvl):
    # nodes
    if oper_output[0].index.equals(tva.index):
        nodes = pd.melt(tva, value_name='variant', ignore_index=False).drop('variable', axis=1)
    else:
        nodes = pd.DataFrame(index=oper_output[0].index)
        for i in intvl:
            nodes[i] = tva.loc[nodes.index,i].values
        nodes = pd.melt(nodes, value_name='variant', ignore_index=False).drop('variable', axis=1)
    nodes = nodes[nodes.variant!=0]
    for attr in stc_attrs:
        nodes[attr] = tia.loc[nodes.index,attr].values
    nodes = nodes.set_index(nodes.columns.values.tolist())
    nodes = nodes.groupby(nodes.index.names).size().to_frame('count')
    nodes = nodes[nodes['count'] != 0]
    # edges
    edges = pd.DataFrame(index=oper_output[1].index)
    for i in intvl:
        edges[i+'L'] = tva.loc[edges.index.get_level_values('Left'),i].values
        edges[i+'R'] = tva.loc[edges.index.get_level_values('Right'),i].values
    colnames = edges.columns.values.tolist()
    lefts = [colnames[i] for i in range(0,len(colnames),2)]
    rights = [colnames[i] for i in range(1,len(colnames),2)]
    edges_lefts = edges[lefts]
    edges_rights = edges[rights]
    edges_lefts = pd.melt(edges_lefts, value_name='variantL', ignore_index=False).drop('variable', axis=1)
    edges_rights = pd.melt(edges_rights, value_name='variantR', ignore_index=False).drop('variable', axis=1)
    edgelr = pd.concat([edges_lefts,edges_rights], axis=1)
    edges = edgelr.loc[~(edgelr==0).any(axis=1)]
    for attr in stc_attrs:
        colslen = len(edges.columns)
        edges.insert(loc=colslen-1, column=attr+'L', \
            value=tia.loc[edges.index.get_level_values('Left'),attr].values)
    for attr in stc_attrs:
        colslen = len(edges.columns)
        edges.insert(loc=colslen, column=attr+'R', \
            value=tia.loc[edges.index.get_level_values('Right'),attr].values)
    edges = edges.set_index(edges.columns.values.tolist())
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    edges = edges[edges['count'] != 0]
    agg = [nodes, edges]
    return(agg)

def Aggregate_Mix_Dist(oper_output,tva,tia,stc_attrs,intvl):
    # nodes
    if oper_output[0].index.equals(tva.index):
        nodes = pd.melt(tva, value_name='variant', ignore_index=False).drop('variable', axis=1)
    else:
        nodes = pd.DataFrame(index=oper_output[0].index)
        for i in intvl:
            nodes[i] = tva.loc[nodes.index,i].values
        nodes = pd.melt(nodes, value_name='variant', ignore_index=False).drop('variable', axis=1)
    nodes = nodes[nodes.variant!=0]
    nodes = nodes.reset_index()
    nodes.columns = ['userID', 'variant']
    nodes = nodes.drop_duplicates()
    nodes = nodes.set_index('userID')
    for attr in stc_attrs:
        nodes[attr] = tia.loc[nodes.index,attr].values
    nodes = nodes.set_index(nodes.columns.values.tolist())
    nodes = nodes.groupby(nodes.index.names).size().to_frame('count')
    nodes = nodes[nodes['count'] != 0]
    # edges
    edges = pd.DataFrame(index=oper_output[1].index)
    for i in intvl:
        edges[i+'L'] = tva.loc[edges.index.get_level_values('Left'),i].values
        edges[i+'R'] = tva.loc[edges.index.get_level_values('Right'),i].values
    colnames = edges.columns.values.tolist()
    lefts = [colnames[i] for i in range(0,len(colnames),2)]
    rights = [colnames[i] for i in range(1,len(colnames),2)]
    edges_lefts = edges[lefts]
    edges_rights = edges[rights]
    edges_lefts = pd.melt(edges_lefts, value_name='variantL', ignore_index=False).drop('variable', axis=1)
    edges_rights = pd.melt(edges_rights, value_name='variantR', ignore_index=False).drop('variable', axis=1)
    edgelr = pd.concat([edges_lefts,edges_rights], axis=1)
    edges = edgelr.loc[~(edgelr==0).any(axis=1)]
    for attr in stc_attrs:
        colslen = len(edges.columns)
        edges.insert(loc=colslen-1, column=attr+'L', \
            value=tia.loc[edges.index.get_level_values('Left'),attr].values)
    for attr in stc_attrs:
        colslen = len(edges.columns)
        edges.insert(loc=colslen, column=attr+'R', \
            value=tia.loc[edges.index.get_level_values('Right'),attr].values)
    edges = edges.reset_index()
    edges = edges.drop_duplicates()
    edges = edges.drop(['Left', 'Right'], axis=1)
    edges = edges.set_index(edges.columns.values.tolist())
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    edges = edges[edges['count'] != 0]
    agg = [nodes, edges]
    return(agg)

############# EFFICIENCT

# EFFICIENT Union
def Union_Eff(agg_tp,intvl_un):
    nagg = pd.concat([agg_tp[i][0][0] for i in intvl_un],axis=1).sum(axis=1)
    eagg = pd.concat([agg_tp[i][0][1] for i in intvl_un],axis=1).sum(axis=1)
    agg = [nagg,eagg]
    return(agg)


# EFFICIENT DIMS | give dimension(s) as subset of the standard aggregation used
def Dims_Eff(dims,agg_std):
    edims = [d+i for i in ['L', 'R'] for d in dims]
    nagg = agg_std[0].groupby(level=dims).sum()
    eagg = agg_std[1].groupby(level=edims).sum()
    agg = [nagg,eagg]
    return(agg)



#################

# =============================================================================
# ##### EVOLUTION
# 
# #####
# ### fig. 12 | DBLP
# 
# # mix
# # 2010 to 2000's
# new = [intvl[10]]
# old = intvl[:10]
# new_old = set(new+old)
# 
# #inx with inx 
# inx,tia_inx,tva_inx = Intersection_StcVar(nodes_df,edges_df,time_invariant_attr,time_variant_attr,new+old)
# 
# intvl_new = new.copy()
# intvl_old = old.copy()
# diff_no,tia_d_no,tva_d_no = Diff_StcVar(nodes_df,edges_df,time_invariant_attr,time_variant_attr,intvl_new,intvl_old)
# diff_on,tia_d_on,tva_d_on = Diff_StcVar(nodes_df,edges_df,time_invariant_attr,time_variant_attr,intvl_old,intvl_new)
# # intersection aggregation
# gdi = Aggregate_Mix_Dist(inx,tva_inx,tia_inx,stc_attrs,new_old)
# # diff aggregation
# # fst-scd
# ddi_no = Aggregate_Mix_Dist(diff_no,tva_d_no,tia_d_no,stc_attrs,new)
# diff_agg_no = Diff_Post_Agg_Mix(ddi_no,stc_attrs)
# # scd-fst
# ddi_on = Aggregate_Mix_Dist(diff_on,tva_d_on,tia_d_on,stc_attrs,old)
# diff_agg_on = Diff_Post_Agg_Mix(ddi_on,stc_attrs)
# # evolution nodes
# evol_nodes = pd.concat([gdi[0],diff_agg_on[0],diff_agg_no[0]], axis=1)
# evol_nodes.columns = ['stable', 'deleted', 'new']
# evol_nodes.index.names = ['variant','gender']
# # evolution edges
# evol_edges = pd.concat([gdi[1],diff_agg_on[1],diff_agg_no[1]], axis=1)
# evol_edges.columns = ['stable', 'deleted', 'new']
# 
# # str time-varying attribute (>4, >4, >4)
# # nodes
# evol_nodes_dict = evol_nodes.to_dict()
# evol_nodes_dict_filtered = {}
# for key,val in evol_nodes_dict.items():
#     tmp = {}
#     for k,v in val.items():
#         str_split = k[0].split('_')
#         if int(str_split[0]) > 4 and int(str_split[1]) > 4 and int(str_split[2]) > 4:
#             tmp[k] = v
#     evol_nodes_dict_filtered[key] = tmp
# 
# evol_nodes_dict_filtered_df = pd.DataFrame(evol_nodes_dict_filtered)
# 
# evol_nodes_dict_filtered_df.index = evol_nodes_dict_filtered_df.index.droplevel(0)
# nodes = evol_nodes_dict_filtered_df.groupby(evol_nodes_dict_filtered_df.index).sum()
# 
# nodes_percent = (nodes.T/nodes.T.sum()).T
# nodes_percent = nodes_percent.applymap("{0:.1%}".format)
# nodes_weight = nodes.T.sum()
# 
# # edges
# evol_edges_dict = evol_edges.to_dict()
# evol_edges_dict_filtered = {}
# for key,val in evol_edges_dict.items():
#     tmp = {}
#     for k,v in val.items():
#         str_split1 = k[0].split('_')
#         str_split2 = k[2].split('_')
#         if int(str_split1[0]) > 4 and int(str_split1[1]) > 4 and int(str_split1[2]) > 4 and \
#             int(str_split2[0]) > 4 and int(str_split2[1]) > 4 and int(str_split2[2]) > 4:
#             tmp[k] = v
#     evol_edges_dict_filtered[key] = tmp
# 
# evol_edges_dict_filtered_df = pd.DataFrame(evol_edges_dict_filtered)
# 
# evol_edges_dict_filtered_df.index = evol_edges_dict_filtered_df.index.droplevel([0,2])
# edges = evol_edges_dict_filtered_df.groupby(evol_edges_dict_filtered_df.index).sum()
# 
# edges_percent = (edges.T/edges.T.sum()).T
# edges_percent = edges_percent.applymap("{0:.1%}".format)
# edges_weight = edges.T.sum()
# 
# ##########
# 
# # 2020 to 2010's
# new = [intvl[20]]
# old = intvl[10:20]
# new_old = set(new+old)
# # filter edges (to avoid memory error) / keep edges whose nodes have #publications > 4
# nodes_to_keep = []
# for col in nodes_df.columns:
#     for i in time_variant_attr.index:
#         if time_variant_attr.loc[i,col] != 0:
#             str_attr = time_variant_attr.loc[i,col].split('_')
#             if int(str_attr[0]) > 4 and int(str_attr[1]) > 4 and int(str_attr[2]) > 4:
#                 nodes_to_keep.append(i)
# nodes_to_keep = set(nodes_to_keep)
# 
# new_nodes_df = nodes_df.loc[nodes_to_keep]
# 
# edges_to_keep = []
# for i in edges_df.index:
#     if i[0] in nodes_to_keep and i[1] in nodes_to_keep:
#         edges_to_keep.append(i)
# new_edges_df = edges_df.loc[edges_to_keep]
# new_edges_df.sort_index(inplace=True)
# 
# nodes_to_keep = list(nodes_to_keep)
# 
# new_nodes_df = new_nodes_df.loc[nodes_to_keep]
# new_nodes_df.sort_index(inplace=True)
# new_time_invariant_attr = time_invariant_attr.loc[nodes_to_keep]
# new_time_invariant_attr.sort_index(inplace=True)
# new_time_variant_attr = time_variant_attr.loc[nodes_to_keep]
# new_time_variant_attr.sort_index(inplace=True)
# new_old = set(new+old)
# 
# #inx with inx 
# inx,tia_inx,tva_inx = Intersection_StcVar(new_nodes_df,new_edges_df,new_time_invariant_attr,new_time_variant_attr,new+old)
# 
# intvl_new = new.copy()
# intvl_old = old.copy()
# diff_no,tia_d_no,tva_d_no = Diff_StcVar(new_nodes_df,new_edges_df,new_time_invariant_attr,new_time_variant_attr,intvl_new,intvl_old)
# diff_on,tia_d_on,tva_d_on = Diff_StcVar(new_nodes_df,new_edges_df,new_time_invariant_attr,new_time_variant_attr,intvl_old,intvl_new)
# # intersection aggregation
# gdi = Aggregate_Mix_Dist(inx,tva_inx,tia_inx,stc_attrs,new_old)
# # diff aggregation
# # fst-scd
# ddi_no = Aggregate_Mix_Dist(diff_no,tva_d_no,tia_d_no,stc_attrs,new)
# diff_agg_no = Diff_Post_Agg_Mix(ddi_no,stc_attrs)
# # scd-fst
# ddi_on = Aggregate_Mix_Dist(diff_on,tva_d_on,tia_d_on,stc_attrs,old)
# diff_agg_on = Diff_Post_Agg_Mix(ddi_on,stc_attrs)
# # evolution nodes
# evol_nodes = pd.concat([gdi[0],diff_agg_on[0],diff_agg_no[0]], axis=1)
# evol_nodes.columns = ['stable', 'deleted', 'new']
# evol_nodes.index.names = ['variant','gender']
# # evolution edges
# evol_edges = pd.concat([gdi[1],diff_agg_on[1],diff_agg_no[1]], axis=1)
# evol_edges.columns = ['stable', 'deleted', 'new']
# 
# # str time-varying attribute (>4, >4, >4)
# # nodes
# evol_nodes_dict = evol_nodes.to_dict()
# evol_nodes_dict_filtered = {}
# for key,val in evol_nodes_dict.items():
#     tmp = {}
#     for k,v in val.items():
#         str_split = k[0].split('_')
#         if int(str_split[0]) > 4 and int(str_split[1]) > 4 and int(str_split[2]) > 4:
#             tmp[k] = v
#     evol_nodes_dict_filtered[key] = tmp
# 
# evol_nodes_dict_filtered_df = pd.DataFrame(evol_nodes_dict_filtered)
# 
# evol_nodes_dict_filtered_df.index = evol_nodes_dict_filtered_df.index.droplevel(0)
# nodes = evol_nodes_dict_filtered_df.groupby(evol_nodes_dict_filtered_df.index).sum()
# 
# nodes_percent = (nodes.T/nodes.T.sum()).T
# nodes_percent = nodes_percent.applymap("{0:.1%}".format)
# nodes_weight = nodes.T.sum()
# 
# # edges
# evol_edges_dict = evol_edges.to_dict()
# evol_edges_dict_filtered = {}
# for key,val in evol_edges_dict.items():
#     tmp = {}
#     for k,v in val.items():
#         str_split1 = k[0].split('_')
#         str_split2 = k[2].split('_')
#         if int(str_split1[0]) > 4 and int(str_split1[1]) > 4 and int(str_split1[2]) > 4 and \
#             int(str_split2[0]) > 4 and int(str_split2[1]) > 4 and int(str_split2[2]) > 4:
#             tmp[k] = v
#     evol_edges_dict_filtered[key] = tmp
# 
# evol_edges_dict_filtered_df = pd.DataFrame(evol_edges_dict_filtered)
# 
# evol_edges_dict_filtered_df.index = evol_edges_dict_filtered_df.index.droplevel([0,2])
# edges = evol_edges_dict_filtered_df.groupby(evol_edges_dict_filtered_df.index).sum()
# 
# edges_percent = (edges.T/edges.T.sum()).T
# edges_percent = edges_percent.applymap("{0:.1%}".format)
# edges_weight = edges.T.sum()
# =============================================================================



# =============================================================================
# evol_nodes_rcd = evol_nodes.query('variant > 10')
# evol_edges_rcd = evol_edges.query('variantL > 10 and variantR > 10')
# evol_nodes_rcd.index = evol_nodes_rcd.index.droplevel(0)
# evol_edges_rcd.index = evol_edges_rcd.index.droplevel([0,2])
# nodes = evol_nodes_rcd.groupby(evol_nodes_rcd.index).sum()
# edges = evol_edges_rcd.groupby(evol_edges_rcd.index).sum()
# 
# nodes_percent = (nodes.T/nodes.T.sum()).T
# nodes_percent = nodes_percent.applymap("{0:.1%}".format)
# edges_percent = (edges.T/edges.T.sum()).T
# edges_percent = edges_percent.applymap("{0:.1%}".format)
# 
# nodes_weight = nodes.T.sum()
# edges_weight = edges.T.sum()
# =============================================================================






##############################################################################

# =============================================================================
# # SCHOOL
# # EXPERIMENTS
# # TPs static
# result = []
# for j in range(3):
#     start_end_agg = []
#     for y in intvl:
#         start = time.time()
#         n = nodes_df[y][nodes_df[y]!=0].to_frame()
#         e = edges_df[y][edges_df[y]!=0].to_frame()
#         tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
#         ne = [n,e]
#         agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['gender'])
#         end = time.time()
#         start_end_agg.append(end-start)
#     result.append(start_end_agg)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/tps/G_dist.csv', columns = ['avg'])
# 
# result = []
# for j in range(3):
#     start_end_agg = []
#     for y in intvl:
#         start = time.time()
#         n = nodes_df[y][nodes_df[y]!=0].to_frame()
#         e = edges_df[y][edges_df[y]!=0].to_frame()
#         tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
#         ne = [n,e]
#         agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['class'])
#         end = time.time()
#         start_end_agg.append(end-start)
#     result.append(start_end_agg)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/tps/C_dist.csv', columns = ['avg'])
# 
# result = []
# for j in range(3):
#     start_end_agg = []
#     for y in intvl:
#         start = time.time()
#         n = nodes_df[y][nodes_df[y]!=0].to_frame()
#         e = edges_df[y][edges_df[y]!=0].to_frame()
#         tia = time_invariant_attr[time_invariant_attr.index.isin(n.index)]
#         ne = [n,e]
#         agg = Aggregate_Static_Dist(ne,tia,stc_attrs=['gender','class'])
#         end = time.time()
#         start_end_agg.append(end-start)
#     result.append(start_end_agg)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/tps/GC_dist.csv', columns = ['avg'])
# 
# # Union Static
# result = []
# for j in range(3):
#     start_end_proj = []
#     for interval in intervals:
#         start = time.time()
#         un,tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         end = time.time()
#         start_end_proj.append(end-start)
#     result.append(start_end_proj)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/su.csv', columns = ['avg'])
# 
# # Intersection Static
# result = []
# for j in range(3):
#     start_end_proj = []
#     for interval in intervals:
#         start = time.time()
#         inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         end = time.time()
#         start_end_proj.append(end-start)
#     result.append(start_end_proj)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/si.csv', columns = ['avg'])
# 
# # OLD - NEW
# # Difference Static
# result = []
# for j in range(3):
#     start_end_proj = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = intervals_diff[i]
#         intvl_scd = [intvl[i+1]]
#         start = time.time()
#         diff,tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         end = time.time()
#         start_end_proj.append(end-start)
#     result.append(start_end_proj)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/sd_old_new.csv', columns = ['avg'])
# 
# # NEW - OLD
# # Difference Static
# result = []
# for j in range(3):
#     start_end_proj = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = [intvl[i+1]]
#         intvl_scd = intervals_diff[i]
#         start = time.time()
#         diff,tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         end = time.time()
#         start_end_proj.append(end-start)
#     result.append(start_end_proj)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/sd_new_old.csv', columns = ['avg'])
# 
# 
# ######## UNION
# # G_ALL_U
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_All(un,tia_un,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GAU.csv', columns = ['avg'])
# 
# # G_DIST_U
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_Dist(un,tia_un,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GDU.csv', columns = ['avg'])
# 
# # C_ALL_U
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_All(un,tia_un,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CAU.csv', columns = ['avg'])
# 
# # C_DIST_U
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         un, tia_un = Union_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_Dist(un,tia_un,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CDU.csv', columns = ['avg'])
# 
# 
# ######## INTERSECTION
# 
# # G_DIST_I
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GDI.csv', columns = ['avg'])
# 
# # P_DIST_I
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         inx, tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,interval)
#         start = time.time()
#         agg = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CDI.csv', columns = ['avg'])
# 
# 
# ######## DIFFERENCE
# 
# # OLD - NEW
# # G_DIST_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = intervals_diff[i]
#         intvl_scd = [intvl[i+1]]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['gender'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GDD_old_new.csv', columns = ['avg'])
# 
# 
# # G_ALL_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = intervals_diff[i]
#         intvl_scd = [intvl[i+1]]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['gender'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GAD_old_new.csv', columns = ['avg'])
# 
# 
# # NEW - OLD
# # G_DIST_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = [intvl[i+1]]
#         intvl_scd = intervals_diff[i]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['gender'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GDD_new_old.csv', columns = ['avg'])
# 
# 
# # NEW - OLD
# # G_ALL_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = [intvl[i+1]]
#         intvl_scd = intervals_diff[i]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['gender'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['gender'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/GAD_new_old.csv', columns = ['avg'])
# 
# 
# # OLD - NEW
# # G_DIST_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = intervals_diff[i]
#         intvl_scd = [intvl[i+1]]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['class'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CDD_old_new.csv', columns = ['avg'])
# 
# 
# # G_ALL_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = intervals_diff[i]
#         intvl_scd = [intvl[i+1]]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['class'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CAD_old_new.csv', columns = ['avg'])
# 
# 
# # NEW - OLD
# # G_DIST_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = [intvl[i+1]]
#         intvl_scd = intervals_diff[i]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs=['class'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CDD_new_old.csv', columns = ['avg'])
# 
# 
# # NEW - OLD
# # G_ALL_D
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for i in range(len(intervals_diff)):
#         intvl_fst = [intvl[i+1]]
#         intvl_scd = intervals_diff[i]
#         diff, tia_d = Diff_Static(nodes_df,edges_df,time_invariant_attr,intvl_fst,intvl_scd)
#         if diff[1].empty:
#             start_end_aggr.append(0)
#             continue
#         start = time.time()
#         agg = Aggregate_Static_All(diff,tia_d,stc_attrs=['class'])
#         diff_agg = Diff_Post_Agg_Stc(agg,stc_attrs=['class'])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/operators/CAD_new_old.csv', columns = ['avg'])
# 
# 
# ######## EFFICIENT
# # EFFICIENT Union
# # G
# agg_G_tps = {}
# for i in intvl:
#     nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
#     edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
#     tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
#     nodes_edges_tp = [nodes_tp,edges_tp]
#     agg = Aggregate_Static_All(nodes_edges_tp,tia_tp,stc_attrs=['gender'])
#     agg_G_tps.setdefault(i, []).append(agg)
# 
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         start = time.time()
#         un_agg_eff = Union_Eff(agg_G_tps,interval)
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/efficient/GAU_EFF.csv', columns = ['avg'])
# 
# # C
# agg_C_tps = {}
# for i in intvl:
#     nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
#     edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
#     tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
#     nodes_edges_tp = [nodes_tp,edges_tp]
#     agg = Aggregate_Static_All(nodes_edges_tp,tia_tp,stc_attrs=['class'])
#     agg_C_tps.setdefault(i, []).append(agg)
# 
# result = []
# for j in range(3):
#     start_end_aggr = []
#     for interval in intervals:
#         start = time.time()
#         un_agg_eff = Union_Eff(agg_C_tps,interval)
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# 
# res.to_csv('results/school/efficient/CAU_EFF.csv', columns = ['avg'])
# 
# 
# # EFFICIENT DIMS | give dimension(s) as subset of the standard aggregation used
# 
# # GC
# agg_GC_tp = {}
# for i in intvl:
#     nodes_tp = nodes_df[i][nodes_df[i]!=0].to_frame()
#     edges_tp = edges_df[i][edges_df[i]!=0].to_frame()
#     tia_tp = time_invariant_attr[time_invariant_attr.index.isin(nodes_tp.index)]
#     nodes_edges_tp = [nodes_tp,edges_tp] 
#     agg = Aggregate_Static_Dist(nodes_edges_tp,tia_tp,stc_attrs=['gender','class'])
#     agg_GC_tp.setdefault(i, []).append(agg)
# 
# # G (GC)
# result = []
# dims = ['gender']
# for j in range(3):
#     start_end_aggr = []
#     for k,v in agg_GC_tp.items():
#         start = time.time()
#         dim_agg_eff = Dims_Eff(dims,v[0])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# res.to_csv('results/school/efficient/G(GC)_dim.csv', columns = ['avg'])
# 
# # C (GC)
# result = []
# dims = ['class']
# for j in range(3):
#     start_end_aggr = []
#     for k,v in agg_GC_tp.items():
#         start = time.time()
#         dim_agg_eff = Dims_Eff(dims,v[0])
#         end = time.time()
#         start_end_aggr.append(end-start)
#     result.append(start_end_aggr)
# 
# res = pd.DataFrame(result).T
# res['avg'] = res.mean(axis=1)
# res.to_csv('results/school/efficient/C(GC)_dim.csv', columns = ['avg'])
# =============================================================================


##############################################################################


# TRIANGLES
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

res.to_csv('results/triangles/tps/G_dist.csv', columns = ['avg'])

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

res.to_csv('results/triangles/tps/P_all.csv', columns = ['avg'])

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

res.to_csv('results/triangles/tps/GP_all.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/si.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/vi.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/sd_new_old.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/vd_new_old.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/GDI.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/PDI.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/GDD_new_old.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/GAD_new_old.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/PDD_new_old.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/PAD_new_old.csv', columns = ['avg'])

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
for j in range(3):
    start_end_aggr = []
    for k,v in agg_GP_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)
res.to_csv('results/triangles/efficient/G_dim.csv', columns = ['avg'])

# P
# from GP All aggregation to P
result = []
dims = ['variant']
for j in range(3):
    start_end_aggr = []
    for k,v in agg_GP_tp.items():
        start = time.time()
        dim_agg_eff = Dims_Eff(dims,v[0])
        end = time.time()
        start_end_aggr.append(end-start)
    result.append(start_end_aggr)

res = pd.DataFrame(result).T
res['avg'] = res.mean(axis=1)
res.to_csv('results/triangles/efficient/P_dim.csv', columns = ['avg'])






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

res.to_csv('results/triangles/operators/su.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/vu.csv', columns = ['avg'])


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

res.to_csv('results/triangles/operators/GAU.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/GDU.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/PAU.csv', columns = ['avg'])

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

res.to_csv('results/triangles/operators/PDU.csv', columns = ['avg'])


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

res.to_csv('results/triangles/efficient/GAU_EFF.csv', columns = ['avg'])

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

res.to_csv('results/triangles/efficient/PAU_EFF.csv', columns = ['avg'])

