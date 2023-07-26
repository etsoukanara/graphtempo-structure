from graphtempo import *
import pandas as pd
import itertools
import copy
import plotly.express as px
import plotly.io as pio
pio.renderers.default='svg'
import plotly.graph_objects as go


# Stability | Inx Semantics
# a Old(INX)&New
def Stability_Intersection_Static_a(k,intvl,nodes_df,edges_df,invar,stc_attrs,values):
    intvl_rv = intvl[::-1]
    stabI_invl_a = []
    for i in intvl[:-1]:
        stabI_invl_a.append([[i],[intvl[intvl.index(i)+1]]])  
    stabI_a = []
    for i in stabI_invl_a:
        inx,tia_inx = Intersection_Static(nodes_df,edges_df,invar,i[0]+i[1])
        if inx[1].empty:
            continue
        else:
            agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
        try:
            attr_value = agg_inx[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            tmp = copy.deepcopy(i)
            while attr_value >= k and i[0][-1] != intvl_rv[-1]:
                tmp = copy.deepcopy(i)
                i[0].append(intvl_rv[intvl_rv.index(i[0][-1])+1])
                inx,tia_inx = Intersection_Static(nodes_df,edges_df,invar,i[0]+i[1])
                if inx[1].empty:
                    break
                agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
                try:
                    attr_value = agg_inx[1].loc[values][0]
                except:
                    attr_value = 0
                if attr_value >= k:
                    tmp = copy.deepcopy(i)
            stabI_a.append(tmp)
    stabI_a = [[i[::-1],j] for i,j in stabI_a]
    return(stabI_a,agg_inx)


def Stability_Intersection_Variant_a(k,intvl,nodes_df,edges_df,varying,values):
    intvl_rv = intvl[::-1]
    stabI_invl_a = []
    for i in intvl[:-1]:
        stabI_invl_a.append([[i],[intvl[intvl.index(i)+1]]])  
    stabI_a = []
    for i in stabI_invl_a:
        inx,tva_inx = Intersection_Variant(nodes_df,edges_df,varying,i[0]+i[1])
        if inx[1].empty:
            continue
        else:
            agg_inx = Aggregate_Variant_Dist(inx,tva_inx,i[0]+i[1])
        try:
            attr_value = agg_inx[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            tmp = copy.deepcopy(i)
            while attr_value >= k and i[0][-1] != intvl_rv[-1]:
                tmp = copy.deepcopy(i)
                i[0].append(intvl_rv[intvl_rv.index(i[0][-1])+1])
                inx,tva_inx = Intersection_Variant(nodes_df,edges_df,varying,i[0]+i[1])
                if inx[1].empty:
                    break
                agg_inx = Aggregate_Variant_Dist(inx,tva_inx,i[0]+i[1])
                try:
                    attr_value = agg_inx[1].loc[values][0]
                except:
                    attr_value = 0
                if attr_value >= k:
                    tmp = copy.deepcopy(i)
            stabI_a.append(tmp)
    stabI_a = [[i[::-1],j] for i,j in stabI_a]
    return(stabI_a,agg_inx)


def Stability_Intersection_Mix_a(k,intvl,nodes_df,edges_df,invar,varying,stc_attrs,values):
    intvl_rv = intvl[::-1]
    stabI_invl_a = []
    for i in intvl[:-1]:
        stabI_invl_a.append([[i],[intvl[intvl.index(i)+1]]])  
    stabI_a = []
    for i in stabI_invl_a:
        inx,tia_inx,tva_inx = Intersection_Mix(nodes_df,edges_df,invar,varying,i[0]+i[1])
        if inx[1].empty:
            continue
        else:
            agg_inx = Aggregate_Mix_Dist(inx,tva_inx,tia_inx,stc_attrs,i[0]+i[1])
        try:
            attr_value = agg_inx[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            tmp = copy.deepcopy(i)
            while attr_value >= k and i[0][-1] != intvl_rv[-1]:
                tmp = copy.deepcopy(i)
                i[0].append(intvl_rv[intvl_rv.index(i[0][-1])+1])
                inx,tia_inx,tva_inx = Intersection_Mix(nodes_df,edges_df,invar,varying,i[0]+i[1])
                if inx[1].empty:
                    break
                agg_inx = Aggregate_Mix_Dist(inx,tva_inx,tia_inx,stc_attrs,i[0]+i[1])
                try:
                    attr_value = agg_inx[1].loc[values][0]
                except:
                    attr_value = 0
                if attr_value >= k:
                    tmp = copy.deepcopy(i)
            stabI_a.append(tmp)      
    stabI_a = [[i[::-1],j] for i,j in stabI_a]
    return(stabI_a,agg_inx)


# Growth | Union Semantics
# a New-Old(UNION)
def Growth_Union_Static_a(k,intvl,nodes_df,edges_df,invar,stc_attrs,values):
    growth_invl_a = []
    for i in intvl[:-1]:
        growth_invl_a.append([[intvl[intvl.index(i)+1]],[i]])  
    growth_a = []
    for i in growth_invl_a:
        diff,tia_d = Diff_Static(nodes_df,edges_df,invar,i[0],i[1])
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs)
        diff_agg = Diff_Post_Agg_Static(agg,stc_attrs)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            growth_a.append(i)
    return(growth_a,diff_agg)


def Growth_Union_Variant_a(k,intvl,nodes_df,edges_df,values):
    growth_invl_a = []
    for i in intvl[:-1]:
        growth_invl_a.append([[intvl[intvl.index(i)+1]],[i]])  
    growth_a = []
    for i in growth_invl_a:
        diff,tva_d = Diff_Variant(nodes_df,edges_df,varying,i[0],i[1])
        agg = Aggregate_Variant_Dist(diff,tva_d,i[0])
        diff_agg = Diff_Post_Agg_Variant(agg)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            growth_a.append(i)
    return(growth_a,diff_agg)


def Growth_Union_Mix_a(k,intvl,nodes_df,edges_df,invar,varying,stc_attrs,values):
    growth_invl_a = []
    for i in intvl[:-1]:
        growth_invl_a.append([[intvl[intvl.index(i)+1]],[i]])  
    growth_a = []
    for i in growth_invl_a:
        diff,tia_d,tva_d = Diff_Mix(nodes_df,edges_df,invar,varying,i[0],i[1])
        agg = Aggregate_Mix_Dist(diff,tva_d,tia_d,stc_attrs,i[0])
        diff_agg = Diff_Post_Agg_Mix(agg,stc_attrs)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            growth_a.append(i)
    return(growth_a,diff_agg)


# Shrinkage | Union Semantics | Static
def Shrink_Union_Static_a(k,intvl,nodes_df,edges_df,invar,stc_attrs,values):
    shrink_invl_a = []
    for i in intvl[:-1]:
        shrink_invl_a.append([[i],[intvl[intvl.index(i)+1]]])
    shrink_invl_a.reverse()
    shrink_a = []
    for i in shrink_invl_a:
        diff,tia_d = Diff_Static(nodes_df,edges_df,invar,i[0],i[1])
        agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs)
        diff_agg = Diff_Post_Agg_Static(agg,stc_attrs)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            shrink_a.append(i)
        else:
            while attr_value < k and i[-1][0] != intvl[-1]:
                if intvl[intvl.index(i[0][-1])+1] not in [j[0][-1] for j in shrink_a] \
            and intvl[intvl.index(i[0][-1])+2] not in [j[1][0] for j in shrink_a]:
                    i[0].append(intvl[intvl.index(i[0][-1])+1])
                    i[1] = [intvl[intvl.index(i[0][-1])+1]]
                    diff,tia_d = Diff_Static(nodes_df,edges_df,invar,i[0],i[1])
                    agg = Aggregate_Static_Dist(diff,tia_d,stc_attrs)
                    diff_agg = Diff_Post_Agg_Static(agg,stc_attrs)
                    try:
                        attr_value = diff_agg[1].loc[values][0]
                    except:
                        attr_value = 0
                else:
                    break
            if attr_value >= k:
                shrink_a.append(i)
    return(shrink_a,diff_agg)
                    
def Shrink_Union_Variant_a(k,intvl,nodes_df,edges_df,varying,values):
    shrink_invl_a = []
    for i in intvl[:-1]:
        shrink_invl_a.append([[i],[intvl[intvl.index(i)+1]]])
    shrink_invl_a.reverse()
    shrink_a = []
    for i in shrink_invl_a:
        diff,tva_d = Diff_Variant(nodes_df,edges_df,varying,i[0],i[1])
        agg = Aggregate_Variant_Dist(diff,tva_d,i[0])
        diff_agg = Diff_Post_Agg_Variant(agg)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            shrink_a.append(i)
        else:
            while attr_value < k and i[-1][0] != intvl[-1]:
                if intvl[intvl.index(i[0][-1])+1] not in [j[0][-1] for j in shrink_a] \
            and intvl[intvl.index(i[0][-1])+2] not in [j[1][0] for j in shrink_a]:
                    i[0].append(intvl[intvl.index(i[0][-1])+1])
                    i[1] = [intvl[intvl.index(i[0][-1])+1]]
                    diff,tva_d = Diff_Variant(nodes_df,edges_df,varying,i[0],i[1])
                    agg = Aggregate_Variant_Dist(diff,tva_d,i[0])
                    diff_agg = Diff_Post_Agg_Variant(agg)
                    try:
                        attr_value = diff_agg[1].loc[values][0]
                    except:
                        attr_value = 0
                else:
                    break
            if attr_value >= k:
                shrink_a.append(i)
    return(shrink_a,diff_agg)

def Shrink_Union_Mix_a(k,intvl,nodes_df,edges_df,invar,varying,stc_attrs,values):
    shrink_invl_a = []
    for i in intvl[:-1]:
        shrink_invl_a.append([[i],[intvl[intvl.index(i)+1]]])
    shrink_invl_a.reverse()
    shrink_a = []
    for i in shrink_invl_a:
        diff,tia_d,tva_d = Diff_Mix(nodes_df,edges_df,invar,varying,i[0],i[1])
        agg = Aggregate_Mix_Dist(diff,tva_d,tia_d,stc_attrs,i[0])
        diff_agg = Diff_Post_Agg_Mix(agg,stc_attrs)
        try:
            attr_value = diff_agg[1].loc[values][0]
        except:
            attr_value = 0
        if attr_value >= k:
            shrink_a.append(i)
        else:
            while attr_value < k and i[-1][0] != intvl[-1]:
                if intvl[intvl.index(i[0][-1])+1] not in [j[0][-1] for j in shrink_a] \
            and intvl[intvl.index(i[0][-1])+2] not in [j[1][0] for j in shrink_a]:
                    i[0].append(intvl[intvl.index(i[0][-1])+1])
                    i[1] = [intvl[intvl.index(i[0][-1])+1]]
                    diff,tia_d,tva_d = Diff_Mix(nodes_df,edges_df,invar,varying,i[0],i[1])
                    agg = Aggregate_Mix_Dist(diff,tva_d,tia_d,stc_attrs,i[0])
                    diff_agg = Diff_Post_Agg_Mix(agg,stc_attrs)
                    try:
                        attr_value = diff_agg[1].loc[values][0]
                    except:
                        attr_value = 0
                else:
                    break
            if attr_value >= k:
                shrink_a.append(i)
    return(shrink_a,diff_agg)


# dblp dataset
stc_attrs = ['gender']
varying = ['#Publications']
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/nodes.csv', sep=' ', index_col=0)
time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_variant_attr.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_invariant_attr.rename(columns={'0': 'gender'}, inplace=True)
nodes_df.index.names = ['userID']
time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)
interval = [i for i in edges_df.columns]

# movielens dataset
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/edges.csv', sep=' ')
edges_df.set_index(['Left', 'Right'], inplace=True)
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/nodes.csv', sep=' ', index_col=0)
time_variant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/time_variant_attr.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.1_DEMO/GraphTempo_APP/datasets/movielens_dataset/time_invariant_attr.csv', sep=' ', index_col=0)

# =============================================================================
# # replace notation of attributes
# time_invariant_attr.gender.replace(['F','M'], [0,1],inplace=True)
# time_invariant_attr.occupation.replace(['other', 'academic/educator', 'artist', \
#         'clerical/admin', 'college/grad student', 'customer service', \
#         'doctor/health care', 'executive/managerial', 'farmer', 'homemaker', \
#         'K-12 student', 'lawyer', 'programmer', 'retired', 'sales/marketing', \
#         'scientist', 'self-employed', 'technician/engineer', \
#         'tradesman/craftsman', 'unemployed', 'writer'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, \
#         15, 16, 17, 18, 19, 20], inplace=True)
# time_invariant_attr.age.replace(['Under 18', '18-24', \
#                     '25-34', '35-44', '45-49', '50-55', '56+'], [1, 18, 25, 35, 45, 50, 56], inplace=True)
# =============================================================================

# school dataset
edges_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/edges.csv', sep=' ', index_col=[0,1])
nodes_df = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/nodes.csv', sep=' ', index_col=0)
time_invariant_attr = pd.read_csv('C:/Users/Lila/Desktop/PROJECT_2.2_EXT/experiments/school_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
nodes_df.index.names = ['userID']
interval = [i for i in edges_df.columns]

#domain
time_invariant_attr['gender'].value_counts()
time_invariant_attr['gender'].nunique()

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

# SKYLINE
# =============================================================================
# # maximal
# # Stability (extend old)
# 
# attr_values = ('F','F')
# stc_attrs = ['gender']
# 
# intvls = [[edges_df.columns[0],edges_df.columns[1]]]
# for i in range(len(edges_df.columns)-2):
#     intvls.append([j for j in intvls[i]]+[str(int(intvls[i][-1])+1)])
# intvls = [[i[:-1],[i[-1]]] for i in intvls]
# 
# buckets_stab = []
# for left,right in intvls:
#     result_last = 0
#     while len(left) >= 1:
#         inx,tia_inx = Intersection_Static(
#                                             nodes_df,
#                                             edges_df,
#                                             time_invariant_attr,
#                                             left+right
#                                             )
#         if not inx[1].empty:
#             agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
#             if attr_values in agg_inx[1].index:
#                 result_curr = agg_inx[1].loc[attr_values,:][0]
#                 if result_curr > result_last:
#                     result_last = result_curr
#                     buckets_stab.append([result_last,left,right])
#         left = left[1:]
# 
# bucket_dict = {}
# for i in buckets_stab:
#     bucket_dict.setdefault(len(i[1]),[]).append(i)
# 
# dictkeys = sorted(list(bucket_dict.keys()))[::-1]
# 
# bucket_list = []
# for key in dictkeys:
#     max_val = max([i[0] for i in bucket_dict[key]])
#     tmp = [i for i in bucket_dict[key] if i[0] == max_val]
#     if tmp[0][0] not in [i[0] for i in bucket_list]:
#         bucket_list.extend(tmp)
# =============================================================================




# --> alternative

############################## ADBIS EXPERIMENTS ##############################

x = [('F', 'F'), ('F', 'M'), ('M', 'F'), ('M', 'M')]
y = [('1A','1A'), ('1B','1B'), ('2A','2A'), ('2B','2B'), ('3A','3A'), \
     ('3B','3B'), ('4A','4A'), ('4B','4B'), ('5A','5A'), ('5B','5B')]

# Stability (tnew(inx)&told / told(inx)&tnew) (maximal)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']
    
    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        result_last = 0
        while len(left) >= 1:
            inx,tia_inx = Intersection_Static(
                                                nodes_df,
                                                edges_df,
                                                time_invariant_attr,
                                                left+right
                                                )
            if not inx[1].empty:
                agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
                if attr_values in agg_inx[1].index:
                    result_curr = agg_inx[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(left),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            left = left[1:]
    
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))


# growth (tnew - told(union)) (maximal)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
        c += 1
    intvls = intvls[:-1]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        result_last = 0
        while len(left) >= 1:
            #print('left',left,'right',right)
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        right,
                                        left
                                        )
            if not diff[1].empty:
                #print('diff empty?', diff[1].empty)
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    #print('curr', result_curr, '>' ,'last', result_last)
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(left),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            left = left[1:]
            #print('----------------------')
    
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))
    

# shrinkage (told(union) - tnew) (minimal)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        flag = True
        result_last = 0
        while flag == True:
            print('left',left,'right',right)
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        left,
                                        right
                                        )
            if not diff[1].empty:
                print('diff empty?', diff[1].empty)
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    print('curr', result_curr, '>' ,'last', result_last)
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(left),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            if left[0] != edges_df.columns[0]:
                flag = True
                left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
            else:
                flag = False
            print('----------------------')
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))


###############################################################################
















# growth (tnew - told(union))
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        flag = True
        result_last = 0
        while flag == True:
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        right,
                                        left
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(left),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            if left[0] != edges_df.columns[0]:
                flag = True
                left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
            else:
                flag = False
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))



# growth (tnew(union) - told)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        flag = True
        result_last = 0
        while flag == True:
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        right,
                                        left
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(right),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            if right[-1] != edges_df.columns[-1]:
                flag = True
                right = right+[edges_df.columns[list(edges_df.columns).index(right[-1])+1]]
            else:
                flag = False
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))



# growth (tnew(inx) - told)

def Diff_Static_inx(nodesdf,edgesdf,tia,intvl_fst,intvl_scd):
    un_init, tia_init = Intersection_Static(nodesdf,edgesdf,tia,intvl_fst)
    un_to_rm, tia_to_rm = Intersection_Static(nodesdf,edgesdf,tia,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    ndiff_idx = set(i for i in nodes.index.values.tolist())
    idx = set(list(ediff_idx) + list(ndiff_idx))
    tia_d = tia_init[tia_init.index.isin(idx)]
    diff = [nodes,edges]
    return(diff,tia_d)

for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[i-1:i]), list(edges_df.columns[i:])])
        c += 1
    intvls = intvls[:-1]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        result_last = 0
        while len(right) >= 1:
            diff,tia_diff = Diff_Static_inx(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        right,
                                        left
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(right),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            right = right[:-1]
    
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))



# SHRINKAGE

# shrinkage (told(union) - tnew)
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        flag = True
        result_last = 0
        while flag == True:
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        left,
                                        right
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(left),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            if left[0] != edges_df.columns[0]:
                flag = True
                left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
            else:
                flag = False
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))



# shrinkage (told - tnew(union))
for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    s = [[str(i)] for i in edges_df.columns[:-1]]
    e = [[str(i)] for i in edges_df.columns[1:]]
    intvls = list(zip(s,e))
    intvls = [list(i) for i in intvls]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        flag = True
        result_last = 0
        while flag == True:
            diff,tia_diff = Diff_Static(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        left,
                                        right
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(right),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            if right[-1] != edges_df.columns[-1]:
                flag = True
                right = right+[edges_df.columns[list(edges_df.columns).index(right[-1])+1]]
            else:
                flag = False
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))



# shrinkage (told - tnew(inx))

def Diff_Static_inx(nodesdf,edgesdf,tia,intvl_fst,intvl_scd):
    un_init, tia_init = Intersection_Static(nodesdf,edgesdf,tia,intvl_fst)
    un_to_rm, tia_to_rm = Intersection_Static(nodesdf,edgesdf,tia,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    ndiff_idx = set(i for i in nodes.index.values.tolist())
    idx = set(list(ediff_idx) + list(ndiff_idx))
    tia_d = tia_init[tia_init.index.isin(idx)]
    diff = [nodes,edges]
    return(diff,tia_d)

for xi in x:
    attr_values = xi
    stc_attrs = ['gender']

    c=0
    intvls = []
    for i in range(1,len(edges_df.columns)+1-c):
        intvls.append([list(edges_df.columns[i-1:i]), list(edges_df.columns[i:])])
        c += 1
    intvls = intvls[:-1]
    
    bucket_len_test = {}
    bucket_k_test = {}
    for left,right in intvls:
        result_last = 0
        while len(right) >= 1:
            diff,tia_diff = Diff_Static_inx(
                                        nodes_df,
                                        edges_df,
                                        time_invariant_attr,
                                        left,
                                        right
                                        )
            if not diff[1].empty:
                agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
                if attr_values in agg_diff[1].index:
                    result_curr = agg_diff[1].loc[attr_values,:][0]
                    if result_curr > result_last:
                        result_last = result_curr
                        bucket_len_test.setdefault(len(right),[]).append([result_last,left,right])
                        bucket_k_test.setdefault(result_last,[]).append([result_last,left,right])
            right = right[:-1]
    
    
    len_list = []
    for key,val in bucket_len_test.items():
        max_k = max([i[0] for i in val])
        tmp = [i for i in val if i[0]==max_k]
        len_list.extend(tmp)
    
    k_list = []
    for key,val in bucket_k_test.items():
        max_len = max([len(i[1]) for i in val])
        tmp = [i for i in val if len(i[1])==max_len]
        k_list.extend(tmp)
    
    k_len_common = [i for i in len_list if i in k_list]
    
    print(attr_values)
    print(len(k_len_common))


##############

# =============================================================================
# # minimal
# # Growth (extend old)
# 
# attr_values = ('F','F')
# stc_attrs = ['gender']
# 
# s = [[str(i)] for i in edges_df.columns[:-1]]
# e = [[str(i)] for i in edges_df.columns[1:]]
# intvls = list(zip(s,e))
# intvls = [list(i) for i in intvls]
# 
# buckets_grow = []
# for left,right in intvls:
#     result_last = 0
#     while left[0] != edges_df.columns[0]:
#         diff,tia_diff = Diff_Static(
#                                     nodes_df,
#                                     edges_df,
#                                     time_invariant_attr,
#                                     right,
#                                     left
#                                     )
#         if not diff[1].empty:
#             agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
#             if attr_values in agg_diff[1].index:
#                 result_curr = agg_diff[1].loc[attr_values,:][0]
#                 if result_curr > result_last:
#                     result_last = result_curr
#                     buckets_grow.append([result_last,left,right])
#         left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
# 
# bucket_dict = {}
# for i in buckets_grow:
#     bucket_dict.setdefault(len(i[1]),[]).append(i)
# 
# dictkeys = sorted(list(bucket_dict.keys()))[::-1]
# 
# bucket_list = []
# for key in dictkeys:
#     max_val = max([i[0] for i in bucket_dict[key]])
#     tmp = [i for i in bucket_dict[key] if i[0] == max_val]
#     if tmp[0][0] not in [i[0] for i in bucket_list]:
#         bucket_list.extend(tmp)
# 
# # minimal
# # Shrinkage (extend old)
# 
# attr_values = ('F','F')
# stc_attrs = ['gender']
# 
# s = [[str(i)] for i in edges_df.columns[:-1]]
# e = [[str(i)] for i in edges_df.columns[1:]]
# intvls = list(zip(s,e))
# intvls = [list(i) for i in intvls]
# 
# buckets_shrink = []
# for left,right in intvls:
#     result_last = 0
#     while left[0] != edges_df.columns[0]:
#         diff,tia_diff = Diff_Static(
#                                     nodes_df,
#                                     edges_df,
#                                     time_invariant_attr,
#                                     left,
#                                     right
#                                     )
#         if not diff[1].empty:
#             agg_diff = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
#             if attr_values in agg_diff[1].index:
#                 result_curr = agg_diff[1].loc[attr_values,:][0]
#                 if result_curr > result_last:
#                     result_last = result_curr
#                     buckets_shrink.append([result_last,left,right])
#         left = [edges_df.columns[list(edges_df.columns).index(left[0])-1]]+left
# 
# bucket_dict = {}
# for i in buckets_shrink:
#     bucket_dict.setdefault(len(i[1]),[]).append(i)
# 
# dictkeys = sorted(list(bucket_dict.keys()))[::-1]
# 
# bucket_list = []
# for key in dictkeys:
#     max_val = max([i[0] for i in bucket_dict[key]])
#     tmp = [i for i in bucket_dict[key] if i[0] == max_val]
#     if tmp[0][0] not in [i[0] for i in bucket_list]:
#         bucket_list.extend(tmp)
# 
# bucket_list.sort()
# =============================================================================


#***************************************


# plot skyline
data = {}
colors = []
x = []
y = []
for lst in bucket_list:
    colors.extend(['blue', 'blue', 'brown'])
    x.extend([int(lst[1][0]), int(lst[1][-1]), None])
    y.extend([str(lst[0]) + ' (PoR: ' + str(lst[-1][0]) + ')', str(lst[0]) + ' (PoR: ' + str(lst[-1][0]) + ')', None])
data['colors'] = colors
data['x'] = x
data['y'] = y

fig = go.Figure(
    data=[
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="lines",
            marker=dict(
                color="#4169E1",
            ),
        ),
        go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="markers+text",
            marker=dict(
                color="#4169E1",
                #color=data["colors"],
                size=15,
            ),
        ),
    ]
)

fig.update_layout(
    xaxis = dict(
        tickvals = [i for i in interval],
        title = 'Interval'
        #ticktext = [i.upper() for i in interval]
    ),
    yaxis = dict(
        title = '#Interactions'
        #ticktext = [i.upper() for i in interval]
    ),
    title="Stability skyline results for " + str(attr_values) + " interactions",
    width=750,
    height=600,
    showlegend=False,
    font_size=20,
    plot_bgcolor='rgba(0,0,0,0)'
)
fig.update_traces(marker_size=15, line=dict(width=2.5))#, line_color="#4169E1")
fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
fig.update_yaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)

fig.show()


############################


#****************************
import sys
sys.path.insert(1, 'graphtempo')
from graphtempo import *



# limits
intvl_pairs = [[i,interval[interval.index(i)+1]] for i in interval[:-1]]
stc_attrs = ['gender']
#stc_attrs = ['class']
attr_values = ('F','F')
#attr_values = ('M','M')

# GENDER
# Stability
stab_pairs = []
for i in intvl_pairs:
    inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,i)
    agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs)
    if not agg_inx[1].empty:
    	stab_pairs.append(agg_inx[1].loc[attr_values][0])
    else:
        stab_pairs.append(0)

# Growth
growth_pairs = []
for i in intvl_pairs:
    diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[1]],[i[0]])
    agg_diff_G = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
    if not agg_diff_G[1].empty:
    	growth_pairs.append(agg_diff_G[1].loc[attr_values][0])
    else:
        growth_pairs.append(0)

# Shrinkage
shrinkage_pairs = []
for i in intvl_pairs:
    diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[0]],[i[1]])
    agg_diff_S = Aggregate_Static_Dist(diff,tia_diff,stc_attrs)
    if not agg_diff_S[1].empty:
    	shrinkage_pairs.append(agg_diff_S[1].loc[attr_values][0])
    else:
        shrinkage_pairs.append(0)



# plot exploration

stc_attrs = ['gender']
#stc_attrs = ['class']
attr_values = ('F','F')
#attr_values = ('M','M')
#attr_values = ('1A','1A')
k=50
result,myagg = Stability_Intersection_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
#result,myagg = Growth_Union_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
#result,myagg = Shrink_Union_Static_a(k,interval,nodes_df,edges_df,time_invariant_attr,stc_attrs,attr_values)
result = result[::-1]

result_lst = []
for lst in result:
	for i in lst[0]:
		tmp = [i,lst[1][0]]
		result_lst.append(tmp)

# map str to num
str_num = {}
for i in interval:
	str_num[i] = int(i)

# map num to str
num_str = {}
for i in interval:
	num_str[int(i)] = str(i)

# convert time points to strings
result_lst = [[str_num[i],str_num[j]] for i,j in result_lst]

result_df = pd.DataFrame(result_lst)
df_cols = ['Interval','Point of Reference']
result_df.columns = df_cols
result_df_grouped = [i[1].values.tolist() for i in result_df.groupby('Point of Reference')]
result_df_grouped = [[i[0],i[-1]] for i in result_df_grouped]
result_df_grouped = [i for sublst in result_df_grouped for i in sublst]
# # return to str
result_df = pd.DataFrame(result_df_grouped)
result_df.columns = df_cols

x = result_df['Interval'].tolist()
y = result_df['Point of Reference'].tolist()
fig = px.line(
            result_df, 
            x="Interval", 
            y="Point of Reference", 
            color='Point of Reference', 
            markers=True, 
            #title='k: '+ str(k) + ' ' + str(attr_values) + ' interactions',
            )
fig.update_traces(textposition="bottom right", marker_size=15, line=dict(width=2.5), line_color="#4169E1")
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = [i for i in interval],
        ticktext = [i.upper() for i in interval],
        ticks = "outside", ticklen=10
    ),
    yaxis = dict(
        tickmode = 'array',
        tickvals = [i for i in interval],
        ticktext = [i.upper() for i in interval],
        ticks = "outside", ticklen=10
    ),

)

#fig.update_xaxes(tickangle= -90)
fig.update_layout(showlegend=False, font_size=20, plot_bgcolor='rgba(0,0,0,0)', autosize=False, width=750, height=600)
fig.update_xaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
fig.update_yaxes(showgrid=True, gridcolor='lightgrey', showline=True, linecolor = 'black', zeroline=False, mirror=True)
fig.update_layout(xaxis_range=[1, 17],yaxis_range=[1, 17])
fig.show()








# =============================================================================
# # F - F exploration
# if __name__ == '__main__':
#     filename = sys.argv[1]
#     
#     #DBLP
#     if filename == 'dblp_dataset':
#         # READ edges, nodes, static and variant attributes from csv
#         edges_df = pd.read_csv(filename + '/edges.csv', sep=' ', index_col=[0,1])
#         nodes_df = pd.read_csv(filename + '/nodes.csv', sep=' ', index_col=0)
#         time_variant_attr = pd.read_csv(filename + '/time_variant_attr.csv', sep=' ', index_col=0)
#         time_invariant_attr = pd.read_csv(filename + '/time_invariant_attr.csv', sep=' ', index_col=0)
#         time_invariant_attr.rename(columns={'0': 'gender'}, inplace=True)
#         nodes_df.index.names = ['userID']
#         time_invariant_attr.gender.replace(['female','male'], ['F','M'],inplace=True)
#         intvl = [str(i) for i in range(2000,2021)]
#         # intersection
#         intvl_pairs = [[i,intvl[intvl.index(i)+1]] for i in intvl[:-1]]
#         inx_pairs = []
#         for i in intvl_pairs:
#             inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,i)
#             agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['gender'])
#             inx_pairs.append(agg_inx[1].loc['F','F'][0])
#         #k=max(inx_pairs)
#         stabI_a_k1 = Stability_Intersection_a(max(inx_pairs),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(inx_pairs)/2
#         stabI_a_k2 = Stability_Intersection_a(max(inx_pairs)/2,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(inx_pairs)/62
#         stabI_a_k3 = Stability_Intersection_a(max(inx_pairs)/62,intvl,nodes_df,edges_df,time_invariant_attr)
#         stabInx_a = [stabI_a_k1] + [stabI_a_k2] + [stabI_a_k3]
#         # difference growth
#         diff_pairs_G = []
#         for i in intvl_pairs:
#             diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[1]],[i[0]])
#             agg_diff_G = Aggregate_Static_Dist(diff,tia_diff,stc_attrs=['gender'])
#             diff_pairs_G.append(agg_diff_G[1].loc['F','F'][0])
#         #k=max(diff_pairs_G)
#         growthU_a_k1 = Growth_Union_a(max(diff_pairs_G),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)/3
#         growthU_a_k2 = Growth_Union_a(max(diff_pairs_G)/3,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)/10
#         growthU_a_k3 = Growth_Union_a(max(diff_pairs_G)/10,intvl,nodes_df,edges_df,time_invariant_attr)
#         growthU_a = [growthU_a_k1] + [growthU_a_k2] + [growthU_a_k3]
#         # difference shrinkage
#         diff_pairs_S = []
#         for i in intvl_pairs:
#             diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[0]],[i[1]])
#             agg_diff_G = Aggregate_Static_Dist(diff,tia_diff,stc_attrs=['gender'])
#             diff_pairs_S.append(agg_diff_G[1].loc['F','F'][0])
#         #k=max(diff_pairs_G)
#         shrinkU_a_k1 = Shrink_Union_a(min(diff_pairs_S),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)*5
#         shrinkU_a_k2 = Shrink_Union_a(min(diff_pairs_S)*5,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)*20
#         shrinkU_a_k3 = Shrink_Union_a(min(diff_pairs_S)*20,intvl,nodes_df,edges_df,time_invariant_attr)
#         shrinkU_a = [shrinkU_a_k1] + [shrinkU_a_k2] + [shrinkU_a_k3]
#     
#     
#     # MovieLens
#     if filename == 'movielens_dataset':
#         # READ edges, nodes, static and variant attributes from csv
#         edges_df = pd.read_csv(filename + '/edges.csv', sep=' ')
#         edges_df.set_index(['Left', 'Right'], inplace=True)
#         nodes_df = pd.read_csv(filename + '/nodes.csv', sep=' ', index_col=0)
#         time_variant_attr = pd.read_csv(filename + '/time_variant_attr.csv', sep=' ', index_col=0)
#         time_invariant_attr = pd.read_csv(filename + '/time_invariant_attr.csv', sep=' ', index_col=0)
#         intvl = ['may','jun','jul','aug','sep','oct']
#         # intersection
#         intvl_pairs = [[i,intvl[intvl.index(i)+1]] for i in intvl[:-1]]
#         inx_pairs = []
#         for i in intvl_pairs:
#             inx,tia_inx = Intersection_Static(nodes_df,edges_df,time_invariant_attr,i)
#             agg_inx = Aggregate_Static_Dist(inx,tia_inx,stc_attrs=['gender'])
#             inx_pairs.append(agg_inx[1].loc['F','F'][0])
#         #k=max(inx_pairs)
#         stabI_a_k1 = Stability_Intersection_a(max(inx_pairs),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(inx_pairs)/2
#         stabI_a_k2 = Stability_Intersection_a(max(inx_pairs)/2,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(inx_pairs)/62
#         stabI_a_k3 = Stability_Intersection_a(max(inx_pairs)/86,intvl,nodes_df,edges_df,time_invariant_attr)
#         stabInx_a = [stabI_a_k1] + [stabI_a_k2] + [stabI_a_k3]
#         # difference growth
#         diff_pairs_G = []
#         for i in intvl_pairs:
#             diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[1]],[i[0]])
#             agg_diff_G = Aggregate_Static_Dist(diff,tia_diff,stc_attrs=['gender'])
#             diff_pairs_G.append(agg_diff_G[1].loc['F','F'][0])
#         #k=max(diff_pairs_G)
#         growthU_a_k1 = Growth_Union_a(max(diff_pairs_G),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)/2
#         growthU_a_k2 = Growth_Union_a(max(diff_pairs_G)/2,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)/12
#         growthU_a_k3 = Growth_Union_a(max(diff_pairs_G)/12,intvl,nodes_df,edges_df,time_invariant_attr)
#         growthU_a = [growthU_a_k1] + [growthU_a_k2] + [growthU_a_k3]
#         # difference shrinkage
#         diff_pairs_S = []
#         for i in intvl_pairs:
#             diff,tia_diff = Diff_Static(nodes_df,edges_df,time_invariant_attr,[i[0]],[i[1]])
#             agg_diff_G = Aggregate_Static_Dist(diff,tia_diff,stc_attrs=['gender'])
#             diff_pairs_S.append(agg_diff_G[1].loc['F','F'][0])
#         #k=max(diff_pairs_G)
#         shrinkU_a_k1 = Shrink_Union_a(min(diff_pairs_S),intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)*2
#         shrinkU_a_k2 = Shrink_Union_a(min(diff_pairs_S)*2,intvl,nodes_df,edges_df,time_invariant_attr)
#         #k=max(diff_pairs_G)*5
#         shrinkU_a_k3 = Shrink_Union_a(min(diff_pairs_S)*5,intvl,nodes_df,edges_df,time_invariant_attr)
#         shrinkU_a = [shrinkU_a_k1] + [shrinkU_a_k2] + [shrinkU_a_k3]
#     
#     
#     
#     
#     #save output for stability 
#     pd.DataFrame(stabInx_a).to_csv('out_stabInx_a.txt', sep=' ', mode='w')
#     #save output for growth
#     pd.DataFrame(growthU_a).to_csv('out_growthU_a.txt', sep=' ', mode='w')
#     #save output for shrinkage
#     pd.DataFrame(shrinkU_a).to_csv('out_shrinkU_a.txt', sep=' ', mode='w')
# 
# =============================================================================





