
# ------------------------------------------------------------------

# for each node in the nodes set, finds and keeps in list the edges with node as first element
# and finds and keeps in another list the edges with node as second element
# next finds all the node combinations between the second elements of the first list
# and the first element of the second list
# creates a list with all combinations of first and third edge and adds the second list if appropriate
# edge in the combinations list

import pandas as pd
import itertools
import networkx as nx

edges_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
#edges_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/movielens_dataset/edges.csv', sep=' ', index_col=[0,1])

nodes_triangles_dict = {}
for i in range(len(edges_df.columns)):
    edges = edges_df.iloc[:,i][edges_df.iloc[:,i]!=0]
    #nodes = list(set([i for e in edges.index.values.tolist() for i in e]))
    nodes_left = set(edges.index.get_level_values('Left').values.tolist())
    nodes_right = set(edges.index.get_level_values('Right').values.tolist())
    nodes = list(set([e for e in nodes_left if e in nodes_right]))
    edges = edges.reset_index().iloc[:,:-1]
 
    triangles = []
    for node in nodes:
        df_first_edges = edges.loc[edges.Left==node]
        if not df_first_edges.empty:
            df_third_edges = edges.loc[edges.Right==node]
            if not df_third_edges.empty:
                second_edges_first = df_first_edges.Right.values.tolist()
                second_edges_third = df_third_edges.Left.values.tolist()
                second_edges_domain = list(itertools.product(second_edges_first, second_edges_third))
                first_edges_domain = df_first_edges.values.tolist()
                first_edges_domain = [tuple(e) for e in first_edges_domain]
                third_edges_domain = df_third_edges.values.tolist()
                third_edges_domain = [tuple(e) for e in third_edges_domain]
                
                triangles_domain = list(
                    itertools.product(
                        first_edges_domain,
                        third_edges_domain
                        )
                    )
                edges_lst = set([tuple(e) for e in edges.values.tolist()])
                for triangle in triangles_domain:
                    second = (triangle[0][1],triangle[1][0])
                    if second in edges_lst:
                        #triangle_sorted = tuple(sorted(triangle[0]+triangle[1][:1]))
                        triangle_final = triangle[0]+triangle[1][:1]
                        triangles.append(triangle_final)
    
    triangles = list(set(sorted(triangles)))
    
    ones = [0]*len(edges_df.columns)
    ones[i] = 1
    for t in triangles:
        if t not in nodes_triangles_dict.keys():
            nodes_triangles_dict[t] = ones
        else:
            nodes_triangles_dict[t][i] = 1


tria_df = pd.DataFrame(nodes_triangles_dict.values(), index=nodes_triangles_dict.keys())
tria_df = tria_df.sort_index()

tria_df.index.names = ['id1','id2','id3']

tria_df2 = pd.DataFrame(tria_df)
tria_df2.index = tria_df.index.tolist()
# save to csv file
tria_df2.reset_index().to_csv('triangles_datasets/dblp/nodes_triangles.csv', sep=',', index=None)



# create attributes

# replace with attributes
time_inv_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/dblp_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
#time_inv_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/movielens_dataset/time_invariant_attr.csv', sep=' ', index_col=0)
time_inv_df.gender.replace(['female','male'], ['F','M'],inplace=True)

# triangles to df
triangles_df = pd.DataFrame(tria_df.reset_index().iloc[:,:3])

attr_triangles = []
for i in range(3):
    tmp = time_inv_df.loc[:,'gender'].to_frame().loc[triangles_df.iloc[:,i],:]['gender'].tolist()
    attr_triangles.append(tmp)

attr_triangles = list(zip(*attr_triangles))
attr_triangles = [''.join(sorted(i)) for i in attr_triangles]

attr_triangles = pd.DataFrame(attr_triangles)

attr_triangles.index = tria_df.index
# or
attr_triangles.index = tria_df.index.tolist()

# save to csv file
attr_triangles.reset_index().to_csv('triangles_datasets/dblp/attr_triangles.csv', sep=',', index=None)

# create edges
edges_triangles_dict = {}
for i in range(len(edges_df.columns)):
    triangles = tria_df.iloc[:,i][tria_df.iloc[:,i]!=0].index.tolist()

    tria_edges = []
    for ind,tria in enumerate(triangles[:-1]):
        combs = list(itertools.product([tria], triangles[ind+1:]))
        for c in combs:
            if len(set(c[0] + c[1])) < 6:
                tria_edges.append(c)
            else:
                all_edges = [(k,j) for k in c[0] for j in c[1]]
                all_edges_rvsd = [k[::-1] for k in all_edges]
                if any(e in edges_lst for e in all_edges+all_edges_rvsd):
                    tria_edges.append(c)

    tria_edges = list(set(sorted(tria_edges)))
    
    ones = [0]*len(edges_df.columns)
    ones[i] = 1
    for t in tria_edges:
        if t not in edges_triangles_dict.keys():
            edges_triangles_dict[t] = ones
        else:
            edges_triangles_dict[t][i] = 1

edges_tria_df = pd.DataFrame(edges_triangles_dict.values(), index=edges_triangles_dict.keys())
edges_tria_df = edges_tria_df.sort_index()

edges_tria_df.index.names = ['Left','Right']

# save to csv file
edges_tria_df.reset_index().to_csv('triangles_datasets/dblp/edges_triangles.csv', sep=',', index=None)
