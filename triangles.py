
# =============================================================================
# import networkx as nx
# import pandas as pd
# 
# 
# def findPaths(G,u,n):
#     if n==0:
#         return [[u]]
#     paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1)]
#     return paths
# 
# def find_cycles(G,u,n):
#     paths = findPaths(G,u,n)
#     return [tuple(path) for path in paths if (path[-1] == u) and sum(x==u for x in path) == 2]
# 
# 
# edges_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/dblp_dataset/edges.csv', sep=' ', index_col=[0,1])
# #edges_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/movielens_dataset/edges.csv', sep=' ', index_col=[0,1])
# 
# edges = edges_df.iloc[:,0][edges_df.iloc[:,0]!=0]
# edges = edges.index.tolist()
# nodes = list(set([i for e in edges for i in e]))
# G = nx.DiGraph(edges)
# 
# 
# cycles = []
# for i in nodes:
#     c = find_cycles(G,i,3)
#     cycles.extend(c)
# =============================================================================







# ------------------------------------------------------------------

# for each node in the nodes set, finds and keeps in list the edges with node as first element
# and finds and keeps in another list the edges with node as second element
# next finds all the node combinations between the second elements of the first list
# and the first element of the second list
# creates a list with all combinations of first and third edge and adds the second list if appropriate
# edge in the combinations list

import pandas as pd
import itertools

edges_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/movielens_dataset/edges.csv', sep=' ', index_col=[0,1])
edges = edges_df.iloc[:,0][edges_df.iloc[:,0]!=0]
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
                    triangles.append(triangle[0]+triangle[1][:1])

triangles = sorted(triangles)


# replace with attributes
time_inv_df = pd.read_csv('C:/Users/Lila/Desktop/GraphTempo_APP/datasets/movielens_dataset/time_invariant_attr.csv', sep=' ', index_col=0)

# triangles to df
triangles_df = pd.DataFrame(triangles)

triangles_lst = []
for i in range(3):
    tmp = time_inv_df.loc[:,'gender'].to_frame().loc[triangles_df.iloc[:,i],:]['gender'].tolist()
    triangles_lst.append(tmp)

triangles_lst = list(zip(*triangles_lst))
triangles_lst = [''.join(sorted(i)) for i in triangles_lst]

triangles_attr_df = pd.DataFrame(triangles_lst)








