from itertools import product
from multiprocessing import Pool, cpu_count, Value
import time
import logging
import pickle

import pandas as pd
from functools import cached_property
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors
import utm



class DistanceMatrixCreator:
    def __init__(self, position_path, neighbours=100):
        self.raw_data = pd.read_csv(f'{position_path}/position.csv', sep='|')
        self.url = 'http://router.project-osrm.org/table/v1/driving/{}'
        self.neighbours = neighbours

    @cached_property
    def data(self):
        df = self.raw_data.copy()
        df[['lat', 'lon']] = df.apply(lambda row: utm.to_latlon(row['utm_x'], row['utm_y'], 30, 'U'), axis=1,
                                      result_type='expand')
        return df

    @cached_property
    def nodes(self):
        return list(self.data['id'].unique())

    @cached_property
    def samples(self):
        return self.data[['utm_x', 'utm_y']].values.tolist()

    @cached_property
    def nodes_map(self):
        return self.data['id'].to_dict()

    @cached_property
    def neighbourhood(self):
        neigh = NearestNeighbors(n_neighbors=self.neighbours)
        neigh.fit(self.samples)
        neighbourhood = neigh.kneighbors(self.samples, return_distance=False)
        neighbourhood = np.vectorize(self.nodes_map.get)(neighbourhood)
        return dict(zip(self.nodes, neighbourhood))

    def node_distance(self, node):
        # neigbour_df = self.data[self.data['id'].isin(self.neighbourhood[node])]
        neigbour_df = self.data.set_index('id').loc[self.neighbourhood[node]].reset_index(inplace=False)
        node_list = neigbour_df[['lon', 'lat']].values.tolist()
        node_str = str(node_list).replace('],', ';').replace('[', '').replace(']', '').replace(' ', '')
        url = self.url.format(node_str)
        r = requests.get(url)
        r_json = r.json()
        df = pd.DataFrame(r_json['durations'], index=neigbour_df.id, columns=neigbour_df.id)
        return df.iloc[0]

    @cached_property
    def node_distance_df(self):
        df = pd.DataFrame()
        start = time.time()

        # with Pool(processes=5) as pool:
        #     imputed_nodes = pool.map(self.node_distance, self.n_nodes)
        # for imputed_node in imputed_nodes:
        #     df = df.append(imputed_node)

        for i, node in enumerate(self.nodes):
            print(f'Starting teration {i}.  Node {node}.    Time {(time.time() - start)/60} minutes')
            df = df.append(self.node_distance(node))
        return df[self.nodes]

    def save_node_distance(self, graph_path):
        self.node_distance_df.to_csv(f'{graph_path}/node_distance.csv', sep='|')


class GraphCreator:
    def __init__(self, distance_matrix_path, nodes=None, k=None, sim=False, max_connections=None):
        self.distance_matrix_filename = f'{distance_matrix_path}/node_distance.csv'
        self.distance_matrix.columns = [int(col) for col in self.distance_matrix.columns]
        self.max_connections = max_connections
        self._nodes = nodes
        self._k = k
        self.sim = sim
        self._weighted_adjacency_matrix = pd.DataFrame()

    @cached_property
    def distance_matrix(self):
        df = pd.read_csv(self.distance_matrix_filename, sep='|', index_col=0)
        df.columns = df.columns.astype(int)
        df.sort_index(inplace=True)
        return df[df.index]

    def set_section_nodes(self, position_path, bottom=0.0, top=99999999.0, left=0.0, right=99999999.0):
        position_df = pd.read_csv(f'{position_path}/position_lc.csv', sep='|')
        df = position_df[
            position_df.utm_y.between(bottom, top) & position_df.utm_x.between(left, right)]
        self._nodes = list(df['id'].unique())
        self._nodes.sort()

    @property
    def nodes(self):
        if self._nodes is None:
            return list(self.distance_matrix.index)
        else:
            return self._nodes

    @cached_property
    def section_distance(self):
        return self.distance_matrix.loc[self.nodes, self.nodes]

    @cached_property
    def node_distance(self):
        df = self.section_distance.copy()
        if self.max_connections is not None:
            aux_df = pd.DataFrame()
            for c in df.columns:
                s = df[c].nsmallest(self.max_connections)
                aux_df = pd.concat([aux_df, pd.DataFrame(s)], axis=1)
            df = aux_df
        if self.sim:
            aux_df = df.copy()
            for x, y in product(self.section_distance.index, self.section_distance.index):
                m = np.nanmin([df[y].loc[x], df[x].loc[y]])
                aux_df[x][y] = m
            df = aux_df
        return df

    @cached_property
    def min_max_distance(self):
        df = self.node_distance.replace(0, np.nan)
        min_org = df.min()
        min_end = df.min(axis=1)
        df = pd.DataFrame({'org': min_org, 'end': min_end})
        min_tot = df.min(axis=1)
        return min_tot.max()

    @cached_property
    def k(self):
        if self._k is None:
            return self.min_max_distance
        else:
            return self._k

    @cached_property
    def distance_std(self):
        dist = self.node_distance.values.reshape(-1)
        dist = [x for x in dist if (x == x) & (x != 0)]
        return np.std(dist)

    @cached_property
    def weighted_adjacency_matrix(self):
        df = list()
        for i, row in self.node_distance.iterrows():
            row_list = list()
            for col in self.node_distance.columns:
                if row[col] <= self.k:
                    v = np.exp(-row[col] ** 2 / self.distance_std ** 2)
                else:
                    v = 0
                row_list.append(v)
            df.append(row_list)
        df = pd.DataFrame(df)
        df.index = self.node_distance.index
        df.columns = self.node_distance.columns
        return df


    @cached_property
    def adjacency_matrix(self):
        return self.weighted_adjacency_matrix.astype(bool).astype(int)

    @cached_property
    def edge_index(self):
        edge_list = list()
        for i, row in self.adjacency_matrix.iterrows():
            for col in self.adjacency_matrix.columns:
                if row[col]:
                    v = np.array((i, int(col)))
                    edge_list.append(v)
        return np.array(edge_list)

    @cached_property
    def edge_attributes(self):
        attributes_list = list()
        for i, row in self.weighted_adjacency_matrix.iterrows():
            for col in self.weighted_adjacency_matrix.columns:
                if row[col]:
                    attributes_list.append(row[col])
        return np.array(attributes_list)

    def save_weighted_adjacency_matrix(self, adjacency_path):
        adjency_name = 'adjacency' if self.max_connections is None else f'adjacency_{self.max_connections}'
        if self.sim:
            filename = f'{adjacency_path}/{adjency_name}_sim.csv'
        else:
            filename = f'{adjacency_path}/{adjency_name}.csv'
        self.weighted_adjacency_matrix.to_csv(filename, sep='|')

    def save_edge_attributes(self, pickle_path):
        pickle_filename = f'{pickle_path}/edge_attributes.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.edge_attributes, f)

    def save_edge_index(self, pickle_path):
        pickle_filename = f'{pickle_path}/edge_index.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.edge_index, f)



if __name__ == "__main__":
    import dask.dataframe as dd
    position_path_f = '../../data/traffic/position/lf'
    position_path_c = '../../data/traffic/position/lc'
    graph_path = '../../data/traffic/graph/graph_full.csv'
    # adjacency_path = '../../data/traffic/graph/adjacency_matrix/small'
    adjacency_path = '../../data/traffic/graph/adjacency_matrix/m30_speed'
    adjacency_path2 = '../../data/traffic/graph/adjacency_matrix/m30_speed2'
    adjacency_path3 = '../../data/traffic/graph/adjacency_matrix/madrid'
    dist_path = '../../data/traffic/graph/distance_matrix'
    traffic_dir_path = '../../data/traffic/historic/lc'
    # north = 4480377.72
    # south = 4478785.58
    # east = 439681.51
    # west = 438717.21


    # north = 4476757.84
    # south = 4474526.87
    # east = 441197.05
    # west = 439133.71

    filename = f'{traffic_dir_path}/historic_lc.parquet'
    ddf = dd.read_parquet(filename, engine='pyarrow')

    nodes = list(ddf.id.unique())
    graph = GraphCreator(dist_path, max_connections=3, nodes=nodes)
    graph.save_weighted_adjacency_matrix(adjacency_path3)

    graph_sim = GraphCreator(dist_path, sim=True, nodes=nodes, max_connections=3)
    graph_sim.save_weighted_adjacency_matrix(adjacency_path3)

    # graph = GraphCreator(dist_path)
    # graph.set_section_nodes(position_path=position_path_c)
    # graph.save_weighted_adjacency_matrix(adjacency_path)
    #
    # graph_sim = GraphCreator(dist_path, sim=True)
    # graph_sim.set_section_nodes(position_path=position_path_c)
    # graph_sim.save_weighted_adjacency_matrix(adjacency_path)
