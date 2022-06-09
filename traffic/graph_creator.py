from itertools import product
import time
import pickle

import pandas as pd
from functools import cached_property
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors
import utm


class DistanceMatrixCreator:
    """
    Driving distance between nodes

    Args:
        position_path: Read position data path
        neighbours:
    """

    def __init__(self, position_path, neighbours=100):
        self.raw_data = pd.read_csv(f'{position_path}/position.csv', sep='|')
        self.url = 'http://router.project-osrm.org/table/v1/driving/{}'
        self.neighbours = neighbours

    @cached_property
    def data(self):
        """
        Curated data
        """
        df = self.raw_data.copy()
        df[['lat', 'lon']] = df.apply(lambda row: utm.to_latlon(row['utm_x'], row['utm_y'], 30, 'U'), axis=1,
                                      result_type='expand')
        return df

    @cached_property
    def nodes(self):
        """
        Node list
        """
        return list(self.data['id'].unique())

    @cached_property
    def samples(self):
        """
        Data utm position
        """
        return self.data[['utm_x', 'utm_y']].values.tolist()

    @cached_property
    def nodes_map(self):
        """
        Node index dictionary
        """
        return self.data['id'].to_dict()

    @cached_property
    def neighbourhood(self):
        """
        Neighbourhood of each node
        """
        neigh = NearestNeighbors(n_neighbors=self.neighbours)
        neigh.fit(self.samples)
        neighbourhood = neigh.kneighbors(self.samples, return_distance=False)
        neighbourhood = np.vectorize(self.nodes_map.get)(neighbourhood)
        return dict(zip(self.nodes, neighbourhood))

    def node_distance(self, node):
        """
        Node driving distance of each node to its neighbourhood

        Args:
            node: Node id

        Returns: Node driving distance of each node to its neighbourhood pandas DataFrame row

        """
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
        """
        Driving distance DataFrame
        """
        df = pd.DataFrame()
        start = time.time()
        for i, node in enumerate(self.nodes):
            print(f'Starting teration {i}.  Node {node}.    Time {(time.time() - start) / 60} minutes')
            df = df.append(self.node_distance(node))
        return df[self.nodes]

    def save_node_distance(self, graph_path):
        """
        Save data path

        Args:
            path: Save data path
        """
        self.node_distance_df.to_csv(f'{graph_path}/node_distance.csv', sep='|')


class GraphCreator:
    """
    Class to create Graphs

    Args:
        distance_matrix_path: Path to distance matrix DataFrame
        nodes: Node list
        k: Distance threshold
        sim: Symmetric or not
        max_connections: Maximum number of connections between nodes
    """

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
        """
        Distance Matrix DataFrame
        """
        df = pd.read_csv(self.distance_matrix_filename, sep='|', index_col=0)
        df.columns = df.columns.astype(int)
        df.sort_index(inplace=True)
        return df[df.index]

    def set_section_nodes(self, position_path, bottom=0.0, top=99999999.0, left=0.0, right=99999999.0):
        """
        Spatial section for graph nodes

        Args:
            position_path: Read position data path
            bottom: bottom threshold
            top: top threshold
            left: left threshold
            right: right threshold

        """
        position_df = pd.read_csv(f'{position_path}/position.csv', sep='|')
        df = position_df[
            position_df.utm_y.between(bottom, top) & position_df.utm_x.between(left, right)]
        self._nodes = list(df['id'].unique())
        self._nodes.sort()

    @property
    def nodes(self):
        """
        Curated node list
        """
        if self._nodes is None:
            return list(self.distance_matrix.index)
        else:
            return self._nodes

    @cached_property
    def section_distance(self):
        """
        Section distance matrix DataFrame
        """
        return self.distance_matrix.loc[self.nodes, self.nodes]

    @cached_property
    def node_distance(self):
        """
        Top driving distance to each node
        """
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
        """
        Minimal Maxima distance in order to achieve a full connected graph.
        """
        df = self.node_distance.replace(0, np.nan)
        min_org = df.min()
        min_end = df.min(axis=1)
        df = pd.DataFrame({'org': min_org, 'end': min_end})
        min_tot = df.min(axis=1)
        return min_tot.max()

    @cached_property
    def k(self):
        """
        Distance threshold
        """
        if self._k is None:
            return self.min_max_distance
        else:
            return self._k

    @cached_property
    def distance_std(self):
        """
        Distance standard deviation
        """
        dist = self.node_distance.values.reshape(-1)
        dist = [x for x in dist if (x == x) & (x != 0)]
        return np.std(dist)

    @cached_property
    def weighted_adjacency_matrix(self):
        """
        Weighted adjacency matrix DataFrame
        """
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
        """
        Adjacency matrix DataFrame
        """
        return self.weighted_adjacency_matrix.astype(bool).astype(int)

    @cached_property
    def edge_index(self):
        """
        Edge connection array
        """
        edge_list = list()
        for i, row in self.adjacency_matrix.iterrows():
            for col in self.adjacency_matrix.columns:
                if row[col]:
                    v = np.array((i, int(col)))
                    edge_list.append(v)
        return np.array(edge_list)

    @cached_property
    def edge_attributes(self):
        """
        Edge connection weight array
        """
        attributes_list = list()
        for i, row in self.weighted_adjacency_matrix.iterrows():
            for col in self.weighted_adjacency_matrix.columns:
                if row[col]:
                    attributes_list.append(row[col])
        return np.array(attributes_list)

    def save_weighted_adjacency_matrix(self, adjacency_path):
        """
        Save adjacency matrix data path

        Args:
            adjacency_path: Save data path
        """
        adjency_name = 'adjacency' if self.max_connections is None else f'adjacency_{self.max_connections}'
        if self.sim:
            filename = f'{adjacency_path}/{adjency_name}_sim.csv'
        else:
            filename = f'{adjacency_path}/{adjency_name}.csv'
        self.weighted_adjacency_matrix.to_csv(filename, sep='|')

    def save_edge_attributes(self, pickle_path):
        """
        Save edge connection weight data path

        Args:
            pickle_path: Save data path
        """
        pickle_filename = f'{pickle_path}/edge_attributes.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.edge_attributes, f)

    def save_edge_index(self, pickle_path):
        """
        Save edge connection data path

        Args:
            pickle_path: Save data path
        """
        pickle_filename = f'{pickle_path}/edge_index.pickle'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.edge_index, f)
