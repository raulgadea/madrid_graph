from functools import cached_property
from datetime import datetime
from prophet import Prophet
from multiprocessing import Pool, cpu_count

import dask.dataframe as dd
import pandas as pd


class DataImputation:
    """
    Data class to detect faulty measures, delete them and impute them.


    Args:
        origin_path: Read historic data path
        start_time: Start period
        end_time: End period
        zero_threshold: Number of repeated zero values to detect
        cte_threshold: Number of repeated non zero values to detect
        freq: Imputation frequency
        per: Percentage of completion before discarding node
        nodes: Node list
    """

    def __init__(self, origin_path, start_time=None, end_time=None, zero_threshold=6 * 4, cte_threshold=8, freq='15min',
                 per=0.95, nodes=None):
        self.origin_path = origin_path
        self._nodes = nodes
        self._start_time = start_time
        self._end_time = end_time
        self.zero_threshold = zero_threshold
        self.cte_threshold = cte_threshold
        self.freq = freq
        self.per = per

    @cached_property
    def nodes(self):
        """
        Eligible nodes
        """
        return list(self.corrected_measures_df['id'].unique())

    @cached_property
    def start_time(self):
        """
        Start time in datetime format
        """
        return datetime.strptime(self._start_time,
                                 '%d/%m/%Y') if self._start_time is not None else self.raw_data.date.min().compute()

    @cached_property
    def end_time(self):
        """
        End time in datetime format
        """
        return datetime.strptime(self._end_time,
                                 '%d/%m/%Y') if self._end_time is not None else self.raw_data.date.max().compute()

    @cached_property
    def date_range(self):
        """
        Pandas date range
        """
        return pd.date_range(self.start_time, self.end_time, freq=self.freq).to_frame(name='ds').reset_index(drop=True)

    @cached_property
    def date_len(self):
        """
        Number of timestamps
        """
        return len(self.date_range)

    @cached_property
    def min_measures(self):
        """
        Minimum measure before discarding node
        """
        return self.per * self.date_len

    @cached_property
    def raw_data(self):
        """
        Original historic data as Dask DataFrame
        """
        return dd.read_parquet(f'{self.origin_path}/historic_lf.parquet')

    @cached_property
    def selected_data(self):
        """
        Filter time and nodes
        """
        time_mask = (self.raw_data['date'] >= self.start_time) & (self.raw_data['date'] < self.end_time)
        if self._nodes is not None:
            df = self.raw_data[self.raw_data['id'].isin(self._nodes) & time_mask].compute()
        else:
            df = self.raw_data[time_mask].compute()
        return df.sort_values(by=['id', 'date']).reset_index(drop=True)

    @cached_property
    def faulty_measures(self):
        """
        Number of faulty measures per node
        """
        data = self.selected_data.copy()
        v = data.groupby('id').apply(lambda a: (a['y'].diff(1) != 0).astype('int').cumsum()).values
        data['value_grp'] = v
        data['zero'] = ~data['y'].astype(bool)
        repeat_measures = pd.DataFrame({'begin_date': data.groupby(['id', 'zero', 'value_grp']).date.first(),
                                        'end_date': data.groupby(['id', 'zero', 'value_grp']).date.last(),
                                        'consecutive': data.groupby(
                                            ['id', 'zero', 'value_grp']).size()}).reset_index().drop(
            columns='value_grp')
        zero_cond = (repeat_measures['consecutive'] > self.zero_threshold) & (repeat_measures['zero'] is True)
        cte_cond = (repeat_measures['consecutive'] > self.cte_threshold) & (repeat_measures['zero'] is not True)
        faulty_measures = repeat_measures[zero_cond | cte_cond]
        return faulty_measures

    @cached_property
    def corrected_measures_df(self):
        """
        Curated data after deleting faulty measures
        """
        if self.faulty_measures.empty:
            return self.selected_data
        else:
            cond = self.faulty_condition(self.faulty_measures)
            df = self.selected_data[~cond].dropna()
            measures_per_node = df[['id', 'date']].groupby('id').size().reset_index(name='n')
            measures_per_node = measures_per_node[measures_per_node['n'] > self.min_measures]
            nodes = list(measures_per_node.id.unique())
            return df[df['id'].isin(nodes)].reset_index(drop=True)

    def faulty_condition(self, df):
        """
        Detect faulty conditions

        Args:
            df: Faulty measures

        Returns: Faulty conditions

        """
        cond = False
        for idx, row in df.iterrows():
            cond = ((self.selected_data['id'] == row['id']) & (self.selected_data.date >= row['begin_date']) & (
                    self.selected_data.date <= row['end_date'])) | cond
        return cond

    @cached_property
    def node_dict(self):
        """
        Dictionary historic data per node
        """
        data = self.corrected_measures_df.copy()
        data.rename(columns={'date': 'ds'}, inplace=True)
        dfs = {node: data[data.id == node].drop(columns=['id', 'tipo_elem']).sort_values(by='ds').reset_index(
            drop=True) for node in self.nodes}
        return dfs

    def prophet_imputation(self, item):
        """
        Imputation function

        Args:
            item: Node historc dictionary

        Returns: imputed dataframe

        """
        node, df = item
        if df.shape[0] == self.date_range.shape[0]:
            return df
        miss_df = pd.merge(self.date_range, df[['ds']], on='ds', how="outer", indicator=True)
        miss_df = miss_df[miss_df['_merge'] == 'left_only'][['ds']].reset_index(drop=True)
        try:
            m = Prophet(yearly_seasonality=True).fit(df)
            miss_df = m.predict(miss_df)[['ds', 'yhat']].rename(columns={'yhat': 'y'})
            miss_df.loc[miss_df['y'] < 0, 'y'] = 0
            impute_df = pd.concat([df, miss_df]).sort_values(by='ds').reset_index(drop=True)
            impute_df['id'] = node
            return impute_df
        except:
            print(f"Error in node : {node}")

    @cached_property
    def imputed_data(self):
        """
        Parallelize imputed data
        """
        dfs = self.node_dict.copy()
        impute_df_nodes = pd.DataFrame()
        with Pool(processes=cpu_count()) as pool:
            imputed_nodes = pool.map(self.prophet_imputation, list(dfs.items()))
        for inputed_node in imputed_nodes:
            impute_df_nodes = pd.concat([impute_df_nodes, inputed_node])
        impute_df_nodes.rename(columns={'ds': 'date'}, inplace=True)
        return impute_df_nodes.sort_values(by=['id', 'y'])[['id', 'date', 'y']]

    def save(self, path):
        """
        Save data path

        Args:
            path: Save data path
        """
        self.imputed_data.to_parquet(f'{path}/historic.parquet')
