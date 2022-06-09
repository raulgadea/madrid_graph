from functools import cached_property
from datetime import datetime
from prophet import Prophet
from multiprocessing import Pool, cpu_count

import dask.dataframe as dd
import pandas as pd


class DataImputation:
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
        return list(self.corrected_measures_df['id'].unique())

    @cached_property
    def start_time(self):
        return datetime.strptime(self._start_time,
                                 '%d/%m/%Y') if self._start_time is not None else self.raw_data.date.min().compute()

    @cached_property
    def end_time(self):
        return datetime.strptime(self._end_time,
                                 '%d/%m/%Y') if self._end_time is not None else self.raw_data.date.max().compute()

    @cached_property
    def date_range(self):
        return pd.date_range(self.start_time, self.end_time, freq=self.freq).to_frame(name='ds').reset_index(drop=True)

    @cached_property
    def date_len(self):
        return len(self.date_range)

    @cached_property
    def min_measures(self):
        return self.per * self.date_len

    @cached_property
    def raw_data(self):
        return dd.read_parquet(f'{self.origin_path}/historic_lf.parquet')

    @cached_property
    def selected_data(self):
        time_mask = (self.raw_data['date'] >= self.start_time) & (self.raw_data['date'] < self.end_time)
        if self._nodes is not None:
            df = self.raw_data[self.raw_data['id'].isin(self._nodes) & time_mask].compute()
        else:
            df = self.raw_data[time_mask].compute()
        return df.sort_values(by=['id', 'date']).reset_index(drop=True)

    @cached_property
    def faulty_measures(self):
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
        cond = False
        for idx, row in df.iterrows():
            cond = ((self.selected_data['id'] == row['id']) & (self.selected_data.date >= row['begin_date']) & (
                    self.selected_data.date <= row['end_date'])) | cond
        return cond

    @cached_property
    def node_dict(self):
        data = self.corrected_measures_df.copy()
        data.rename(columns={'date': 'ds'}, inplace=True)
        dfs = {node: data[data.id == node].drop(columns=['id', 'tipo_elem']).sort_values(by='ds').reset_index(
            drop=True) for node in self.nodes}
        return dfs

    def prophet_imputation(self, item):
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
        dfs = self.node_dict.copy()
        impute_df_nodes = pd.DataFrame()
        with Pool(processes=cpu_count()) as pool:
            imputed_nodes = pool.map(self.prophet_imputation, list(dfs.items()))
        for inputed_node in imputed_nodes:
            impute_df_nodes = pd.concat([impute_df_nodes, inputed_node])
        impute_df_nodes.rename(columns={'ds': 'date'}, inplace=True)
        return impute_df_nodes.sort_values(by=['id', 'y'])[['id', 'date', 'y']]

    def save(self, path):
        self.imputed_data.to_parquet(f'{path}/historic.parquet')


if __name__ == "__main__":
    # origin_path = '../../data/traffic/historic/m30_lf'
    # save_path = '../../data/traffic/historic/m30_lc'
    # origin_path = '../../data/traffic/historic/lf_15'
    # save_path = '../../data/traffic/historic/lc_15'
    origin_path = '../../data/traffic/historic/m30_speed_aux_lf'
    save_path = '../../data/traffic/historic/m30_speed_aux_lc2'
    start_time = '01/01/2018'
    end_time = '01/01/2019'
    s = DataImputation(origin_path=origin_path, start_time=start_time, end_time=end_time)

    # print(len(s.nodes))
    # print(s.date_range)
    #
    # df = s.imputed_data
    # print(df[df.isnull().any(axis=1)])
    s.save(save_path)
