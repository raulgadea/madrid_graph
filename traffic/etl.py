from datetime import datetime
from pathlib import Path
import re
import dask.dataframe as dd
import pandas as pd
import glob


def create_traffic_position_l1(read_path, save_path, year_list=None):
    """
    Create first position traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path
        year_list: Year list to process

    Returns: Processed position DataFrame

    """
    if year_list is None:
        filenames = [file for file in glob.glob(f'{read_path}/*/*.csv')]
    else:
        pattern_str = '|'.join(map(str, year_list))
        pattern = re.compile(f'.*({pattern_str}).*')
        filenames = [file for file in glob.glob(f'{read_path}/*/*.csv') if pattern.match(file)]

    dfs = {Path(filename).stem: pd.read_csv(filename, sep=';', encoding="ISO-8859-1") for filename in filenames}
    df_full = pd.DataFrame()
    for key, df in dfs.items():
        df_l1 = df.copy()
        df_l1 = df_l1.dropna(how='all')
        df_l1.rename(
            columns={'ï»¿tipo_elem': 'tipo_elem', 'st_x': 'utm_x', 'st_y': 'utm_y', 'X': 'utm_x', 'y': 'utm_y'},
            inplace=True)
        if df_l1.utm_x.dtype != float:
            df_l1['utm_x'] = df_l1['utm_x'].str.replace(',', '.').astype(float)
        if df_l1.utm_y.dtype != float:
            df_l1['utm_y'] = df_l1['utm_y'].str.replace(',', '.').astype(float)
        df_l1['month'] = int(key[-7:-5])
        df_l1['year'] = int(key[-4:])
        df_full = pd.concat([df_full, df_l1[['tipo_elem', 'id', 'utm_x', 'utm_y', 'month', 'year']]])
    df_full = df_full.reset_index(drop=True)
    df_full['id'] = df_full['id'].astype(int)
    df_full.loc[df_full['tipo_elem'] == 'M-30', 'tipo_elem'] = 'M30'
    df_full.to_csv(f'{save_path}/position_l1.csv', sep='|', index=False)
    return df_full


def create_traffic_position_l2(read_path, save_path):
    """
    Create second position traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path

    Returns: Processed position DataFrame

    """
    traffic_position = pd.read_csv(f'{read_path}/position_l1.csv', sep='|')
    sensor_pos = traffic_position[['id', 'utm_x', 'utm_y']].drop_duplicates()
    sensor_all_time = sensor_pos.groupby('id').size().reset_index(name='n').sort_values(by='n', ascending=False)
    id_position = list(sensor_all_time[sensor_all_time.n == 1].id.unique())

    sensor_time = traffic_position[['id', 'year', 'month']].drop_duplicates()
    sensor_time = sensor_time.groupby('id').size().reset_index(name='n').sort_values(by='n', ascending=False)
    files = len(traffic_position[['year', 'month']].drop_duplicates())
    id_time = list(sensor_time[sensor_time.n == files].id.unique())

    id_pos_time = set(id_position) & set(id_time)
    traffic_position = traffic_position[traffic_position['id'].isin(id_pos_time)]
    year_month_min = (traffic_position.year.astype(str) + '-' + traffic_position.month.astype(str)).min()
    year_month_max = (traffic_position.year.astype(str) + '-' + traffic_position.month.astype(str)).max()
    traffic_position = traffic_position[['id', 'tipo_elem', 'utm_x', 'utm_y']].drop_duplicates()
    traffic_position['year_month_min'] = year_month_min
    traffic_position['year_month_max'] = year_month_max
    traffic_position.to_csv(f'{save_path}/position_l2.csv', sep='|', index=False)
    return traffic_position


def create_traffic_historic_l1(read_path, save_path, year_list=None):
    """
    Create first historic traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path
        year_list: Year list to process

    Returns: Processed historic DataFrame

    """
    if year_list is None:
        filenames = [file for file in glob.glob(f'{read_path}/*/*.csv')]
    else:
        pattern_str = '|'.join(map(str, year_list))
        pattern = re.compile(f'.*({pattern_str}).*')
        filenames = [file for file in glob.glob(f'{read_path}/*/*.csv') if pattern.match(file)]

    traffic_historic = dd.read_csv(filenames, sep=';', assume_missing=True)
    traffic_historic.to_parquet(f'{save_path}/historic_l1.parquet', engine='pyarrow')
    return traffic_historic


def create_traffic_historic_l2(read_path, save_path, variable='intensidad'):
    """
    Create second historic traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path

    Returns: Processed historic DataFrame

    """
    traffic_historic = dd.read_parquet(f'{read_path}/historic_l1.parquet',
                                       columns=['fecha', 'id', 'tipo_elem', variable])
    traffic_historic['tipo_elem'] = traffic_historic['tipo_elem'].mask(
        traffic_historic['tipo_elem'] == 'PUNTOS MEDIDA M-30', 'M30')
    traffic_historic['tipo_elem'] = traffic_historic['tipo_elem'].mask(
        traffic_historic['tipo_elem'] == 'PUNTOS MEDIDA URBANOS', 'URB')
    traffic_historic['fecha'] = dd.to_datetime(traffic_historic['fecha'])
    traffic_historic['id'] = traffic_historic['id'].astype(int)
    traffic_historic = traffic_historic.categorize(columns=['tipo_elem'])
    traffic_historic = traffic_historic.rename(columns={'fecha': 'date', variable: 'y'})
    traffic_historic.to_parquet(f'{save_path}/historic_l2.parquet', engine='pyarrow')
    return traffic_historic


def agg_hourly(traffic_historic, split_out=128):
    """
    Aggregates data to hourly format.

    Args:
        traffic_historic: Historic DataFrame
        split_out: Dask partition

    Returns: Hourly aggregated data

    """
    traffic_historic = traffic_historic[traffic_historic.y >= 0].reset_index(drop=True)
    traffic_historic['ds'] = traffic_historic.date.dt.date
    traffic_historic['hour'] = traffic_historic.date.dt.hour
    traffic_historic = traffic_historic.groupby(['id', 'ds', 'hour']).agg(
        {'y': 'mean', 'tipo_elem': 'first'}, split_out=split_out, split_every=8).reset_index()
    traffic_historic['date'] = dd.to_datetime(traffic_historic.ds) + dd.to_timedelta(traffic_historic.hour, unit='h')
    traffic_historic = traffic_historic[['id', 'tipo_elem', 'date', 'y']]
    return traffic_historic


def create_traffic_lf(position_read_path, historic_read_path, position_save_path, historic_save_path, agg_hour=False):
    """
    Create second final traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path

    Returns: Processed historic DataFrame

    """
    traffic_position = pd.read_csv(f'{position_read_path}/position_l2.csv', sep='|')
    traffic_historic = dd.read_parquet(f'{historic_read_path}/historic_l2.parquet')

    traffic_historic = agg_hourly(traffic_historic) if agg_hour else traffic_historic

    year_month_min = traffic_position['year_month_min'].min()
    year_month_max = traffic_position['year_month_max'].min()

    id_pos = traffic_position['id'].to_list()
    traffic_historic = traffic_historic[traffic_historic['id'].isin(id_pos)]
    traffic_id_time = traffic_historic.copy()
    traffic_id_time['year_month'] = traffic_id_time['date'].dt.year.astype(str) + '-' + traffic_id_time[
        'date'].dt.month.astype(str)
    traffic_id_time = traffic_id_time[['id', 'year_month']].groupby('id').agg(['min', 'max'])[
        'year_month'].reset_index()
    traffic_id_time = traffic_id_time[
        (traffic_id_time['min'] == year_month_min) & (traffic_id_time['max'] == year_month_max)]

    id_historic = list(traffic_id_time['id'].unique())
    traffic_historic = traffic_historic[traffic_historic['id'].isin(id_historic)].reset_index(drop=True)
    traffic_position = traffic_position[traffic_position['id'].isin(id_historic)].reset_index(drop=True)

    traffic_position.to_csv(f'{position_save_path}/position_lf.csv', sep='|', index=False)
    traffic_historic.to_parquet(f'{historic_save_path}/historic_lf.parquet', engine='pyarrow')
    return traffic_historic, traffic_position


def create_traffic_position(position_read_path, historic_read_path, position_save_path):
    """
    Create final position traffic layer in parquet format

    Args:
        read_path: Read path
        save_path: Save path

    Returns: Processed position DataFrame

    """
    traffic_position = pd.read_csv(f'{position_read_path}/position_lf.csv', sep='|')
    traffic_historic = dd.read_parquet(f'{historic_read_path}/historic.parquet')
    nodes = list(traffic_historic.id.unique())

    traffic_position = traffic_position[traffic_position['id'].isin(nodes)].drop(
        columns=['year_month_min', 'year_month_max'])
    traffic_position.to_csv(f'{position_save_path}/position.csv', sep='|', index=False)
    return traffic_position


def load_data(path, dask=True):
    """

    Args:
        path: File path
        dask: is it a Dask or csv DataFrame

    Returns: Loaded data

    """
    return dd.read_parquet(path) if dask else pd.read_csv(path, sep='|')
