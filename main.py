from traffic import *

PATHS = {'position': {'l0': 'data/position/l0',
                      'l1': 'data/position/l1',
                      'l2': 'data/position/l2',
                      'lf': 'data/position/lf',
                      'm30': 'data/position/m30'},
         'historic': {'l0': 'data/historic/l0',
                      'l1': 'data/historic/l1',
                      'l2': 'data/historic/l2',
                      'lf': 'data/historic/lf',
                      'm30': 'data/historic/m30'},
         'graph': {'dist': 'data/graph/distance_matrix',
                   'adj': 'data/graph/adjacency_matrix'}
         }
if __name__ == "__main__":
    import dask.dataframe as dd

    year_list = ['2021', '2022']

    # create_traffic_position_l1(read_path=PATHS['position']['l0'], save_path=PATHS['position']['l1'],
    #                            year_list=year_list)
    # create_traffic_historic_l1(read_path=PATHS['historic']['l0'], save_path=PATHS['historic']['l1'])
    # create_traffic_historic_l2(read_path=PATHS['historic']['l1'], save_path=PATHS['historic']['l2'])
    # create_traffic_lf(position_read_path=PATHS['position']['l2'], historic_read_path=PATHS['historic']['l2'],
    #                   position_save_path=PATHS['position']['lf'], historic_save_path=PATHS['historic']['lf'],
    #                   agg_hour=False)
    historic_lf = load_data(path=PATHS['historic']['lf'], dask=True)
    m30_nodes = list(historic_lf[historic_lf.tipo_elem == 'M30'].id.unique())
    # m30_nodes = [1001, 3827]
    start_time = '01/01/2022'
    end_time = '01/02/2022'
    s = DataImputation(origin_path=PATHS['historic']['lf'], start_time=start_time, end_time=end_time, nodes=m30_nodes,
                       per=0.7)
    s.save(PATHS['historic']['m30'])
    create_traffic_position(position_read_path=PATHS['position']['lf'], historic_read_path=PATHS['historic']['m30'],
                            position_save_path=PATHS['position']['m30'])
