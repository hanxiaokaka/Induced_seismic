import pandas as pd
#import time
from saif.utils.pressure_table import PressureTableModel

def load_pressure_table(pressure_data: pd.DataFrame) -> PressureTableModel:
    """Create pressure table by loading from csv file

    Args:
        pressure_data: dataframe containing a pressure data
    Returns:
        A catalog instance
    """

    #data = {k: np.loadtxt(os.path.join(table_dir, '%s.csv' % (k))) for k in ['x', 'y', 'z', 't', 'pressure']}
    data = {columnName: columnData  for (columnName, columnData) in pressure_data.iteritems() }

    flow_model = PressureTableModel()
    flow_model.load_array(**data)

    return flow_model