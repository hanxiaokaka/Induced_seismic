import pandas as pd
#import time
from saif.utils.pressure_table import PressureTableModel
from saif.utils.seismic_catalog import SeismicCatalog

def forecast_crs(pressure_table: PressureTableModel, seismic_catalog: SeismicCatalog) -> pd.DataFrame:
    """Forecasting cumulative seismic events using the Coulomb Rate State model

    Args:
        pressure_table: pressure table in custom format
        seismic_catalog: seismic catalog in custom format
    Returns:
        Timeseries of cumulative forecasted seismic events
    """

    # unpack pressure and catalog information.
    """
        "Pressure" and "Pressurization Rate" (dpdt) are second order parameters.
        Some site-specific data only come with Injection Rate data which needs to
        be converted to pressure and dpdt with use of a reservoir model or
        the simple analytical Theis solution, provided along side this code. 
    """
    pressure_epoch = pressure_table.table_data['t']
    pressure = pressure_table.table_data['pressure']
    pressurization_rate = pressure_table.table_data['dpdt']

    forecast_time = np.linspace(np.amin(pressure_epoch), np.amax(pressure_epoch), num=25000)

    N = CRSModel()
    cumulative_number = N.generate_forecast(forecast_time, seismic_catalog, pressure, pressurization_rate)

    return cumulative_number