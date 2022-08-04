import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List, Tuple, Callable
####################################################################
def overlap_and_interpolate(seismic_df: pd.DataFrame, pressure_df: pd.DataFrame
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
    '''
    Find overlap between pressure and seismic data.
    Also, interpolate pressure and seismic data on to a temporal grid defined by the overlap.
    This function assumes that epochs of pressure and seismic data are supplied in seconds.

    Parameters:
    ----------------------
    seismic_df: pd.DataFrame
        Induced earthquake data

    pressure_df: pd.DataFrame
        Pressure and pressurization rate data

    Returns:
    ----------------------
    seismic_df: pd.DataFrame
        Seismic data in the overlap region

    resampled_pressure_df: pd.DataFrame
        Resampled pressure data in the overlap

    bins: 1D NumPy array
        Time bins (days) across resampled pressure data

    t0: float
       Start epoch (s) of data overlap
    '''
    # Find overlap.
    seismic_start = seismic_df['epoch'].min()
    seismic_end = seismic_df['epoch'].max()
    pressure_start = pressure_df['epoch'].min()
    pressure_end = pressure_df['epoch'].max()
    # Start time
    t0 = max(seismic_start, pressure_start)
    # End time
    t1 = min(seismic_end, pressure_end)

    # Create column 'days'.
    s2day = 1./86400 # seconds to days
    seismic_df['days'] = (seismic_df['epoch'] - t0) * s2day
    pressure_df['days'] = (pressure_df['epoch'] - t0) * s2day
    # Bin day counts between t0 and t1 (including both limits)
    N_days = (t1 - t0) * s2day
    bins = np.arange(0, math.ceil(N_days))
    # Truncate seismic data to overlap region.
    trunc_seismic_df = seismic_df.copy()
    trunc_seismic_df = trunc_seismic_df[trunc_seismic_df['days'].between(0., N_days, inclusive='both')]
    # Bin indices corresponding to different earthquake epochs
    trunc_seismic_df['tbin_idx'] = np.digitize(trunc_seismic_df['days'], bins, right=False)

    # Interpolate pressure (Pa) and pressurization rate (Pa/s).
    p_func = interp1d(pressure_df['days'], pressure_df['pressure'], kind='linear', fill_value= 'extrapolate')
    dpdt_func = interp1d(pressure_df['days'], pressure_df['dpdt'], kind='linear', fill_value='extrapolate')
    resampled_pressure_df = {'days':bins}
    resampled_pressure_df['pressure'] = p_func(bins)
    resampled_pressure_df['dpdt'] = dpdt_func(bins)
    resampled_pressure_df = pd.DataFrame(resampled_pressure_df)

    return trunc_seismic_df, resampled_pressure_df, bins, t0
####################################################################
def aggregate_seismic(seismic_df: pd.DataFrame, n_steps: int, features: List[str],
                      column_name: str ='days') -> Tuple[pd.DataFrame, np.ndarray]:
    '''
    Aggregate seismic data by specified bin name. Default is to aggregate by day of occurrence.

    Parameters:
    ----------------------
    seismic_df: pd.DataFrame
        Seismic data

    n_steps: integer
        No. of bins desired across range of values covered by seismic_df[column_name]

    features: list
        Data features to be aggregated

    column_name: str
       Column of seismic_df based on which aggregation will be done

    Returns:
    ----------------------
    output_df: pd.DataFrame
        Aggregated and binned data

    binned_counts: np.array
        Binned arthquake counts
    '''
    # Compute earthquake counts per time bin.
    seismic_counts = seismic_df[[column_name,'tbin_idx']].groupby('tbin_idx').agg('count')
    binned_counts = np.zeros(n_steps)
    binned_counts[seismic_counts.index.values-1] = seismic_counts[column_name].values

    # Aggregate feature data.
    seismic_features = seismic_df[features].groupby('tbin_idx').agg([np.mean, np.std]).fillna(0)
    seismic_features.columns = ['_'.join(col).strip() for col in seismic_features.columns.values]
    n_features = len(seismic_features.columns)
    output_vals = np.zeros((n_steps, n_features))
    output_vals[seismic_features.index.values-1] = seismic_features.values
    output_df = pd.DataFrame(output_vals, columns=seismic_features.columns)

    return output_df, binned_counts
####################################################################
def daily_seismic_and_interpolated_pressure(seismic_df: pd.DataFrame, pressure_df: pd.DataFrame,
                                            target_fn: Callable =np.cumsum) -> Tuple[pd.DataFrame, int, np.ndarray]:
    '''
    Aggregate seismic and pressure data onto a temporal grid quantized in days.

    Parameters:
    ----------------------
    seismic_df: pd.DataFrame
        Seismic data

    pressure_df: pd.DataFrame
        Pressure and pressurization rate data

    target_fn: Callable
        Target function to be applied on binned earthquake counts. Default is a cumulative sum.

    Returns:
    ----------------------
    data_df: pd.DataFrame
        Daily seismic and interpolated pressure data

    t0: float
        Start epoch (s) of data overlap

    target_vals: 1D NumPy array
        Values generated by calling target function on daily seismic event counts
    '''
    _features = ['depth', 'easting', 'northing', 'magnitude', 'tbin_idx']
    trunc_seismic_df, resampled_pressure_df, day_bins, t0 = overlap_and_interpolate(seismic_df, pressure_df)
    seismic_features, binned_eq_counts = aggregate_seismic(trunc_seismic_df, len(day_bins), _features, column_name='days')
    target_vals = target_fn(binned_eq_counts)

    data_df = pd.concat([seismic_features, resampled_pressure_df], axis=1)
    data_df['counts'] = binned_eq_counts

    return data_df, t0, target_vals
####################################################################
