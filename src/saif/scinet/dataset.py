import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class EarthquakeDataset(Dataset):
    def __init__(self, seismic, pressure, input_len, horizon, target_fn=np.cumsum):
        """
        seismic: pd.DataFrame
        pressure: pd.DataFrame
        input_len: length of the input time series
        horizon: number of future steps to forecast
        """
        #TODO: implement pre-processing to normalize northing, easting, etc
        self.input_len = input_len
        self.horizon = horizon
        seismic, pressure = self._normalize(seismic, pressure)
        seismic_features, target = self._align(seismic, pressure, target_fn)
        
        features = pd.concat([seismic_features, pressure], axis=1)
        
        self.feature_names = features.columns
        self.X = torch.FloatTensor(features.values)
        self.Y = torch.FloatTensor(target)
    
    def _normalize(self, seismic, pressure):
        #TODO: implement
        return seismic, pressure
    
    def _align(self, seismic, pressure, target_fn):
        # Aggregate the seismic events that occur between pressure readings
        seismic['epoch_bin'] = np.digitize(
            seismic.epoch.values, 
            pressure.epoch.values
        )

        _features = ['depth', 'easting', 'northing', 'magnitude', 'epoch_bin']
        
        # TODO: edge case, what if max(bin) > len(pressure) ?
        seismic = seismic[seismic.epoch_bin < len(pressure)]
        
        seismic_counts = seismic[['epoch','epoch_bin']].groupby('epoch_bin').agg('count').reset_index()
        seismic_counts = seismic_counts.rename(columns={'epoch' : 'bin_counts'})
        
        #TODO: you can use other stats here too
        #ex: quantiles, high order cumulants
        #the data I've seen so far, however, is too small to justify these
        seismic_features = seismic[_features].groupby('epoch_bin').agg([np.mean, np.std])
        seismic_features.columns = ['_'.join(col).strip() for col in seismic_features.columns.values]
        
        # Align with pressure data
        
        n_steps = len(pressure)
        n_features = len(seismic_features.columns)
        
        output_vals = np.zeros((n_steps, n_features))
        output_vals[seismic_features.index.values] = seismic_features.values
        
        output_df = pd.DataFrame(output_vals, columns=seismic_features.columns)
        
        target_vals = np.zeros((n_steps,))
        target_vals[seismic_counts.index.values] = seismic_counts.bin_counts
        target_vals = target_fn(target_vals)        
        
        return output_df, target_vals
    
    def _make_tensors(self, seismic_features, pressure):
        target = 'running_counts'
        pass
    
    def __getitem__(self, index):
        x_start = index
        x_end = x_start + self.input_len
        y_start = x_end + 1
        y_end = y_start + self.horizon
        
        return self.X[x_start:x_end], self.Y[y_start:y_end]
    
    def __len__(self):
        #TODO: double check off-by-one
        return len(self.Y) - self.input_len - self.horizon
        
        