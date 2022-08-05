import math
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch

from saif.crs_model.peakdetector import pk_indxs
from saif.crs_model.utils import extract_sections

from saif.crs_model.batch_model import BatchCRSModel

from saif.crs_model.optimizer import generate_train_loss_fn
from saif.crs_model.optimizer import generate_test_loss_fn
from saif.crs_model.optimizer import ValidationCallback

from torchmin import minimize

def overlap(seismic, pressure):
    ## Find overlap
    seismic_start = seismic.epoch.values[0]
    seismic_end = seismic.epoch.values[-1]
    
    pressure_start = pressure.epoch.values[0]
    pressure_end = pressure.epoch.values[-1]
    
    #start time
    t0 = max(seismic_start, pressure_start)
    #end time
    t1 = min(seismic_end, pressure_end)
    
    #convert epoch to days
    ep2day = 1. / 86400
    seismic['days'] = (seismic.epoch - t0) * ep2day
    pressure['days'] = (pressure.epoch - t0) * ep2day
    
    dt = (t1 - t0) * ep2day
    
    seismic = seismic[
        seismic.days.between(0, dt)
    ].reset_index()
    
    pressure = pressure[
        pressure.days.between(0, dt)
    ].reset_index()
    
    return seismic, pressure

class CRSInterface():
    def __init__(
            self, site_info, seismic_df, pressure_df,
            train_frac=0.75,
            signal_to_partition='magnitude', 
            threshold=0.6, min_dist=100,
            optimizer_method='newton-exact',
            max_iter=50, disp=2, lr=1e-3
        ):

        self.site_info = site_info
        self.data_df = self.process_data(seismic_df, pressure_df)

        n_samples = len(self.data_df)
        self.train_cutoff = int(train_frac * n_samples)

        self.init_params = torch.FloatTensor([[0.5,1e-3,1e-4]])

        self.lower_bounds = torch.zeros(3)[None,:] + 1e-7
        self.upper_bounds = torch.ones(3)[None, :] - 1e-7
        
        self.model = BatchCRSModel(site_info)

        # Params for partitioning
        self.signal_to_partition = signal_to_partition
        self.threshold = threshold
        self.min_dist= min_dist

        # Params for optimization
        self.optimizer_method = optimizer_method
        self.max_iter = max_iter
        self.disp = disp
        self.lr = lr

    def process_data(self, seismic, pressure):
        background_rate = self.site_info['background_rate']

        seismic, pressure = overlap(seismic, pressure)

        n_func = interp1d(seismic.days, seismic.index.values, kind='linear')
        n_interpolated = n_func(pressure.days)

        mag_func = interp1d(seismic.days, seismic.magnitude.values, kind='linear')
        mag_interpolated = mag_func(pressure.days)

        data_df = pressure.copy()
        data_df['number'] = n_interpolated
        data_df['magnitude'] = mag_interpolated

        #Incorporate initial conditions of the site
        data_df['delta_t'] = np.diff(data_df.epoch, prepend=0)  #/ 86400 / 365.25
        data_df = data_df.iloc[1:]
        data_df.number += background_rate * data_df.delta_t.values[0]

        data_df['rate'] = np.gradient(data_df.number, data_df.epoch)

        return data_df

    def set_lower_bounds(self, lower_bounds):
        self.lower_bounds = lower_bounds

    def set_upper_bounds(self, upper_bounds):
        self.upper_bounds = upper_bounds

    def set_init_params(self, init_params):
        self.init_params = init_params

    @property
    def train_df(self):
        return self.data_df.iloc[:self.train_cutoff]

    @property
    def test_df(self):
        return self.data_df.iloc[self.train_cutoff:]

    def infer_partition(self):
        self.pks = pk_indxs(
            self.train_df[self.signal_to_partition], 
            trshd=self.threshold, 
            min_dist=self.min_dist
        )

        all_sections_idx = np.append(self.pks, self.train_cutoff)
        signal_starts, signal_sections = extract_sections(
            self.data_df, all_sections_idx
        ) 

        self.signal_starts = signal_starts
        self.signal_sections = signal_sections

    def fit(self):
        scale = self.train_df.number.values.std()

        param_projector, train_loss_fn = generate_train_loss_fn(
            self.signal_sections, self.signal_starts,
            self.model, self.lower_bounds, self.upper_bounds, scale
        )

        test_loss_fn = generate_test_loss_fn(
            self.signal_sections, self.signal_starts, self.model, 
            param_projector, scale
        )

        self.callback = ValidationCallback(train_loss_fn, test_loss_fn)

        init_pre_params = param_projector.inverse(self.init_params)

        result = minimize(
            train_loss_fn, init_pre_params, method=self.optimizer_method,
            max_iter=self.max_iter, disp=self.disp,
            options={'lr' : self.lr},
            callback=self.callback
        )
        self.result = result

        self.fitted_params = param_projector(result.x)
        return self.fitted_params

    def generate_forecast(self):
        mask = (self.signal_sections['number'] != -1).float()
        _, Nt = self.model.masked_forward(
            self.fitted_params, 
            p=self.signal_sections['pressure'], 
            dpdt=self.signal_sections['dpdt'], 
            delta_t=self.signal_sections['delta_t'], 
            R0=self.signal_starts['rate'],
            mask=mask
        )

        return Nt

    def compute_delta_cfs_terms(self):
        pass

    def plot_forecast(self):
        Nt = self.generate_forecast()

        fig, axs = plt.subplots(2,1,figsize=(12,8), sharex=True)
        axs[0].plot(self.train_df.days, self.train_df.rate)
        axs[0].plot(
            self.train_df.days.values[self.pks], 
            self.train_df.rate.values[self.pks],
            'o', color='purple'
        )
        axs[0].plot(
            self.test_df.days.values, 
            self.test_df.rate.values,
            color='blue'
        )

        axs[1].plot(self.train_df.days, self.train_df.number)
        axs[1].plot(self.test_df.days, self.test_df.number)

        axs[0].plot([], [], color='r', label='train')
        axs[0].plot([], [], color='b', label='test')
        axs[0].legend(loc='upper left')

        for i, ix in enumerate(reversed(range(self.signal_sections['days'].shape[1]))):
            steps = self.signal_sections['days'][:, ix, :].numpy() != 0
            axs[1].plot(
                self.signal_sections['days'][:, ix, :].numpy()[steps], 
                Nt[:, ix, :].data.numpy()[steps], color='g'
            )
            axs[1].plot(
                self.signal_starts['days'][0, ix], 
                self.signal_starts['number'][0, ix], 
                marker='o', color='purple'
            )

        ylim = axs[1].get_ylim()
        axs[1].vlines(
            self.train_df.days.values[self.pks], 
            ylim[0], ylim[1], color='purple', linestyle='--',
            label='peaks'
        )
        axs[1].set_ylim(ylim)

        axs[0].set_ylabel('Empirical rate')

        axs[1].set_ylabel('Empirical rate')
        axs[1].set_xlabel('Days')

        axs[1].legend(loc='upper left')

        plt.show()








    


