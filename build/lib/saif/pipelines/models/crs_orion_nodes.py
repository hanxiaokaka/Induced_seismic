import pandas as pd
import numpy as np

# import time
from saif.orion_utils.pressure_table import PressureTableModel
from saif.orion_utils.seismic_catalog import SeismicCatalog


class CRSModel():
    """
    CRS forecast model

    Attributes:
        active (bool): Flag to indicate whether the model is active
        weight (float): Model relative weight
        pressure_method (str): Pressure calculation method
        forecast_time (ndarray): Time values for forecast calculation
    """

    def __init__(self, **kwargs):
        """
        CRS model initialization

        """
        # Call the parent's initialization
        super().__init__(**kwargs)

        # Initialize model-specific parameters
        self.short_name = 'CRS'
        self.long_name = 'Coupled Coulomb Rate-State Model'
        self.forecast_time = []

        # User-provided variables used in the forecast and
        # Not included in the optimization
        # These should be identified by the user a-priori
        # **Note** the background_rate_input could conceivable be found
        # by looking at the long-term rate (i.e. 20+ years before current)
        # of seismicity in the region using the USGS ComCat catalog
        self.background_rate_input = 1.36  # events/year
        self.biot = 0.3
        self.sigma_input = 30  # MPa
        self.tectonicNormalStressingRate_input = 0  # MPa/year
        self.tectonicShearStressingRate_input = 3.5e-4  # MPa/year

        # Best-fitting hyperparameters for Cushing 2014 use-case example
        # These parameters should be determined via optimization algorithm
        self.alpha = 0.155  # normal stress parameter in rate-state formulation
        self.mu = 0.73  # nominal coefficient of friction in rate-state eq.
        self.rate_coefficient = 0.00264  # 'a' parameter in rate-state eq.
        # Correction term added to equate the observed and forecasted number of events
        # This parameter adjusts the forecasted-N due to uncertainties in estimating
        # The background tectonic stressing rate term.
        self.rate_factor_input = 163.7424  # 1/MPa;

        # This magnitude threshold is arbitrarily set for the Cushing 2014
        # use-case because, when filtering events above this magnitude, it
        # it results in a relatively small number of events that correspond
        # With the onset of the changes in seismicity rate, when "clustering"
        # (e.g. mainshock/aftershock sequences) occur in this catalog.
        # Mainshock/aftershock sequences are only considered if
        # enable_cluster = True
        self.enable_clustering = True
        self.target_magnitude = 3.2  # if clustering is enabled

        # Ititialize parameters where input units will be modified to code-specific units
        # through the process_inputs method.
        self.background_rate = -999
        self.rate_factor = -999
        self.sigma = -999
        self.tectonicNormalStressingRate = -999
        self.tectonicShearStressingRate = -999

        self.process_inputs()

    def process_inputs(self):
        """
        Process any required gui inputs
        """
        mpa_yr2pa_s = 1e6 / 365.25 / 86400  # MPa/year to Pa/s
        self.sigma = self.sigma_input * 1e6  # Pa
        self.tectonicShearStressingRate = self.tectonicShearStressingRate_input * mpa_yr2pa_s  # Pa/s
        self.tectonicNormalStressingRate = self.tectonicNormalStressingRate_input * mpa_yr2pa_s  # Pa/s
        self.background_rate = self.background_rate_input / 365.25 / 86400  # event/second
        self.rate_factor = self.rate_factor_input / 1e6  # 1/Pa

    def computeSigmaEffective(self, pressure):
        """
        Effective (time-dependent) normal stress that varies with pressure

        Args:
            sigma (float): initial normal stress (Pa)
            biot (float): biot coefficient
            pressure (np.array): pore-fluid pressure time series at a point (Pa)

        Returns:
            np.array: effective normal stress time series
        """
        return (self.sigma - (self.biot * pressure))

    def computeCoulombStressingRate(self, pressureRate):
        """
        Effective (time-dependent) normal stress that varies with pressure

        Args:
            pressureRate (np.array): pore-fluid pressurization rate time
                series at a point (MPa)

        Returns:
            np.array: Coulomb stressing rate time series
        """
        return (self.tectonicShearStressingRate - (
                    (self.mu - self.alpha) * (self.tectonicNormalStressingRate - pressureRate)))

    def instantRateChange(self, static_coulomb_stress_change, background_rate,
                          rate_coefficient, sigma_effective):
        """
        Instantaneous change in Rate (R) due to step change in Coulomb stress

        Args:
            static_coulomb_stress_change (float): Coulomb stress step (Pa by default)
            background_rate (float): long-term background seismicity rate before stress step (seconds by default)
            rate_coefficient (float): rate-coefficient in rate-state formulation
            sigma_effective (np.array): effective normal stress (Pa; sigma_effective = sigma - biot*pressure)

        Returns:
            float: Instantaneous change in rate
        """
        asigma = rate_coefficient * sigma_effective
        return (background_rate * np.exp(static_coulomb_stress_change / asigma))

    def interseismicRate(self, forecast_time, eta, coulomb_stressing_rate, rate_at_prev_step,
                         rate_coefficient, sigma_effective):
        """
        Evolution of the Rate during a time of constant stressing rate

        Args:
            forecast_time (np.array): Times at which number should be computed (units:
                            same as the time units in sigma_effective; seconds by default)
            eta (float): reference stressing rate divided by background rate (steady
                         event rate that would be produced by constant stressing at the
                         reference stressing rate) (units: stress/event with stress
                         in same units as the stress units in sigma_effective; Pa/second by default)
            coulomb_stressing_rate(np.array): Constant Coulomb stressing rate (units: same stress units
                               as used in eta, same time units as t; Pa/second by default)
            rate_at_prev_step (float): event rate at the previous time step (units: events/time, same time units
                        as forecast_time (seconds by default))
            rate_coefficient (float): rate-coefficient in rate-state formulation
            sigma_effective (np.array): effective normal stress (Pa; sigma_effective = sigma - biot*pressure)

        Returns:
            np.array: interseismic rate
        """
        # for simplicity and because these terms should always vary together, define new variable
        asigma = rate_coefficient * sigma_effective
        if (coulomb_stressing_rate == 0):
            return (1 / (1 / rate_at_prev_step + eta * forecast_time / asigma))
        else:
            x = np.exp(-coulomb_stressing_rate * forecast_time / asigma)
            return (1 / (eta / coulomb_stressing_rate + (1 / rate_at_prev_step - eta / coulomb_stressing_rate) * x))

    def interseismicNumber(self, forecast_time, eta, coulomb_stressing_rate,
                           rate_at_prev_step, rate_coefficient, sigma_effective):
        """
        Compute expected total number of events during a time of constant stressing rate

        Args:
           forecast_time (np.array): np.array of times at which number should be computed
                         (units: same as the time units in sigma_effective; seconds by default)
           eta (float): reference stressing rate divided by background rate (steady
                        event rate that would be produced by constant stressing at the
                        reference stressing rate) (units: stress/event with stress
                        in same units as the stress units in sigma_effective; Pa/second by default)
           coulomb_stressing_rate(np.array): np.array of constant Coulomb stressing rate (units: same stress units
                                             as used in eta, same time units as forecast_time; Pa/second by default)
           rate_at_prev_step (float): event rate at the previous time step (units: events/time, same time units
                                      as forecast_time (seconds by default))
           rate_coefficient (float): rate-coefficient in rate-state formulation
           sigma_effective (np.array): effective normal stress (Pa; sigma_effective = sigma - biot*pressure)

        Returns:
           np.array: interseismic number
        """
        asigma = rate_coefficient * sigma_effective
        if (coulomb_stressing_rate == 0):
            return ((asigma / eta) * np.log(1 + rate_at_prev_step * eta * forecast_time / asigma))
        else:
            x = np.exp(coulomb_stressing_rate * forecast_time / asigma)
            x *= (rate_at_prev_step * eta / coulomb_stressing_rate)
            return ((asigma / eta) * np.log((1 - rate_at_prev_step * eta / coulomb_stressing_rate) + x))

    def rateEvolution(self, forecast_time, large_event_times_in_forecast,
                      static_coulomb_stress_change, coulomb_stressing_rate, background_rate, rate_factor,
                      rate_coefficient, sigma_effective):
        """
        Calculate the earthquake rate for all times passed into t.
        t and large_event_times_in_forecast  must all have the same units of time.
        for consistency, we will use "years" as the standard time unit.

        Instaneous stress step vector (static_coulomb_stress_change) must be [length(sigma_effective) - 1]
        (stressing rate vector) beginning at the time of the first change

        Args:
            forecast_time (np.array): Times at which number should be computed
                          (units: same as the time units in sigma_effective; seconds by default)
            large_event_times_in_forecast (np.array): Times of the stress steps (seconds by default)
            static_coulomb_stress_change (np.array): Amplitude of the stress steps (+/- ; Pa by default)
                           must be the same length as large_event_times_in_forecast
            coulomb_stressing_rate: (np.array): constant Coulomb stressing rate (units: same stress units
                              as used in eta, same time units as t; Pa/second by default)
                              len(sigma_effective) must equal len(forecast_time)
            background_rate (float): Event rate at time forecast_time=0 (units: events/time, same time units
                        as forecast_time (seconds by default))
            rate_factor (float): 1/eta; the background seismicity rate divided by the background stressing rate
            rate_coefficient (float): rate-coefficient in rate-state formulation
            sigma_effective (np.array): effective normal stress (Pa; sigma_effective = sigma - biot*pressure)

        Returns:
           np.array: Rate evolution
        """
        R = np.empty(len(forecast_time))
        eta = 1 / rate_factor

        if (self.enable_clustering):
            if len(forecast_time) != len(sigma_effective):
                self.logger.error("forecast_time must be the same length as sigma_effective")
                return

            if len(static_coulomb_stress_change) != len(large_event_times_in_forecast):
                self.logger.error(
                    "large_event_times_in_forecast must be the same length as static_coulomb_stress_change")
                return

            # Separate the time array into individual periostatic_coulomb_stress_change defining the interseismic period
            # and the time of the stress steps
            index = np.digitize(forecast_time,
                                bins=large_event_times_in_forecast,
                                right=False)
            R[index == 0] = background_rate

            # Loop over all times, t, check if t[i] is the time of a stress step. If it is,
            # compute the instantaneous rate change. If it is not, then compute the gamma evolution
            # due to changes in sigma_effective
            for i in range(1, len(forecast_time)):
                if not (index[i] == index[i - 1]):
                    R[i] = self.instantRateChange(static_coulomb_stress_change[index[i - 1]],
                                                  R[i - 1],
                                                  rate_coefficient,
                                                  sigma_effective[i])
                else:
                    R[i] = self.interseismicRate(forecast_time[i] - forecast_time[i - 1],
                                                 eta,
                                                 coulomb_stressing_rate[i],
                                                 R[i - 1],
                                                 rate_coefficient,
                                                 sigma_effective[i])
        else:
            R[0] = background_rate
            for i in range(1, len(forecast_time)):
                R[i] = self.interseismicRate(forecast_time[i] - forecast_time[i - 1],
                                             eta,
                                             coulomb_stressing_rate[i],
                                             R[i - 1],
                                             rate_coefficient,
                                             sigma_effective[i])
        return (R)

    def numberEvolution(self, forecast_time, large_event_times_in_forecast,
                        static_coulomb_stress_change, coulomb_stressing_rate, background_rate,
                        rate_factor, rate_coefficient, sigma_effective):
        """
        Calculate the number of events for all times passed into forecast_time.
        forecast_time and large_event_times_in_forecast  must all have the same units of time.
        for consistency, we will use "seconds" as the standard time unit.

        Instaneous stress step vector (static_coulomb_stress_change) must be [length(sigma_effective) - 1]
        (stressing rate vector) beginning at the time of the first change

        Args:
            forecast_time (np.array): Times at which number should be computed
                                      (units same as the time units in sigma_effective; seconds by default)
            large_event_times_in_forecast (np.array): Times of the stress steps (seconds by default)
            static_coulomb_stress_change (np.array): Amplitude of the stress steps (+/- ; Pa by default)
                                                     must be the same length as large_event_times_in_forecast
            coulomb_stressing_rate(np.array): Constant Coulomb stressing rate (units: same stress units
                                              as used in eta, same time units as t; Pa/second by default)
            background_rate (float): Event rate at time forecast_time=0 (units: events/time, same time units
                                     as forecast_time (seconds by default))
            rate_factor (float): 1/eta; the background seismicity rate divided by the background stressing rate
            rate_coefficient (float): rate-coefficient in rate-state formulation
            sigma_effective (np.array): effective normal stress (Pa; sigma_effective = sigma - biot*pressure,
                                        same length as forecast_time)

        Returns:
            np.array: Number evolution
        """
        N = np.empty(len(forecast_time))
        R = np.empty(len(forecast_time))
        eta = 1 / rate_factor

        if (self.enable_clustering):
            if len(forecast_time) != len(sigma_effective):
                # self.logger.error("forecast_time must be the same length as sigma_effective")
                return 0

            if len(static_coulomb_stress_change) != len(large_event_times_in_forecast):
                # self.logger.error("large_event_times_in_forecast must be the same length as static_coulomb_stress_change")
                return 0

            # Separate the time array into individual periods defining the interseismic period
            # and the time of the stress steps
            index = np.digitize(forecast_time,
                                bins=large_event_times_in_forecast,
                                right=False)

            useT = np.where(index == 0)
            R[useT] = background_rate
            N[useT] = background_rate * (forecast_time[useT] - forecast_time[0])

            # Loop over all forecast_times
            # 1) compute the instataneous rate at the time of each large_events
            #    1a) use sigma_effective at the time of the large event
            # 2) compute the aftershock decay until the time of the next large event
            # 3) update the current rate (rCurrent) and nCurrent at the time step
            #    just before the next large event to be used in the next iteration
            #    3a) use the coulomb_stressing_rate and sigma_effective at the time step
            #       just before the next large event
            for i in range(1, len(forecast_time)):
                if not (index[i] == index[i - 1]):
                    R[i] = self.instantRateChange(static_coulomb_stress_change[index[i - 1]],
                                                  R[i - 1],
                                                  rate_coefficient,
                                                  sigma_effective[i])
                    N[i] = N[i - 1] + self.interseismicNumber(forecast_time[i] - forecast_time[i - 1],
                                                              eta,
                                                              coulomb_stressing_rate[i],
                                                              R[i - 1],
                                                              rate_coefficient,
                                                              sigma_effective[i])
                else:
                    R[i] = self.interseismicRate(forecast_time[i] - forecast_time[i - 1],
                                                 eta,
                                                 coulomb_stressing_rate[i],
                                                 R[i - 1],
                                                 rate_coefficient,
                                                 sigma_effective[i])
                    N[i] = N[i - 1] + self.interseismicNumber(forecast_time[i] - forecast_time[i - 1],
                                                              eta,
                                                              coulomb_stressing_rate[i],
                                                              R[i - 1],
                                                              rate_coefficient,
                                                              sigma_effective[i])
        else:
            R[0] = background_rate
            N[0] = background_rate * (forecast_time[1] - forecast_time[0])

            for i in range(1, len(forecast_time)):
                R[i] = self.interseismicRate(forecast_time[i] - forecast_time[i - 1],
                                             eta,
                                             coulomb_stressing_rate[i],
                                             R[i - 1],
                                             rate_coefficient,
                                             sigma_effective[i])
                N[i] = N[i - 1] + self.interseismicNumber(forecast_time[i] - forecast_time[i - 1],
                                                          eta,
                                                          coulomb_stressing_rate[i],
                                                          R[i - 1],
                                                          rate_coefficient,
                                                          sigma_effective[i])
        return (N)

    def generate_forecast(self, forecast_time, seismic_catalog,
                          pressure, pressurization_rate):

        self.forecast_time = forecast_time

        if (self.enable_clustering):

            # Mininum target_magnitude is currently hard-coded, but should be exposed to the user
            # Time_window is training period to use for parameter optimization.
            # This is currently set to be the start of the seismic catalog and for nearly 2 years.
            # We need to determine this automatically and then understand the impact of this
            # selection on the parameter uncertainty.
            time_window = [44312204, 1.9 * 86400 * 365.25] + np.amin(
                seismic_catalog.epoch)  # Need to figure out a way to do this automatically
            # Currently, the time period to look for the mainshocks is hard-coded and should
            # be updated so that mainshcoks above a given magntiude are automatrically selected.
            minimum_interevent_time = 0.02 * 86400 * 365.25
            # Minimim time between mainshcoks is currently hard-coded and COULD eventually be
            # exposed to user or a default set.
            maximum_magnitude = 10
            seismic_catalog.set_slice(time_range=time_window,
                                      magnitude_range=[self.target_magnitude, maximum_magnitude],
                                      minimum_interevent_time=minimum_interevent_time)
            large_event_times_in_catalog = seismic_catalog.get_epoch_slice()
            # Currently, a static stress step is being added at the beginning of the injection period
            # (at the second time step) in order to offset seismicity rates above or below steady-state
            # to account for unknown initial conditions.
            # large_event_magnitudes = seismic_catalog.get_magnitude_slice()

            # Make sure to set the time of the large_event_times to the closest value
            # in the forecast_times
            large_event_times_in_forecast = []
            for time in large_event_times_in_catalog:
                large_event_times_in_forecast = np.append(large_event_times_in_forecast,
                                                          self.forecast_time[
                                                              np.abs(time - self.forecast_time).argmin()])

            # Add sample at the beginning of the pressure time series to account for offset from
            # steady-state at the beginning of the operation
            # Note, this sample cannot be at forecast_time[0], so make it one time step after forecast_time[0]
            dt = self.forecast_time[1] - self.forecast_time[0]
            large_event_times_in_forecast = np.insert(large_event_times_in_forecast, 0, self.forecast_time[0] + dt)

            # Set the initial static Coulomb Stress chane to be some fraction of
            # the event magnitude for now, but we may want to change this in the future.
            # Through trial and error, 0.1*M seems to work well.
            # coulomb_stress_change_by_percentage_magnitude = 0.1
            # static_coulomb_stress_change = coulomb_stress_change_by_percentage_magnitude*large_event_magnitudes
            # A more physical approach derived from a circular crack model is:
            # static_coulomb_stress_change = seismic_moment / 6 / pi / pow(radius,3) from Mancini et al., 2019
            static_coulomb_stress_change = np.array([-0.6477964, 0.5145350, 0.6200820]) * 1e6
            # Using hard-coded static Coulomb stress changes now for the Cushing 2014 sequence.
            # This will be updated when the parameter optimzation is added.
        else:
            # self.forecast_time = grid.t[:-1]
            large_event_times_in_forecast = -999
            # large_event_magntiudes = -999
            static_coulomb_stress_change = -999

        sigma_effective = self.computeSigmaEffective(pressure)
        coulomb_stressing_rate = self.computeCoulombStressingRate(pressurization_rate)

        N = self.numberEvolution(self.forecast_time,
                                 large_event_times_in_forecast,
                                 static_coulomb_stress_change,
                                 coulomb_stressing_rate,
                                 self.background_rate,
                                 self.rate_factor,
                                 self.rate_coefficient,
                                 sigma_effective)

        return (N)


def forecast_crs(
    pressure_table: PressureTableModel, seismic_catalog: SeismicCatalog
) -> pd.DataFrame:
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
    pressure_epoch = pressure_table.table_data["t"]
    pressure = pressure_table.table_data["pressure"]
    pressurization_rate = pressure_table.table_data["dpdt"]

    forecast_time = np.linspace(
        np.amin(pressure_epoch), np.amax(pressure_epoch), num=25000
    )

    N = CRSModel()
    cumulative_number = N.generate_forecast(
        forecast_time, seismic_catalog, pressure, pressurization_rate
    )

    return cumulative_number
