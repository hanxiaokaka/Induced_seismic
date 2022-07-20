import numpy as np
from scipy.special import exp1


class RadialFlowModel:
    """
    Pressure model based off of Theis Solution

    Attributes:
        viscosity (float): Fluid viscosity (cP)
        permeability (float): Matrix permeability (nD)
        storativity (float): Reservoir storativity factor
        payzone_thickness (float): Reservoir thickness
        min_radius (float): Minimum radius for solution
        wells_xyz (list): Well loctions (m)
        wells_t (list): Well start times (s)
        wells_q (list): Well flow rates (m3/s)

    """

    def __init__(self, **kwargs):
        """
        Initialization function

        """
        self.viscosity = 1.0
        self.permeability = 1.0
        self.storativity = 1.0e-3
        self.payzone_thickness = 1.0
        self.min_radius = 1.0
        self.t_origin = 0.0

        self.wells_xyz = []
        self.wells_t = []
        self.wells_q = []

    def pressure_well(self, x, y, t, well_id, derivative=False):
        """
        Estimate pressure or dpdt for a single well component

        Note: x, y, z, and t arguments can either be scalars, or arrays of identical size

        Args:
            - x (float, array): x-locations
            - y (float, array): y-locations
            - z (float, array): z-locations
            - t (float, array): time values
            - well_id (int): target well component
            - derivative (bool): If set, return dpdt instead of pressure

        Returns:
            float: pressure or dpdt components at the target point(s)
        """
        dt = np.maximum(t + self.t_origin - self.wells_t[well_id], 1.0)
        r = np.sqrt(
            (x - self.wells_xyz[well_id, 0]) ** 2
            + (y - self.wells_xyz[well_id, 1]) ** 2
        )

        unit_scale = 1e-13  # cP/mD
        K = unit_scale * self.permeability * 1000.0 * 9.81 / self.viscosity
        T = K * self.payzone_thickness
        b = r * r * self.storativity / (4.0 * T)
        u = b / dt
        u = np.minimum(np.maximum(u, 1e-6), 100.0)
        if derivative:
            s = (self.wells_q[well_id] / (4.0 * np.pi * T)) * np.exp(-u) / dt
        else:
            s = (self.wells_q[well_id] / (4.0 * np.pi * T)) * exp1(u)
        dp = s * 1000.0 * 9.81

        return dp

    def __call__(self, x, y, z, t):
        """
        Estimate pressure

        Note: arguments can either be scalars, or arrays of identical size

        Args:
            - x (float, array): x-locations
            - y (float, array): y-locations
            - z (float, array): z-locations
            - t (float, array): time values
        """
        p = 1000.0 * 9.81 * z
        for ii in range(len(self.wells_t)):
            p += self.pressure_well(x, y, t, ii)
        return p

    def dpdt(self, x, y, z, t):
        """
        Estimate dpdt

        Note: arguments can either be scalars, or arrays of identical size

        Args:
            - x (float, array): x-locations
            - y (float, array): y-locations
            - z (float, array): z-locations
            - t (float, array): time values
        """
        p = 0.0 * z
        for ii in range(len(self.wells_t)):
            p += self.pressure_well(x, y, t, ii, derivative=True)
        return p

    def setup_model(self, wells):
        """
        Setup the pressure model well terms.

        Each well must have the following values defined:
            - x (float): x-position of the well (local or utm coordinates)
            - y (float): y-position of the well (local or utm coordinates)
            - z (float): z-position of the well (local or utm coordinates)

        Wells can either have constant of time-varying pumping schedules.
        For constant flow-rate wells, this function expects:
            - q (float): constant flow rate (injection = + values)
            - t (float): start time of pumping

        For time-varying wells, this function expects:
            - q (array): length-N array of flow rate values (injection = positive)
            - t (array): length-N array of times

        Args:
            wells (list): list of well data to use in the calculation (see above for required information)

        """
        # Count the number of required well terms
        N = 0
        for well in wells:
            if isinstance(well["t"], float):
                N += 1
            else:
                N += len(well["t"])

        self.wells_xyz = np.zeros((N, 3))
        self.wells_t = np.zeros(N)
        self.wells_q = np.zeros(N)

        # Setup the well terms
        ii = 0
        for well in wells:
            if isinstance(well["t"], float):
                self.wells_xyz[ii, 0] = well["x"]
                self.wells_xyz[ii, 1] = well["y"]
                self.wells_xyz[ii, 2] = well["z"]
                self.wells_t[ii] = well["t"]
                self.wells_q[ii] = well["q"]
                ii += 1
            else:
                M = len(well["t"])
                self.wells_xyz[ii : ii + M, 0] = well["x"]
                self.wells_xyz[ii : ii + M, 1] = well["y"]
                self.wells_xyz[ii : ii + M, 2] = well["z"]
                self.wells_t[ii] = well["t"][0]
                self.wells_q[ii] = well["q"][0]
                ii += 1
                for jj in range(1, M):
                    self.wells_t[ii] = well["t"][jj]
                    self.wells_q[ii] = well["q"][jj] - well["q"][jj - 1]
                    ii += 1
