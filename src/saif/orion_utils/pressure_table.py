import numpy as np

# from orion_light import file_io
from . import orion_file_io
from scipy import integrate


class PressureTableModel:
    """
    Pressure model based off of Theis Solution

    Attributes:
        file_name (string): Table filename
        p_interp (scipy.interpolate.LinearNDInterpolator): pressure interpolator
        dpdt_interp (scipy.interpolate.LinearNDInterpolator): dpdt interpolator

    """

    def __init__(self, **kwargs):
        """
        Initialization function

        """
        # Table data
        self.p_interp = None
        self.dpdt_interp = None
        self.table_data = {}

    def __call__(self, x, y, z, t):
        return self.p_interp(x, y, z, t)

    def dpdt(self, x, y, z, t):
        return self.dpdt_interp(x, y, z, t)

    """
    def load_table(self, file_name):
        ""
        Load tables and generate interpolator objects.

        Table files should have t, and some combination of pressure and/or dpdt defined.
        The data can be formatted as a structure tables or be unstructured.

        Args:
            file_name (str): Location of the table files (csv or hdf5)

        ""
        if ('hdf5' in file_name):
            print('Loading pressure table from hdf5 file: %s' % (file_name))
            tmp = file_io.hdf5_wrapper(file_name)
            self.table_data = tmp.get_copy()
        elif ('csv' in file_name):
            print('Loading pressure table from csv file: %s' % (file_name))
            self.table_data = file_io.parse_csv(file_name)
        else:
            raise Exception('File format not recognized: %s' % (file_name))

        self.setup_model()
    """

    def load_array(self, **xargs):
        """
        Initialize pressure model from pre-loaded arrays.

        At a minimum, t and some combination of pressure/dpdt should be defined.
        The data can be formatted as a structure tables or be unstructured.

        Args:
            pressure (np.ndarray): Array of pressure values
            dpdt (np.ndarray): Array of dpdt values
            x (np.ndarray): Array of x location values
            y (np.ndarray): Array of y location values
            z (np.ndarray): Array of z location values
            t (np.ndarray): Array of time values
        """
        self.table_data = xargs.copy()
        self.setup_model()

    def setup_model(self):
        """
        If either of these values are missing, this function will estimate the remaining component
        """
        orion_file_io.check_table_shape(self.table_data)

        # Check to see whether we need to calculate pressure or dpdt
        if "t" not in self.table_data:
            raise Exception("The pressure table file is missing t")
        if "pressure" not in self.table_data:
            if "dpdt" in self.table_data:
                self.table_data["pressure"] = integrate.cumtrapz(
                    self.table_data["dpdt"], self.table_data["t"], initial=0.0, axis=-1
                )
            else:
                raise Exception(
                    "The pressure table file requires either pressure or dpdt"
                )
        if "dpdt" not in self.table_data:
            if "pressure" in self.table_data:
                scale_shape = np.ones(
                    len(np.shape(self.table_data["pressure"])), dtype=int
                )
                scale_shape[-1] = -1
                dt = np.reshape(np.diff(self.table_data["t"]), scale_shape)
                dpdt = np.diff(self.table_data["pressure"], axis=-1) / dt
                self.table_data["dpdt"] = np.concatenate([dpdt[..., :1], dpdt], axis=-1)
            else:
                raise Exception(
                    "The pressure table file requires either pressure or dpdt"
                )

        # Build interpolators
        interps = orion_file_io.convert_tables_to_interpolators(self.table_data)
        self.p_interp = interps["pressure"]
        self.dpdt_interp = interps["dpdt"]
