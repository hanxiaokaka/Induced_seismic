import h5py
import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d
from orion_light import function_wrappers


class hdf5_wrapper():
    """
    Class for reading/writing hdf5 files, which behaves similar to a native dict
    """

    def __init__(self, fname='', target='', mode='r'):
        """
        Initialize the hdf5_wrapper class

        Args:
            fname (str): the filename of a new or existing hdf5 database
            target (str): the handle of an existing hdf5 dataset
            mode (str): the read/write behavior of the database (default='r')
        """

        self.mode = mode
        self.target = target
        if fname:
            self.target = h5py.File(fname, self.mode)

    def __getitem__(self, k):
        """
        Get a target from the database
        Note: The returned value depends on the type of the target:
                - An existing hdf5 group will return an instance of hdf5_wrapper
                - An existing array will return an numpy ndarray
                - If the target is not present in the datastructure and the
                    database is open in read/write mode, the wrapper will create a
                    new group and return an hdf5_wrapper
                - Otherwise, this will throw an error

        Args:
            k (str): name of target group or array

        Returns
            hdf5_wrapper.hdf5_wrapper, array
        """
        if (k not in self.target):
            if (self.mode in ['w', 'a']):
                self.target.create_group(k)
            else:
                raise ValueError('Entry does not exist in database: %s' % (k))

        tmp = self.target[k]

        if isinstance(tmp, h5py._hl.group.Group):
            return hdf5_wrapper(target=tmp, mode=self.mode)
        elif isinstance(tmp, h5py._hl.dataset.Dataset):
            tmp = np.array(tmp)

            # Decode any string types
            if (tmp.dtype.kind in ['S', 'U', 'O']):
                tmp = np.core.defchararray.decode(tmp)

            # Convert any 0-length arrays to native types
            if not tmp.shape:
                tmp = tmp[()]

            return tmp
        else:
            return tmp

    def __setitem__(self, k, value):
        """
        Write an object to the database if write-mode is enabled

        Args:
            k (str): the name of the object
            value (float, np.ndarray): the object to be written
        """

        if (self.mode in ['w', 'a']):
            if isinstance(value, dict):
                # Recursively add groups and their children
                if (k not in self.target):
                    self.target.create_group(k)
                new_group = self[k]
                for x in value:
                    new_group[x] = value[x]
            else:
                # Delete the old copy if necessary
                if (k in self.target):
                    del(self.target[k])

                # Add everything else as an ndarray
                tmp = np.array(value)
                if (tmp.dtype.kind in ['S', 'U', 'O']):
                    tmp = np.core.defchararray.encode(tmp)
                self.target[k] = tmp
        else:
            raise ValueError('Cannot write to an hdf5 opened in read-only mode!  This can be changed by overriding the default mode argument for the wrapper.')

    def link(self, k, target):
        """
        Link an external hdf5 file to this location in the database

        Args:
            k (str): the name of the new link in the database
            target (str): the path to the external database
        """
        self.target[k] = h5py.ExternalLink(target, '/')

    def keys(self):
        """
        Get a list of groups and arrays located at the current level

        Returns
            list: a list of strings
        """
        if isinstance(self.target, h5py._hl.group.Group):
            return list(self.target)
        else:
            raise ValueError('Object not a group!')

    def items(self):
        """
        Return the key-value pairs for entries at the current level

        Returns
            tuple: keys, values
        """
        tmp = self.keys()
        values = [self[k] for k in tmp]
        return zip(tmp, values)

    def __enter__(self):
        """
        Entry point for an iterator
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        End point for an iterator
        """
        self.target.close()

    def __del__(self):
        """
        Closes the database on wrapper deletion
        """
        try:
            if isinstance(self.target, h5py._hl.files.File):
                self.target.close()
        except:
            pass

    def close(self):
        """
        Closes the database
        """
        if isinstance(self.target, h5py._hl.files.File):
            self.target.close()

    def get_copy(self):
        """
        Copy the entire database into memory

        Returns
            dict: A copy of the database
        """
        tmp = {}
        self.copy(tmp)
        return tmp

    def copy(self, output):
        """
        Pack the contents of the current database level onto the target dict

        Args:
            output (dict): the dictionary to pack objects into
        """
        for k in self.keys():
            tmp = self[k]

            if isinstance(tmp, hdf5_wrapper):
                output[k] = {}
                tmp.copy(output[k])
            else:
                output[k] = tmp


def check_table_shape(data, axes_names=['x', 'y', 'z', 't']):
    """
    Check shape of table arrays

    Attributes:
        data: Dictionary of table entries
        axes_names: List of potential axes names
    """
    pnames = [k for k in data.keys() if k not in axes_names]

    # Check to see if the data needs to be reshaped
    structured_shape = tuple([len(data[k]) for k in axes_names if k in data])
    for k in pnames:
        if (data[k].size != structured_shape[0]):
            data[k] = np.reshape(data[k], structured_shape, order='F')


def convert_tables_to_interpolators(data, axes_names=['x', 'y', 'z', 't']):
    """
    Convert structured or unstructured tables into interpolators

    Attributes:
        data: Dictionary of table entries
        axes_names: List of potential axes names
    """
    table_interpolators = {}
    pnames = [k for k in data.keys() if k not in axes_names]

    # Check to see if the data is structured/unstructured
    if (len(np.shape(data[pnames[0]])) > 1):
        # The data appears to be structrued
        points = []
        for k in axes_names:
            if k in data.keys():
                points.append(data[k])
        Ndim = len(points)

        for p in pnames:
            tmp = RegularGridInterpolator(tuple(points),
                                          np.ascontiguousarray(data[p]),
                                          bounds_error=False,
                                          fill_value=0.0)
            table_interpolators[p] = function_wrappers.variable_len_fn(tmp, Ndim, list_arg=True)

    else:
        # Unstructured data
        points = []
        for k in axes_names:
            if k in data.keys():
                points.append(np.reshape(data[k], (-1, 1)))
        Ndim = len(points)

        tmp = np.meshgrid(*points, indexing='ij')
        points = [np.reshape(x, (-1, 1), order='F') for x in tmp]
        points = np.ascontiguousarray(np.squeeze(np.concatenate(points, axis=1)))

        # Load the data
        for p in pnames:
            if (points.ndim == 1):
                pval = np.squeeze(data[p])
                tmp = interp1d(points, pval, kind='linear', bounds_error=False, fill_value=(pval[0], pval[-1]))
                table_interpolators[p] = function_wrappers.variable_len_fn(tmp, Ndim)
            else:
                pval = np.reshape(data[p], (-1, 1), order='F')
                tmp = LinearNDInterpolator(points, pval, fill_value=0.0)
                table_interpolators[p] = function_wrappers.variable_len_fn(tmp, Ndim)

    return table_interpolators


def parse_csv(fname):
    """
    Parse csv file with headers, units

    Args:
            fname (string): Filename
            header_size (int): number of header lines

    Returns:
            dict: File results

    """
    # Check headers
    headers = []
    header_size = 0
    units_scale = []
    with open(fname) as f:
        headers = [x.strip() for x in f.readline()[1:-1].split(',')]
        tmp = f.readline()
        if ('#' in tmp):
            header_size += 1
            unit_converter = unit_conversion.UnitManager()
            units = [x.strip() for x in tmp[1:].split(',')]
            units_scale = [unit_converter(x) for x in units]
        else:
            units_scale = np.ones(len(headers))

    # Parse body
    tmp = np.loadtxt(fname, unpack=True, delimiter=',', skiprows=header_size)
    data = {headers[ii]: tmp[ii] * units_scale[ii] for ii in range(len(headers))}

    return data
