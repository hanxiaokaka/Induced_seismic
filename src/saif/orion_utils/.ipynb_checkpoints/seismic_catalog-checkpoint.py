import utm
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime
import logging
import numpy as np

# from orion_light.file_io import hdf5_wrapper


class SeismicCatalog:
    """
    Structure for holding seismic catalog information

    Attributes:
        N (int): Length of the catalog
        epoch (ndarray): Event timestamps (seconds)
        latitude (ndarray): Event latitudes (degrees)
        longitude (ndarray): Event longitudes (degrees)
        depth (ndarray): Event depths (m)
        utm_zone (int): UTM Zone for projection
        easting (ndarray): Eastings in UTM projection or local coordinates (m)
        northing (ndarray): Northings in UTM projection or local coordinates (m)
        magnitude (ndarray): Event magnitude magnitudes
        magnitude_bins (ndarray): magnitude magnitude bin edges
        magnitude_exceedance (ndarray): magnitude magnitude exceedance per bin
        a_value (float): Gutenberg-Richter a-value
        b_value (float): Gutenberg-Richter b-value
        varying_b_time (ndarray): Times for estimating sub-catalog b-values
        varying_b_value (ndarray): Gutenberg-Richter b-values over time
        magnitude_completeness (float): Magnitude of completeness for catalog
        background_seismicity_rate (float): Background seismicity rate

    """

    def __init__(self):
        """
        Seismic catalog initialization function

        """
        # Logging
        self.logger = logging.getLogger("seismic_catalog")
        logging.basicConfig(
            level=logging.WARNING,
            format="(%(asctime)s %(module)s:%(lineno)d) %(message)s",
        )

        # location
        self.N = 0
        self.latitude = np.zeros(0)
        self.longitude = np.zeros(0)
        self.depth = np.zeros(0)
        self.utm_zone = ""
        self.easting = np.zeros(0)
        self.northing = np.zeros(0)

        # Timing
        self.epoch = np.zeros(0)
        self.time_range = [-1e100, 1e100]
        self.data_slice = []
        self.time_range = [-1e99, 1e99]

        # Size, distribution
        self.magnitude = np.zeros(0)
        self.magnitude_bins = []
        self.magnitude_exceedance = []
        self.a_value = 0.0
        self.b_value = 0.0
        self.varying_b_time = []
        self.varying_b_value = []
        self.magnitude_completeness = -3.0
        self.background_seismicity_rate = 0.0
        self.magnitude_range = [-1e99, 1e99]

        # Time varying magnitude rate
        self.magnitude_rate_resolution = 100
        self.magnitude_rate = []
        self.magnitude_rate_time = []

        self.other_data = {}

    def set_log_level(self, log_level):
        if log_level == "debug":
            self.logger.setLevel(logging.DEBUG)
        elif log_level == "info":
            self.logger.setLevel(logging.INFO)
        elif log_level == "warning":
            self.logger.setLevel(logging.WARNING)
        elif log_level == "error":
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.setLevel(logging.CRITICAL)

    def load_catalog_dict(self, data):
        """
        Load the seismic catalog from an dictionary.

        Required entries in the catalog include: epoch, magnitude, depth
        Location entries can include one of the following:
            - latitude, longitude
            - easting, northing (local coordinates)
            - easting, northing, utm_zone

        Args:
            data (dict): catalog dictionary
        """
        # Sort values by epoch
        Ia = np.argsort(data["epoch"])
        print("\n\n before Ia \n\n")
        print("Ia ", Ia)

        self.epoch = data["epoch"][Ia]
        self.magnitude = data["magnitude"][Ia]
        self.depth = data["depth"][Ia]
        self.N = len(self.epoch)

        # Load location information
        self.longitude = np.zeros(0)
        self.latitude = np.zeros(0)
        self.easting = np.zeros(0)
        self.northing = np.zeros(0)

        if "longitude" in data.keys():
            self.longitude = data["longitude"][Ia]
            self.latitude = data["latitude"][Ia]

        if "easting" in data.keys():
            self.easting = data["easting"][Ia]
            self.northing = data["northing"][Ia]

        if "utm_zone" in data.keys():
            self.utm_zone = data["utm_zone"]
        else:
            self.utm_zone = ""

        self.other_data = {}
        targets = [
            "magnitude",
            "depth",
            "longitude",
            "latitude",
            "easting",
            "northing",
            "utm_zone",
        ]
        for k in data.keys():
            if k not in targets:
                if len(data[k]) == self.N:
                    self.other_data[k] = data[k][Ia]
                else:
                    # If lengths differ, do not sort
                    self.other_data[k] = data[k]

        self.convert_coordinates()
        self.reset_slice()

    def load_catalog_array(self, **xargs):
        """
        Initialize catalog from pre-loaded arrays.
        Required arguments include: epoch, magnitude, depth
        Location entries can include one of the following:
            - latitude, longitude
            - easting, northing (local coordinates)
            - easting, northing, utm_zone
        Additional arguments will be placed in the other_data dict

        Args:
            epoch (np.ndarray): 1D array of event time in epoch
            depth (np.ndarray): 1D array of event depths
            magnitude (np.ndarray): 1D array of event magnitudes
            longitude (np.ndarray): 1D array of event longitudes
            latitude (np.ndarray): 1D array of event latitudes
            easting (np.ndarray): 1D array of event eastings
            northing (np.ndarray): 1D array of event northings
            utm_zone (str): UTM zone string (e.g.: '4SU')
        """
        self.load_catalog_dict(xargs)

    """
    def load_catalog_hdf5(self, filename):
        ""
        Load the seismic catalog from an hdf5 format file.
        See load_catalog_dict for required entries

        Args:
            filename (str): catalog file name
        ""
        self.logger.info('Loading catalog from hdf5 format file: %s' % (filename))
        with hdf5_wrapper(filename) as data:
            self.load_catalog_dict(data)
    """

    def load_catalog_csv(self, filename):
        """
        Reads .csv format seismic catalog files
        The file should have an optional first line with the zone information "utm_zone, zone_id"
        and a line with variable names separated by commas
        See load_catalog_dict for required entries

        Args:
            filename (str): catalog file name

        """
        self.logger.info("Loading catalog from csv format file: %s" % (filename))
        value_names = []
        utm_zone = ""
        header_size = 1
        with open(filename) as f:
            line = f.readline()[:-1]
            if "utm_zone" in line:
                header_size += 1
                utm_zone = line.split(",")[1]
                line = f.readline()[:-1]
            value_names = line.split(",")

        tmp = np.loadtxt(filename, delimiter=",", skiprows=header_size, unpack=True)
        data = {"utm_zone": utm_zone}
        for ii, k in enumerate(value_names):
            data[k] = tmp[ii]
        self.load_catalog_dict(data)

    def get_catalog_as_dict(self):
        """
        Save key catalog entries to a dict

        Returns:
            dict: A dictionary of catalog data
        """
        data = self.other_data.copy()
        data["epoch"] = self.get_epoch_slice()
        data["magnitude"] = self.get_magnitude_slice()
        data["depth"] = self.get_depth_slice()

        longitude = self.get_longitude_slice()
        latitude = self.get_latitude_slice()
        if len(longitude):
            data["longitude"] = longitude
            data["latitude"] = latitude

        easting = self.get_easting_slice()
        northing = self.get_northing_slice()
        if len(easting):
            data["easting"] = easting
            data["northing"] = northing

        if self.utm_zone:
            data["utm_zone"] = self.utm_zone

        return data

    """
    def save_catalog_hdf5(self, filename):
        ""
        Save the seismic catalog to an hdf5 format file

        Args:
            filename (str): catalog file name
        ""
        self.logger.info('Saving catalog to hdf5 format file: %s' % (filename))
        catalog = self.get_catalog_as_dict()
        with hdf5_wrapper(filename, mode='w') as data:
            for k, value in catalog.items():
                data[k] = value
    """

    def save_catalog_csv(self, filename):
        """
        Save the seismic catalog as a .csv format file

        Args:
            filename (str): catalog file name

        """
        self.logger.info("Saving catalog to csv format file: %s" % (filename))
        catalog = self.get_catalog_as_dict()

        # Build the header
        header = ""
        if "utm_zone" in catalog.keys():
            header += "utm_zone,%s\n" % (catalog["utm_zone"])
            del catalog["utm_zone"]
        header_keys = sorted(catalog.keys())
        header += ",".join(header_keys)

        # Split any tensor data
        initial_catalog_keys = list(catalog.keys())
        for k in initial_catalog_keys:
            if isinstance(catalog[k], np.ndarray):
                M = np.shape(catalog[k])
                if len(M) > 1:
                    tmp = np.reshape(catalog[k], (M[0], -1))
                    for ii in range(np.shape(tmp)[1]):
                        catalog["%s_%i" % (k, ii)] = np.squeeze(tmp[:, ii])
                    del catalog[k]

        # Assemble the data, padding where necessary to keep a consistent length
        N = max([len(catalog[k]) for k in catalog.keys()])
        for k in catalog.keys():
            M = len(catalog[k])
            if M < N:
                catalog[k] = np.resize(catalog[k], N)

        # Save the data
        data = np.concatenate(
            [np.expand_dims(catalog[k], -1) for k in header_keys], axis=1
        )
        np.savetxt(filename, data, delimiter=",", comments="", header=header)

    def load_catalog_comcat(self, epoch_range, latitude_range, longitude_range):
        # Note: pycsep is an optional prerequisite, so this import is delayed
        import csep

        self.logger.info("Loading catalog from comcat")
        ta = datetime.date.fromtimestamp(epoch_range[0])
        tb = datetime.date.fromtimestamp(epoch_range[1])
        catalog = csep.query_comcat(
            ta,
            tb,
            min_magnitude=0,
            min_latitude=latitude_range[0],
            max_latitude=latitude_range[1],
            min_longitude=longitude_range[0],
            max_longitude=longitude_range[1],
            verbose=True,
        )

        # Note: pycsep seems to return milliseconds for epoch and km for depth
        self.epoch = catalog.get_epoch_times() * 1e-3
        self.magnitude = catalog.get_magnitudes()
        self.longitude = catalog.get_longitudes()
        self.latitude = catalog.get_latitudes()
        self.depth = catalog.get_depths() * 1e3
        self.N = len(self.latitude)
        self.calculate_utm_coordinates()
        self.logger.info("Done!")

        # Calculate attributes
        self.reset_slice()
        self.comcat_request_complete = True

    def convert_coordinates(self):
        """
        Converts utm coordinates to lat/lon or vice-versa if required
        """
        if len(self.longitude):
            if len(self.easting) == 0:
                self.calculate_utm_coordinates()

        else:
            if self.utm_zone:
                self.calculate_latlon_coordinates()

    def calculate_utm_coordinates(self):
        """
        Convert catalog lat/lon coordinates to UTM
        """
        if self.N:
            tmp = utm.from_latlon(self.latitude, self.longitude)
            self.easting = tmp[0]
            self.northing = tmp[1]
            self.utm_zone = str(tmp[2]) + tmp[3]
        else:
            self.easting = np.zeros(0)
            self.northing = np.zeros(0)
            self.utm_zone = "1N"

    def calculate_latlon_coordinates(self):
        """
        Convert catalog UTM coordinates to lat/lon
        """
        if self.N:
            zone_id = int(self.utm_zone[:-1])
            zone_letter = self.utm_zone[-1]
            try:
                self.latitude, self.longitude = utm.to_latlon(
                    self.easting, self.northing, zone_id, zone_letter
                )
            except utm.error.OutOfRangeError:
                self.logger.error("Unable to convert utm to lat/lon coordinates")
        else:
            self.latitude = np.zeros(0)
            self.longitude = np.zeros(0)

    def calculate_magnitude_rate(self, time_bins):
        """
        Estimate magnitude rate as a function of time

        Args:
            time_bins (list): bin values in time

        Returns:
            tuple: The bin centers, estimated magnitude rate in each bin
        """
        self.logger.debug("Calculating catalog moment rate")

        # Get data slices
        t_slice = self.get_epoch_slice()
        magnitude_slice = self.get_magnitude_slice()

        bin_centers = np.zeros(0)
        bin_magnitude_rate = np.zeros(0)

        if len(t_slice):
            # Find bins
            bin_ids = np.digitize(t_slice, time_bins) - 1
            dt = time_bins[1] - time_bins[0]
            bin_centers = time_bins[:-1] + 0.5 * dt
            bin_ids[bin_ids == len(bin_centers)] -= 1

            # Bin magnitude rate values
            bin_magnitude_rate = np.zeros(len(bin_centers))
            magnitude_rate_slice = (10.0 ** (1.5 * (magnitude_slice + 6))) / dt
            for catalog_index, time_index in enumerate(bin_ids):
                bin_magnitude_rate[time_index] += magnitude_rate_slice[catalog_index]

        return bin_centers, bin_magnitude_rate

    def calculate_cumulative_event_count(self, time_bins):
        """
        Count the number of events over time

        Args:
            time_bins (list): bin values in time

        Returns:
            tuple: The bin centers, event count in each bin
        """
        self.logger.debug("Calculating catalog cumulative event count")

        # Get data slices
        t_slice = self.get_epoch_slice()

        bin_centers = np.zeros(0)
        bin_event_count = np.zeros(0)
        if len(t_slice):
            # bin_centers = 0.5 * (time_bins[:-1] + time_bins[1:])
            # bin_event_count = np.histogram(t_slice, time_bins)[0]
            tmp = np.cumsum(np.histogram(t_slice, time_bins)[0])
            bin_centers = time_bins
            bin_event_count = np.concatenate([tmp[:1], tmp], axis=0)

        return bin_centers, bin_event_count

    def calculate_seismic_characteristics(
        self, magnitude_bin_res=0.1, time_bin_segments=10, time_bin_overlap=0.0
    ):
        """
        Generate various seismic characteristics

        Args:
            magnitude_bin_res (float): bin spacing for calculating a, b values
            time_bin_segments (int): number of segments to calculate b values over time
            time_bin_overlap (float): overlap amount for time bins in percent

        """
        self.logger.debug("Calculating catalog seismic characteristics")

        # Get the slice data
        magnitude_slice = self.get_magnitude_slice()
        t_slice = self.get_epoch_slice()
        M = len(t_slice)

        if M:
            # Calculate generic statistics
            magnitude_min = np.amin(magnitude_slice)
            magnitude_max = np.amax(magnitude_slice)
            dt = (np.amax(t_slice) - np.amin(t_slice)) / (60 * 60 * 24 * 365.25)

            # Build the magnitude bins
            bin_min = np.floor(magnitude_min / magnitude_bin_res) * magnitude_bin_res
            bin_max = np.ceil(magnitude_max / magnitude_bin_res) * magnitude_bin_res
            N_bins = int((bin_max - bin_min) / magnitude_bin_res) + 1
            magnitude_bins = np.linspace(bin_min, bin_max, N_bins)

            # Determine the global a, b values
            tmp = gutenberg_richter_a_b(magnitude_slice, magnitude_bins, dt)
            self.a_value = tmp[0]
            self.b_value = tmp[1]
            self.magnitude_completeness = tmp[2]
            self.magnitude_bins = tmp[3]
            self.magnitude_exceedance = tmp[4]

            # Calculate the b-value as a function of time
            # Note: the 'time bins' are psuedo-time and are based on event number
            bins_left, bins_right = get_overlapping_bin_ids(
                M, time_bin_segments, time_bin_overlap
            )
            self.varying_b_value = np.zeros(time_bin_segments)
            self.varying_b_time = 0.5 * (t_slice[bins_left] + t_slice[bins_right])
            for ii in range(time_bin_segments):
                dt = t_slice[bins_right[ii]] - t_slice[bins_left[ii]]
                tmp = gutenberg_richter_a_b(
                    magnitude_slice[bins_left[ii] : bins_right[ii]], magnitude_bins, dt
                )
                self.varying_b_value[ii] = tmp[1]

            # Estimate magnitude rate as a function of time
            time_bins = np.linspace(
                self.time_range[0],
                self.time_range[-1],
                self.magnitude_rate_resolution + 1,
            )
            (
                self.magnitude_rate_time,
                self.magnitude_rate,
            ) = self.calculate_magnitude_rate(time_bins)
        else:
            self.a_value = 0.0
            self.b_value = 0.0
            self.magnitude_completeness = -10.0
            self.magnitude_bins = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            self.magnitude_exceedance = np.zeros(7)
            self.varying_b_value = np.zeros(0)
            self.varying_b_time = np.zeros(0)
            self.magnitude_rate_time = np.zeros(0)
            self.magnitude_rate = np.zeros(0)

        self.logger.debug(
            "Estimated magnitude of completeness = %1.2f"
            % (self.magnitude_completeness)
        )

    def reset_slice(self):
        """
        Sets the catalog time slice to fit the entire catalog
        """
        self.set_slice()

    def set_slice(
        self,
        time_range=[-1e99, 1e99],
        magnitude_range=[-1e99, 1e99],
        minimum_interevent_time=-1.0,
        **kwargs
    ):
        """
        Set the catalog time slice

        Args:
            time_range (list): list of sub-catalog min/max times
            magnitude_range (list): list of sub-catalog min/max event magnitudes
            minimum_interevent_time (float): only include events if this amount of time has elapsed since the last
            kwargs (dict): Arguments to pass to calculate_seismic_characteristics
        """
        self.logger.debug("Setting seismic catalog slice")

        # time_range[0] = max(time_range[0], self.magnitude_completeness)
        self.time_range = time_range[:]
        self.magnitude_range = magnitude_range[:]

        t = self.epoch
        valid_points = np.ones(self.N, dtype=int)
        if time_range[0] > -1e98:
            self.logger.debug("t_min=%1.1f s" % (time_range[0]))
            valid_points[t < time_range[0]] = 0

        if time_range[1] < 1e98:
            self.logger.debug("t_max=%1.1f s" % (time_range[1]))
            valid_points[t > time_range[1]] = 0

        if magnitude_range[0] > -1e98:
            self.logger.debug("m_min=%1.1f s" % (magnitude_range[0]))
            valid_points[self.magnitude < magnitude_range[0]] = 0

        if magnitude_range[1] < 1e98:
            self.logger.debug("m_max=%1.1f s" % (magnitude_range[1]))
            valid_points[self.magnitude > magnitude_range[1]] = 0

        if minimum_interevent_time > 0:
            last_t = -1e99
            for ii in range(self.N):
                if valid_points[ii]:
                    if (t[ii] - last_t) < minimum_interevent_time:
                        valid_points[ii] = 0
                    else:
                        last_t = t[ii]

        self.data_slice = np.where(valid_points)[0]
        self.calculate_seismic_characteristics(**kwargs)

    def get_copy(
        self,
        time_range=[-1e99, 1e99],
        magnitude_range=[-1e99, 1e99],
        minimum_interevent_time=-1.0,
        seismic_characteristics_dt=-1.0,
    ):
        """
        Get a copy of the the catalog that matches the slice configuration

        Args:
            time_range (list): list of sub-catalog min/max times
            magnitude_range (list): list of sub-catalog min/max event magnitudes
            minimum_interevent_time (float): only include events if this amount of time has elapsed since the last
            seismic_characteristics_dt (float): timestep for seismic characteristic calculation
        """
        old_time_range = self.time_range[:]
        old_magnitude_range = self.magnitude_range[:]

        self.set_slice(
            time_range=time_range,
            magnitude_range=magnitude_range,
            minimum_interevent_time=minimum_interevent_time,
            seismic_characteristics_dt=seismic_characteristics_dt,
        )

        new_catalog = SeismicCatalog()
        new_catalog.latitude = self.get_latitude_slice().copy()
        new_catalog.longitude = self.get_longitude_slice().copy()
        new_catalog.depth = self.get_depth_slice().copy()
        new_catalog.easting = self.get_easting_slice().copy()
        new_catalog.northing = self.get_northing_slice().copy()
        new_catalog.epoch = self.get_epoch_slice().copy()
        new_catalog.magnitude = self.get_magnitude_slice().copy()
        new_catalog.N = len(new_catalog.epoch)
        new_catalog.utm_zone = self.utm_zone
        new_catalog.catalog_source = self.catalog_source + "_slice"
        new_catalog.old_catalog_source = new_catalog.catalog_source
        new_catalog.reset_slice()

        self.set_slice(time_range=old_time_range, magnitude_range=old_magnitude_range)

        return new_catalog

    def get_latitude_slice(self):
        """
        Get the catalog latitude slice
        """
        return self.latitude[self.data_slice]

    def get_longitude_slice(self):
        """
        Get the catalog longitude slice
        """
        return self.longitude[self.data_slice]

    def get_depth_slice(self):
        """
        Get the catalog depth slice
        """
        return self.depth[self.data_slice]

    def get_easting_slice(self):
        """
        Get the catalog easting slice
        """
        return self.easting[self.data_slice]

    def get_northing_slice(self):
        """
        Get the catalog northing slice
        """
        return self.northing[self.data_slice]

    def get_epoch_slice(self):
        """
        Get the catalog time slice
        """
        return self.epoch[self.data_slice]

    def get_magnitude_slice(self):
        """
        Get the catalog magnitude slice
        """
        return self.magnitude[self.data_slice]

    def get_magnitude_rate_data_slice(self):
        """
        Get the estimated catalog magnitude rate, time vector
        """
        return self.magnitude_rate_time, self.magnitude_rate

    def generate_plots(self, x_origin=0.0, y_origin=0.0, t_origin=0.0, point_scale=0.5):
        self.logger.debug("Generating seismic catalog plots")

        # Setup
        t_scale = 60 * 60 * 24.0
        x = self.get_easting_slice() - x_origin
        y = self.get_northing_slice() - y_origin
        t = (self.get_epoch_slice() - t_origin) / t_scale
        magnitude = self.get_magnitude_slice()
        if len(magnitude) == 0:
            self.logger.warning("No seismic data found for plotting")
            return

        # Map view
        magnitude_range = [np.amin(magnitude), np.amax(magnitude)]
        ms_point_size = point_scale * (3 ** (1 + magnitude - magnitude_range[0]))

        plt.figure()
        ax = plt.gca()
        ca = ax.scatter(
            x, y, s=ms_point_size, c=t, cmap=cm.jet, edgecolors="k", linewidths=0.1
        )
        cb = plt.colorbar(ca)
        cb.set_label("t (days)")
        ax.set_xlabel("UTM East (m)")
        ax.set_ylabel("UTM North (m)")
        ax.set_title("Map View")

        # Magnitude distribution
        tmp_N = 10 ** (self.a_value - self.b_value * self.magnitude_bins)
        tmp_w = self.magnitude_bins[1] - self.magnitude_bins[0]

        plt.figure()
        ax = plt.gca()
        ax.bar(
            self.magnitude_bins,
            self.magnitude_exceedance,
            tmp_w,
            facecolor="b",
            edgecolor="k",
        )
        ax.semilogy(
            self.magnitude_bins,
            tmp_N,
            "k--",
            label="a=%1.2f, b=%1.2f" % (self.a_value, self.b_value),
        )
        ax.legend(loc=1)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("N")
        ax.set_title("Magnitude Distribution")

        # Time series
        plt.figure()
        ax = plt.gca()
        ax.stem(
            t,
            magnitude,
            linefmt="b",
            markerfmt="None",
            use_line_collection=True,
        )
        ax.plot(t, magnitude, "bo")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("magnitude")
        ax.set_title("Time Series")

        # B value with time
        plt.figure()
        ax = plt.gca()
        ax.plot(self.varying_b_time / t_scale, self.varying_b_value, "b")
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("b-value")
        ax.set_title("b-value Variations")


def gutenberg_richter_a_b(magnitude, bins, dt, min_points=10):
    """
    Estimation Gutenbert Richter a, b values
    using the least-squares method

    Args:
        magnitude (ndarray): 1D array of magnitude magnitudes
        bins (narray): 1D array of magnitude bins for calculating a,b
        dt (float): time range of catalog

    Returns:
        tuple: a-value (float),
        b-value (float),
        magnitude_completeness (float),
        magnitude bin centers (ndarray),
        magnitude exceedance (ndarray)

    """
    tmp_a = np.NaN
    tmp_b = np.NaN
    magnitude_complete = np.NaN
    bin_centers = bins[:-1] + 0.5 * (bins[1] - bins[0])
    magnitude_exceedance = np.zeros(len(bin_centers))
    magnitude_exceedance[:] = np.NaN

    if len(magnitude) > min_points:
        hist = np.histogram(magnitude, bins)
        magnitude_exceedance = np.cumsum(hist[0][::-1])[::-1] / dt
        Ia = np.where(magnitude_exceedance > 0)
        tmp = np.polyfit(bin_centers[Ia], np.log10(magnitude_exceedance[Ia]), 1)

        # Estimate the magnitude of completeness
        magnitude_complete = (tmp[1] - np.log10(magnitude_exceedance[0])) / (-tmp[0])
        tmp_a = tmp[1]
        tmp_b = -tmp[0]

        Ia = np.where(bin_centers > magnitude_complete)[0]
        bin_centers = bin_centers[Ia]
        magnitude_exceedance = magnitude_exceedance[Ia]

    return tmp_a, tmp_b, magnitude_complete, bin_centers, magnitude_exceedance


def get_overlapping_bin_ids(target_length, segments, overlap):
    """
    Generate overlapping bins

    Args:
        target_length (int): Length of target to bin
        segments (int): number of segments
        time_bin_overlap (float): overlap amount for segments in percent

    Returns:
        tuple: left and right bin ids (np.ndarray)

    """
    tmp = np.linspace(0, target_length - 1, segments + 1, dtype=int)
    bin_offset = int(tmp[1] * overlap * 0.5)
    bins_left = tmp[:-1].copy()
    bins_right = tmp[1:].copy()
    bins_left[1:] -= bin_offset
    bins_left[-1] -= bin_offset
    bins_right[:-1] += bin_offset
    bins_right[0] += bin_offset
    return bins_left, bins_right
