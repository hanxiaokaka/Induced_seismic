"""Peak detector for seismic catalog time series (seismic rate)
This function is mainly used to monitor the occurrence of earthquake clusters.
We can define the step size of detection, that is, how long to detect a peak.
This way we can avoid the repetition of two earthquakes that occur very close in time.

At the same time, we can set a threshold, that is, when the earthquake rate exceeds a
certain value, it is counted as an earthquake swarm.

Besides, this function also avoids multiple peaks in the high plateau period,
that is, the high plateau period is only counted as one peak.
"""
import numpy as np


def pk_indxs(seis_sr, trshd=0.3, min_dist=1, trshd_abs=False):
    """Peak detection routine.

    Finds the numeric index of the peaks in the seismic catalog *seis_sr* by taking its first order difference. By using
    *trshd* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *seis_sr* must be signed.

    Parameters definition
    ----------
    seis_sr : ndarray (signed)
        1D seismic catalog data to search for peaks.
    trshd : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    trshd_abs: boolean
        If True, the trshd value will be interpreted as an absolute value, instead of
        a normalized threshold.
    Returns: ndarray
        Array containing the numeric indexes of the peaks that were detected.
    """
    if isinstance(seis_sr, np.ndarray) and np.issubdtype(seis_sr.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not trshd_abs:
        trshd = trshd * (np.max(seis_sr) - np.min(seis_sr)) + np.min(seis_sr)

    min_dist = int(min_dist)

    # compute first order difference
    ds_s = np.diff(seis_sr)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros, = np.where(ds_s == 0)

    # check if the signal is totally flat
    if len(zeros) == len(seis_sr) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        zeros_diff_not_one, = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            ds_s[zero_plateaus[0]] = ds_s[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(ds_s) - 1:
            ds_s[zero_plateaus[-1]] = ds_s[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            ds_s[plateau[plateau < median]] = ds_s[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            ds_s[plateau[plateau >= median]] = ds_s[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([ds_s, 0.0]) < 0.0)
        & (np.hstack([0.0, ds_s]) > 0.0)
        & (np.greater(seis_sr, trshd))
    )[0]
    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(seis_sr[peaks])][::-1]
        rem = np.ones(seis_sr.size, dtype=bool)
        rem[:]=False
        rem[peaks] = True

        for peak in highest:
            if  rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = False
                rem[peak] = True

        peaks = np.arange(seis_sr.size)[rem]

    return peaks