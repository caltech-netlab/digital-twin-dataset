# Third-party imports
import os
import pathlib
from fractions import Fraction
import sys
from collections import OrderedDict
import numpy as np

# First-party imports
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
import utils
from global_param import ONE_SEC, FILE_PATHS, F


"""General phasors and waveform signal processing scripts"""


def find_first_zero_crossing(v, find_last=False):
    """
    Given an array of value, find first and/or last zero crossing with rising edge.
    :param v: numpy.ndarray, floating point values
    :param find_last: bool, whether also to return the last zero crossing.
    :return: int, or tuple of int, index into array v. None if no zero crossing is found.
        The returned index is the index BEFORE the zero crossing (inclusive, i.e. index of exactly zero).
    """
    # All zero crossings. Note that if a value is exactly zero, there will be two crossings due to the definition of np.sign.
    ind_crossings = np.where(np.diff(np.sign(v)))[0]
    if not len(ind_crossings):
        return (None, None) if find_last else None
    return (ind_crossings[0], ind_crossings[-1]) if find_last else ind_crossings[0]


def find_first_zero_crossing_weighted(t, v, find_last=False):
    """
    :return: np.datetime64, timestamp of first zero crossing,
        or None if no zero crossing is found.
    """
    ind = find_first_zero_crossing(v, find_last=find_last)
    if ind is None:
        return None
    if find_last:
        ind, ind_last = ind
        # Weighted average to find time
        delta_t = t[ind_last + 1] - t[ind_last]
        weight = abs(v[ind_last]) / (abs(v[ind_last + 1]) + abs(v[ind_last]))
        last = t[ind_last] + delta_t * weight

    # Weighted average to find time
    delta_t = t[ind + 1] - t[ind]
    weight = abs(v[ind]) / (abs(v[ind + 1]) + abs(v[ind]))
    first = t[ind] + delta_t * weight
    return (first, last) if find_last else first


def rms(v):
    """
    Note: this function introduces up to 0.2% error compared to eGauge onboard RMS.
    :param v: numpy.ndarray, must be in regular interval
    :returns: float, rms value of array
    """
    # Take the whole periods only
    lo, hi = find_first_zero_crossing(v, find_last=True)
    if (lo is None) or (lo == hi):
        return None
    return np.sqrt(np.mean(v[lo:hi] ** 2))


def rms_cumtrapz(t, v):
    t0, t1 = find_first_zero_crossing_weighted(t, v, find_last=True)
    lo, hi = find_first_zero_crossing(v, find_last=True)
    find_first_zero_crossing_weighted(t, v)
    v = np.concatenate([[0], v[lo:hi], [0]])
    t = np.concatenate([[t0], t[lo:hi], [t1]]).astype(float)
    delta_t = t[-1] - t[0]
    return np.sqrt(np.trapz(v**2, t) / delta_t)


def fft_frequency(
    df, 
    period=np.timedelta64(400, 'us').astype('timedelta64[us]'), 
    freq_res=1e0
):
    """
    Find dominant frequency using fft. Pad with zeros to increase frequency resolution.
    :param df: dict of numpy.ndarray, custom dataframe
    :param freq_res: float, unit: Hz, frequency resolution
    :return: float, unit: Hz, dominant frequency
    """
    # Run FFT to determine dominant frequency
    if not period:
        period = utils.determine_interval_size(df["t"])[0]
    df = utils.resample(df, interval_size=period, max_interval=8)[0]
    assert np.all(df["err"] == 0), f'Resample error detected: {(df["err"] != 0).sum()}, {df["t"]}'
    # frequency resolution = 1 / (sampling period * number of samples)
    # Frequency resolution is the smaller the better.
    if freq_res:
        n = int(ONE_SEC / (period * freq_res))
    else:
        n = utils.df_len(df)        
    X = np.fft.fft(df["v"], n=n)
    freq = np.fft.fftfreq(len(X), d=period / np.timedelta64(1, "s"))[:int(len(X) / 2)]
    basefreq = freq[np.argmax(np.abs(X[: int(len(X) / 2)]))]
    return basefreq


def fft_harmonics(
    df, 
    period=np.timedelta64(400, 'us').astype('timedelta64[us]'), 
    print_info=False, 
    n_th_harmonic=3, 
    freq_res=1e0
):
    """
    Run Fast Fourier Transform on waveform
    Definition of phase angle (theta):
        df['v'] = sin(2 * pi * basefreq * df['t'] + theta)
        Unit is degrees.
    A note on numpy.fft's output meaning:
        The fft result corresponds to frequencies resample_freq / (n / N)
        for n in range(1, N / 2) (n is the index into the array).
        So the frequency goes all the way up to resample_freq / 2,
        in increments of resample_freq / N, (i.e. basefreq / (2 ** k), where
        the 2 ** k is specified in duration.
    A note on frequency resolution:
        Frequency resolution is fixed in the np.fft implementation. To increase
        resolution, increasing the original waveform sample count is the only option
        (i.e. either sample at higher resolution or lengthen waveform capture duration).
    :returns: a tuple of
        - t0, numpy.datetime64, timestamp of the first sample
        - list of dict, each dict is describes n-th harmonic. The information
            is also in the rest of the outputs, but summarized here for convenience.
        - magnitudes, magnitude at each frequency, derived from complex magnitude
        - phase_angles, phase angle at each frequency, derived from complex magnitude
        - freq, np.ndarray, 1-dimensional, frequencies
        (magnitude, phase_angles, and freq are the same shape, and aligned to each other)
        - df, resampled and cropped waveform
    """
    # Resample to regular intervals
    if period:
        intervals = None
    else:
        period, intervals, err = utils.determine_interval_size(df["t"])
        assert not err, ("determine_interval_size failed", err, df["t"])
    df, _ = utils.resample(
        df, interval_size=period, intervals=intervals, max_interval=None
    )
    # Determine padding N, satisfying the following requirements
    # 1) n_th_harmonic (maximum frequency), 2) freq_res (resolution), 3) whole periods
    # Notation: f - frequency, d - period, n - number of samples
    # 1) f_res = 1 / (d * n)  --> n = 1 / (d * f_res)
    N_freq_res = utils.ceildiv(1, freq_res * period / ONE_SEC) if freq_res else 0
    # 2) original sample length
    N_orig_samples = utils.df_len(df)
    if print_info:
        print(
            f"Samples required from 1) frequency resolution: {N_freq_res}, "
            f"2) rounding original sample count: {N_orig_samples}"
        )
    # Take the maximum to satisfy all requirements
    if N_orig_samples > N_freq_res:
        N_target = round(N_orig_samples / N_freq_res) * N_freq_res
    else:
        N_target = N_freq_res
    N = round(N_target / 2) * 2
    if print_info:
        print(
            f"Padding input length {utils.df_len(df)} to {N}, with minimum satisfied {N_target}"
        )
    
    # Run FFT
    X = np.fft.fft(df["v"], n=N)
    N, half_N = len(X), int(len(X) / 2)
    freq = np.fft.fftfreq(N, d=period / ONE_SEC)[:half_N]

    magnitudes = np.abs(X[:half_N])
    phase_angles = np.angle(X[:half_N], deg=True) + 90.0

    # Find harmonics
    basefreq = freq[np.argmax(np.abs(X[: int(len(X) / 2)]))]
    harmonic_indices = utils.np_searchsorted(
        freq, [basefreq * (n + 1) for n in range(n_th_harmonic)], mode="nearest"
    )
    harmonics = [
        {
            "index": i,
            "frequency": freq[i],
            "complex_magnitude": X[i],
            "magnitude": magnitudes[i],
            "phase_angle": phase_angles[i],
        }
        for i in harmonic_indices
    ]
    for i in range(len(harmonics)):
        harmonics[i]["harmonic"] = i

    # RMS value calculated by Parseval's theorem
    # rms = np.sqrt(np.mean((np.abs(X)/ 1) ** 2) / len(df['v']))

    return (
        df["t"][0].astype("datetime64[us]"),
        harmonics,
        magnitudes,
        phase_angles,
        freq,
        df,
    )
    

def phase_offset(
    c1=None,
    c2=None,
    a1=None,
    a2=None,
    t1=None,
    t2=None,
    period=None,
    freq=None,
    deg=False,
):
    """
    Find phase offeset of two phasors. Denote f1 = sin(wt + theta), f2 as the two waveforms,
    the phase_offset is defined as:
        f2 = sin(wt + theta + phase_offset)

    :param c1: numpy.imag or numpy.ndarray, complex phasor magnitue and angle (reference)
    :param c2: numpy.imag or numpy.ndarray, complexphasor magnitue and angle
    :param a1: float or numpy.ndarray, phase angle in radian/degree (argument deg) (reference)
    :param a2: float or numpy.ndarray, phase angle in radian/degree (arguemnt deg)
    :param t1: numpy.datetime64 or numpy.ndarray, timestamp of waveform first sample (reference)
    :param t2: numpy.datetime64 or numpy.ndarray, timestamp of waveform first sample
    :param period: numpy.datetime64, the period of the signal
    :param freq: float or numpy.ndarray, the frequency of the signal
    :param deg: bool, whether input/output will be interpreted in degree of radian
    :return: pahse_offset, float, unit=deg/radian (see argument deg)
    """
    phase_offset_angle, phase_offset_time = 0, 0
    # Phase offset from angle offset
    if (c1 is not None) and (c2 is not None):
        phase_offset_angle = np.angle(c2) - np.angle(c1)
    elif (a1 is not None) and (a2 is not None):
        phase_offset_angle = ((a2 - a1) * np.pi / 180) if deg else (a2 - a1)
    # Phase offset from time offset (t1 - t2 gives the right polarity)
    if not ((t1 is None) and (t2 is None)):
        if freq is not None:
            if type(freq) is float:
                period = np.timedelta64(round(1 / freq * 1e9), "ns")
            else:
                period = (1 / freq * 1e9).astype(int).astype("timedelta64[ns]")
        else:
            assert period is not None, "argument period is None"
        phase_offset_time = 2 * np.pi * ((t1 - t2) % period) / period
    phase_offset = (phase_offset_angle + phase_offset_time) % (2 * np.pi)
    if type(phase_offset) is np.ndarray:
        phase_offset[phase_offset > np.pi] = (
            phase_offset[phase_offset > np.pi] - 2 * np.pi
        )
    else:
        phase_offset = (
            phase_offset - 2 * np.pi if phase_offset > np.pi else phase_offset
        )
    return phase_offset / np.pi * 180 if deg else phase_offset


def polar2rectangular(phase, magnitude, deg=True):
    """
    Convert phase & magnitude into a + bi complex number in rectangular coordinates
    :param phase: np.ndarray, float
    :param magnitude: np.ndarray, float
    :param deg: bool, whether phase is in degree or in radian
    :return: np.ndarray, complex float
    """
    if deg:
        phase = phase / 180 * np.pi
    return magnitude * np.exp(1j * phase)


def rectangular2polar(imag_arr, deg=True):
    return np.abs(imag_arr), (
        np.angle(imag_arr) / np.pi * 180 if deg else np.angle(imag_arr)
    )


def smooth_phase_180_oscillation(phase_arr, buffer=45):
    if buffer < np.mean(phase_arr) <= 180:
        phase_arr[phase_arr < -120] += 360
    elif -180 <= np.mean(phase_arr) < -buffer:
        phase_arr[phase_arr > 120] -= 360
    return phase_arr


def align_phasors(
    input_data,
    time_column_file=os.path.join(FILE_PATHS["phasors"], "t"),
    datetimespan=(None, None),
    columns=[
        "delta_t",
        "phase",
        "frequency",
        "rms",
    ],
    ref=None,
    delta_t_threshold=None,
    print_info=False,
    t_mode='nearest',
):
    """
    Compute relative phase angles for all locations, i.e. resample/align phasor to the same timestamp.
    This is similar to read_ts_dict but for phasor data.

    A key contribution of this function is graceful handling of missing samples among devices.
    This is accomplished with the help of time_column_file, which is saved by egauge_waveform scraper script.

    :param input_data: Either 
        - dict, with 
            key = str, name of timeseries; 
            value = str, path to timeseries data files.
        - or dict, with
            key = str, name of timeseries; 
            value = dict, df of data (already loaded into memory)
    :param time_column_file: either
        - str, path to the shared time column file.
        This is the time of synchronized api query, produced by egauge waveform scraping code.
        - or np.ndarray, dtype='datetime64[ms]', time column.
    :param datetimespan: see read_ts
    :param columns: either
        - list of str,  columns to return (e.g. 'rms', 'frequency'), anything
        in input_data plus reference time 't', time offset 'delta_t', fundamental 'phase'.
        - or 'all', return all columns in input_data
    :param ref: str or None.
        If supplied, str, name of data folder (i.e. device-channel, node) to be defined phase = 0.
            All other phase angles and delta_t are relative to this reference node.
        If None, returned phase angles are relative to the common t column.
    :param delta_t_threshold: float,  number of seconds. If ANY device's delta_t is above this threshold,
        then discard the entire timestamp of ALL phasors, i.e. if the start of two waveforms are too far apart,
        or more commonly, at least one device is unresponsive during a synchronized capture request,
        we discard all synchrophasors for that timestamp instead of borrowing the nearest phasor from
        an adjacent capture request. Set to None to deactivate.
    :return: a tuple of
        - t, numpy.ndarray, time column, reference time (phasors are defined relative to t)
        - df, dict of numpy.ndarray, key = same as keys in input_data, values = dataframe with
            - delta_t: float, first sample's timestamp minus refernce timestamp
            - phase: float, in range (-180, 180), phase of fundamental frequency in degrees
                The meaning of phase depends on input parameter 'ref'.
            - any other columns specified by input argument columns.
    """
    if columns == "all":
        columns_to_read = None
    else:
        columns_to_read = {"t"}
        for k in columns:
            if k == "phase":
                columns_to_read.add("phase_angle_harmonic_0")
                columns_to_read.add("frequency")
                columns_to_read.add("phase")
            elif k == "delta_t":
                pass
            else:
                columns_to_read.add(k)
        columns_to_read = list(columns_to_read)

    # Read timeseries. If data is already loaded, skip this step.
    if type(list(input_data.values())[0]) is str:
        df_dict = {}
        for name, f in input_data.items():
            df, err = utils.read_ts(f, datetimespan=datetimespan, usecols=columns_to_read, mode=t_mode)
            # Ignore/throw out missing data
            if (not err) and df and utils.df_len(df):
                df_dict[name] = df
    else:
        df_dict = input_data
        columns = 'all'
        
    # Read time column (if not loaded already)
    if type(time_column_file) is np.ndarray:
        time_column = time_column_file
    else:
        time_column, err = utils.read_ts(time_column_file, datetimespan=datetimespan, mode=t_mode)
        assert not err in (1, 2), f"{time_column}\nReading time column returned error {err}"
        time_column = time_column["t"]

    # Reference node
    if ref:
        ref_inds = utils.np_searchsorted(df_dict[ref]["t"], time_column, mode="nearest")
        ref_t = df_dict[ref]["t"][ref_inds]
        ref_df = {
            k: arr[ref_inds]
            for k, arr in df_dict[ref].items()
            if (columns == "all") or (k in columns)
        }
        if "phase" in columns:
            ref_df["phase"] = df_dict[ref][
                "phase" if "phase" in df_dict[ref] else "phase_angle_harmonic_0"
            ][ref_inds]

    # Align columns. Iterate through each df.
    out_df_dict = OrderedDict()  # device-channel --> column_header --> array
    for name in df_dict:
        if name.split(".")[0] == "t":
            print("[Warning] Did you pass in the time column file in input_data?")
        # Align time column (this handles time column correspondance issue)
        inds = utils.np_searchsorted(df_dict[name]["t"], time_column, mode="nearest")
        df = utils.index_select_df(df_dict[name], inds)

        # Iterate through all columns.
        if ('delta_t' in columns) or delta_t_threshold or (columns == "all"):
            delta_t = ((ref_t if ref else time_column) - df['t']) / np.timedelta64(1, 's')
            utils.insert_dict(out_df_dict, [name, 'delta_t'], delta_t)
        for k in (df.keys() if columns == "all" else columns):
            if k == 'delta_t':
                continue
            elif (k == "phase") or (
                (k == "phase_angle_harmonic_0") and (columns == "all")
            ):
                phase = phase_offset(
                    a1=ref_df["phase"] if ref else 0,
                    a2=df["phase" if "phase" in df else "phase_angle_harmonic_0"],
                    t1=ref_t if ref else time_column,
                    t2=df["t"],
                    freq=df["frequency"],
                    deg=True,
                )
                utils.insert_dict(out_df_dict, [name, 'phase'], phase)
            elif k[:11] == "phase_angle":
                phase = phase_offset(
                    a1=df[k],
                    a2=ref_df[k] if ref else 0,
                    t1=df["t"],
                    t2=ref_t if ref else time_column,
                    freq=df["frequency"],
                    deg=True,
                )
                utils.insert_dict(out_df_dict, [name, k], phase)
            elif df[k].dtype in (np.csingle, np.cdouble, complex):
                if ref:
                    phase_diff = phase_offset(
                        c1=df[k],
                        c2=ref_df[k],
                        t1=df['t'],
                        t2=ref_t,
                        freq=df['frequency'] if 'frequency' in df else F,
                        deg=False
                    )
                    angle_shifted = np.abs(df[k]) * np.exp(1j * phase_diff)
                    utils.insert_dict(out_df_dict, [name, k], angle_shifted)
                else:
                    phase_diff = phase_offset(
                        t1=df['t'],
                        t2=ref_t if ref else time_column,
                        freq=df['frequency'] if 'frequency' in df else F,
                        deg=False
                    )
                    angle_shifted = df[k] * np.exp(1j * phase_diff)
                    utils.insert_dict(out_df_dict, [name, k], angle_shifted)
            else:
                utils.insert_dict(out_df_dict, [name, k], df[k])

    if delta_t_threshold:
        info = {}
        T = len(time_column)
        valid_rows = np.ones(T, dtype=bool)
        # Concatenate columns of data into matrix
        for name, df in out_df_dict.items():
            valid = abs(df["delta_t"]) < delta_t_threshold
            valid_rows &= valid
            if print_info:
                info[name] = "%.3f" % valid.mean()

        V = valid_rows.sum()
        if print_info:
            info = {k: v for k, v in sorted(info.items(), key=lambda it: it[1])}
            print(info)
            print(f"Valid/total samples: {V}/{T}, valid rate: {V/T}")
        # Remove timestamps that does not satisfy delta_t_threshold
        time_column = time_column[valid_rows]
        out_df_dict = {
            name: {k: out_df_dict[name][k][valid_rows] for k in out_df_dict[name]}
            for name in out_df_dict
        }

    return time_column, out_df_dict


def produce_phasors_matrix(
    input_data,
    time_column_file=os.path.join(FILE_PATHS["phasors"], "t"),
    datetimespan=(None, None),
    ref=None,
    delta_t_threshold=0.5,
    return_align_phasors=False,
):
    """
    Uses align_phasors as the core algorithm. Outputs a matrix of phasors (complex floats a + bi),
    where each column is a meter-channel, and each row is a synchronized timestamp.
    
    :param input_data: see align_phasors. Supports both df with 'phase' and 'rms' keys, 
        and df with multiple columns of complex numbers, in which case the complex numbers 
        are interpreted as phasors and these columns are added to the returned array.
    :param time_column_file: str, path to the shared time column file.
        Thit is the time of synchronized api query, produced by egauge waveform scraping code.
    :param datetimespan: see read_ts
    :param columns: list of str, columns to return (e.g. 'rms', 'frequency'), anything
        in input_data plus reference time 't', time offset 'delta_t', fundamental 'phase'
    :param ref: str or None.
        If supplied, str, name of data folder (i.e. device-channel, node) to be defined phase = 0.
            All other phase angles and delta_t are relative to this reference node.
        If None, returned phase angles are relative to the common t column.
    :param delta_t_threshold: float, number of seconds, if delta_t (difference between the waveform
        capture request and the first timestamp of the waveform) is larger than this threshold, then
        the entire row (timestamp) is discarded, i.e. if the start of two waveforms are too far apart,
        or more commonly, at least one device is unresponsive during a synchronized capture request,
        we discard all synchrophasors for that timestamp instead of borrowing the nearest phasor from
        an adjacent capture request.
    :return: a tuple of
        - header, list of str, name of meter-channels
        - timestamps, numpy.ndarray, time of capture request initiation (exact waveforms start time is different)
        - data, np.ndarray, shape = (timestamps, meter channels), dtype = np.csingle
    """
    
    timestamps, df_dict = align_phasors(
        input_data=input_data,
        time_column_file=time_column_file,
        datetimespan=datetimespan,
        columns=["delta_t", "phase", "rms"],
        ref=ref,
        delta_t_threshold=delta_t_threshold,
    )
    
    header, arr = [], []
    # Concatenate columns of data into matrix
    for dev_ch, df in df_dict.items():
        if 'phase' and 'rms' in df:
            arr.append(polar2rectangular(df['phase'], df['rms']))
            header.append(dev_ch)
        else:
            for k, v in df.items():
                if v.dtype in (np.csingle, np.cdouble, complex):
                    arr.append(v)
                    header.append(k)
    arr = np.stack(arr, axis=1)
    
    if return_align_phasors:
        return header, timestamps, arr, df_dict
    else:
        return header, timestamps, arr
