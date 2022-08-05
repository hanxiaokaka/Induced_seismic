import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence

def split_chunks(arr, sections, padding_value=-1):
    _splits = np.array_split(arr, sections)
    peaks, inter_peaks = _splits[::2], _splits[1::2]
    
    peaks = torch.FloatTensor(peaks)[None, :, 0]
    inter_peaks = list(map(torch.FloatTensor, inter_peaks))
    inter_peaks = pad_sequence(inter_peaks, padding_value=padding_value).t()[None, :, :]
    
    return peaks, inter_peaks


def extract_sections(data_df, pks):
    sections = [1] + sorted((pks + 1).tolist() + pks.tolist())

    signal_sections = {}
    signal_starts = {}

    #TODO: check that everything is the right scale

    #padding _delta_t with zeros will make EVERYTHING constant
    #no numerical instability! The system simply does not evolve.

    signals_to_extract = ['delta_t', 'pressure', 'dpdt', 'number', 'rate', 'days']
    pads = [0,0,0,-1,0,0]

    for s, p in zip(signals_to_extract, pads):
        _start, _signal = split_chunks(data_df[s].values, sections, padding_value=p)
        signal_starts[s] = _start
        signal_sections[s] = _signal

    return signal_starts, signal_sections