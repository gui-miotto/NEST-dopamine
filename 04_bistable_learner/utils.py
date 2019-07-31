import numpy as np

def get_raster_data(events, gids, tmin=None, tmax=None):
    matches = np.isin(events['senders'], gids)
    senders = np.array(events['senders'])[matches]
    times = np.array(events['times'])[matches]
    if tmin is not None:
        matches = np.where(times >= tmin)
        senders = events['senders'][matches]
        times = events['times'][matches]
    if tmax is not None:
        matches = np.where(times <= tmax)
        senders = events['senders'][matches]
        times = events['times'][matches]
    return senders, times
    