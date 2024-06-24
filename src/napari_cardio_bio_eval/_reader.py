"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import os
from nanobio_core. epic_cardio.processing import load_data, preprocessing, load_params


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".npy"):
        return None

    """
    We need a directory and in that directory we need a file named 'test_avg' and a file named 'test_WL_Power'.

    files = [ obj for obj in os.listdir(dir_path) if '_wl_power' in obj.lower()]
    if len(files) == 0:
        print("Missing test wl power file!!!")
        return None
        
    files = [ obj for obj in os.listdir(dir_path) if '_avg' in obj.lower()]
    if len(files) == 0:
        print("Missing test avg!!!")
        return None
    
    """

    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    preprocessing_params = {
        'flip': [False, False], # Vertical and Horizontal flipping
        'signal_range' : {
            'range_type': RangeType.MEASUREMENT_PHASE,
            'ranges': [0, None],
        },
        'drift_correction': {
            'threshold': 75,
            'filter_method': 'mean',
            'background_selector': True,
        }
    } 
    Path = path
    RESULT_PATH = os.path.join(path, 'result')
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    raw_wells, full_time, full_phases = load_data(path, flip=preprocessing_params['flip'])
    filter_params, _, _ = load_params(RESULT_PATH)
    well_data, time, phases, filter_ptss, selected_range = preprocessing(preprocessing_params, raw_wells, full_time, full_phases, filter_params)

    return [(well_data, 'image')]
