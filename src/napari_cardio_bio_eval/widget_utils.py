import os
import napari
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch

from nanobio_core.epic_cardio.processing import load_data, load_params, preprocessing, localization
from nanobio_core.epic_cardio.defs import WELL_NAMES
from export_and_plot.export import export_results

from napari.qt.threading import thread_worker
from matplotlib.backends.backend_qt5agg import FigureCanvas


# thread worker functions - these functions are called from the GUI and run in a separate thread to not block the GUI
@thread_worker
def load_data_thread(widget):
    """ 
    Load the data from the selected directory and the parameters from the previous run if they exist.
    It needs the widget to access the GUI elements and save the data and parameters to the widget.
    This is a part of the data loading process it is called from the loading button and is run in a separate thread to not block the GUI.
    The data and parameter loading is done in the nanobio functions.
    The next step is data preprocessing which called by the user from a different button.
    """
    path = widget.dirLineEdit.text()
    widget.RESULT_PATH = os.path.join(path, 'result')
    if not os.path.exists(widget.RESULT_PATH):
        os.mkdir(widget.RESULT_PATH)

    widget.raw_wells, widget.full_time, widget.full_phases = load_data(path, flip=widget.preprocessing_params['flip'])
    widget.filter_params, widget.preprocessing_params_loaded, widget.localization_params_loaded = load_params(widget.RESULT_PATH)

    # we load in the parameters from the previous run if they exist
    # and set the values in the GUI so it is clear what was used and can be changed
    loaded_params_to_GUI(widget)
    # Set the range selection to the loaded values if they exist, else full range
    widget.rangeLabel.setText(f'Phases: {[(n+1, p) for n, p in enumerate(widget.full_phases)]}, Time: {len(widget.full_time)}')
    widget.rangeTypeSelect.currentIndexChanged.connect(widget.range_type_changed)
    # Enable the range selection, before this we dont know the range parameters for this data
    widget.rangeTypeSelect.setEnabled(True)
    widget.rangeMin.setEnabled(True)
    widget.rangeMax.setEnabled(True)
    # Set the extremes of the range
    widget.range_type_changed()
    # Enable the process so the user can start the preprocessing, before this we cant start the preprocessing so its disabled
    widget.processButton.setEnabled(True)

@thread_worker
def preprocess_data_thread(widget):
    """
    Simply update the preprocessing parameters from the GUI and preprocess the data with the nanobio function. 
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    widget.preprocessing_params = update_preprocessing_params(widget)
    widget.well_data, widget.time, widget.phases, widget.filter_ptss, widget.selected_range = preprocessing(widget.preprocessing_params, widget.raw_wells, widget.full_time, widget.full_phases, widget.filter_params)

@thread_worker
def peak_detection_thread(widget):
    """
    Next step of the pipeline after the preprocessing. Also updates the parameters from the GUI and calls the nanobio function.
    There can be a background selection before the peak detection, and if it is done we need to get the new filter/bacground points 
    and give it to the peak detection function so it will process the wells again with the new (hopefully better) background points.
    These bacground points are calculated automatically in the first prepocessing and will be saved at the export.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    widget.backgroundSelectorButton.setEnabled(False)
    widget.peakButton.setEnabled(False)
    widget.exportButton.setEnabled(False)
    widget.preprocessing_params = update_preprocessing_params(widget)
    widget.localization_params = update_localization_params(widget)

    if widget.background_selector:
        widget.filter_ptss = get_filter_points(widget.viewer, widget._bg_points)
    widget.background_selector = False

    # From here the well data contains the wells, the selected points and the filter points (which are the background points)
    widget.well_data = localization(widget.preprocessing_params, widget.localization_params,
                                    widget.raw_wells, widget.selected_range, widget.filter_ptss)
    # Get the remaining wells for the GUI function
    # If the user deletes a well layer from the napari viewer after the preprocessing it will not show up again
    # (Exception is the bacground selection because all of the wells are shown there)
    widget.remaining_wells = get_remaining_wells_from_layers(widget.viewer)

@thread_worker
def segmentation_thread(widget):
    """
    Similar to the peak detection but it is not using a nanobio function but a segmentation model to highlight the cells in the wells.
    The model is loaded from the path given in the GUI and the well data is processed with the model.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    # loadin the script torch model
    widget.model = torch.jit.load(widget.modelPath.text())
    biosensor = []
    # The model input channel size is the biosensor length
    # for now this is 8 but it may vary later or needs to be dinamically set from the model
    bio_len = 8
    for name in WELL_NAMES: # remaining_wells?
        # the lin_indices get a linearly spaced indices from the well data length to the biosensor length
        biosensor.append(widget.well_data[name][lin_indices(widget.well_data[name].shape[0], bio_len)])

    biosensor_tensor = torch.tensor(np.array(biosensor)).float() 

    with torch.no_grad():
        output = widget.model(biosensor_tensor)

    widget.image_size = output.shape[2]
    widget.scaling_factor = widget.image_size // 80
    # aplying the sigmoid function to the output to get the smaller cells too
    output = torch.sigmoid(output).squeeze().detach().numpy()
    widget.bin_output = (output > 0.5).astype(int)

    widget.cell_centers = []
    widget.labels = []
    for i in range(len(WELL_NAMES)):
        pred = widget.bin_output[i].squeeze().astype(np.uint8)
        _, labels, _, centers = cv2.connectedComponentsWithStats(pred, connectivity=8)
        widget.cell_centers.append(centers[1:])
        widget.labels.append(labels)

@thread_worker
def export_results_thread(widget):
    """
    Export the cells to the result subdirectory. The export function is a part of the nanobio package 
    and the kind of export is set by the GUI with the export parameters.
    Because of this the time of the export can be long and it is run in a separate thread to not block the GUI.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    export_results(widget.export_params, widget.RESULT_PATH, widget.selected_ptss, widget.filter_ptss, widget.well_data, 
                   widget.time, widget.phases, widget.raw_wells, widget.full_time, widget.full_phases, widget.selected_range)


# GUI functions
def manual_background_selection(widget):
    """
    This is an optional step. The user can select the background points manually if the automatically selected are not good enough.
    Basically the whole function is a GUI function to show the background points so its not worth to run in a separate thread.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    widget.background_selector = True
    # Remove the docked plot if it exists
    if widget.docked_plot is not None:
        try:
            widget.viewer.window.remove_dock_widget(widget=widget.docked_plot)
            widget.docked_plot = None
        except Exception as e:
            pass
    # Remove the previous layers from the viewer
    clear_layers(widget.viewer)
    # Add the wells and the filter points to the viewer, in napari we can add, remove and move the points
    for name in WELL_NAMES:
        visible = (name == WELL_NAMES[0])
        # if the peak detection happened once the well_data contains more data: the wells, the selected points and the filter points
        # so we need to select the first element of the tuple
        if hasattr(widget, 'cell_selector') and widget.cell_selector:
            widget.viewer.add_image(widget.well_data[name][0], name=name, colormap='viridis', visible=visible)
        else:
            widget.viewer.add_image(widget.well_data[name], name=name, colormap='viridis', visible=visible)
        widget.viewer.add_points(invert_coords(widget.filter_ptss[name]), name=name + widget._bg_points, size=1, face_color='orange', visible=visible)

    # Once the background selection is started new data can't be loaded for now because of a napari draw bug
    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)

def load_and_preprocess_data_GUI(widget):
    """
    Simpliy clear the layers and add the wells to the viewer. 
    The next step is the preprocessing so the process button is enabled.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    clear_layers(widget.viewer)
    for name in WELL_NAMES:
        visible = (name == WELL_NAMES[0])
        widget.viewer.add_image(widget.well_data[name], name=name, colormap='viridis', visible=visible)

    widget.backgroundSelectorButton.setEnabled(True)

def peak_detection_GUI(widget):
    """
    Simpliy clear the layers and add the wells and peaks to the viewer. 
    Also add the cell signals plot to the viewer.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    clear_layers(widget.viewer)
    for name in widget.remaining_wells:
        visible = (name == widget.remaining_wells[0])
        widget.viewer.add_image(widget.well_data[name][0], name=name, colormap='viridis', visible=visible)
        # invert the coordinates of the peaks to plot in napari (later invert back for other plots)
        widget.viewer.add_points(invert_coords(widget.well_data[name][1]), name=name + widget._peaks, size=1, face_color='red', visible=visible)
        # filter points for background selection
        # widget.viewer.add_points(invert_coords(widget.well_data[name][2]), name=name + ' filter', size=1, face_color='blue', visible=visible)

    current_line = get_cell_line_by_coords(widget.well_data[widget.remaining_wells[0]][0], 0, 0, widget.phases)
    well_data = {key: value[0] for key, value in widget.well_data.items()}
    # Adds the cell signal plot to the viewer
    plot_GUI(widget, well_data, current_line)
    
    # Once the peak detection is started new data can't be loaded for now because of a napari draw bug
    widget.backgroundSelectorButton.setEnabled(True)
    widget.peakButton.setEnabled(True)
    widget.exportButton.setEnabled(True)
    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)

def SRUNet_segmentation_GUI(widget):
    """
    Simpliy clear the layers and add the wells and segments to the viewer. 
    This is for the super resolution model so the well images needs to be upscaled to the model output size,
    but only the last frame because if we interpolate the whole well data it will be to long and needs to much memory.
    Also add the cell signals plot to the viewer.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    clear_layers(widget.viewer)
    for i in range(len(WELL_NAMES)):
        visible = (i == 0)
        name = WELL_NAMES[i]            
        well_tensor = torch.tensor(widget.well_data[name][-1]).unsqueeze(0).unsqueeze(0)
        upscaled_well = torch.nn.functional.interpolate(well_tensor, size=(widget.image_size, widget.image_size), mode='nearest').squeeze(0).squeeze(0).numpy()
        widget.viewer.add_image(upscaled_well, name=name, colormap='viridis', visible=visible)
        widget.viewer.add_labels(widget.labels[i], name=name + widget._segment, visible=visible)

    current_line = get_cell_line_by_coords(widget.well_data[WELL_NAMES[0]], 0, 0, widget.phases)
    plot_GUI(widget, widget.well_data, current_line)

    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)   
    widget.exportButton.setEnabled(True)
    widget.segmentationButton.setEnabled(True)
    widget.segmentationButton.setText("Segment")

def UNet_segmentation_GUI(widget):
    """
    Simpliy clear the layers and add the wells and segments to the viewer. 
    This is for the normal model so the well images are not upscaled.
    Also add the cell signals plot to the viewer.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    """
    clear_layers(widget.viewer)
    for i in range(len(WELL_NAMES)):
        visible = (i == 0)
        name = WELL_NAMES[i]
        widget.viewer.add_image(widget.well_data[name], name=name, colormap='viridis', visible=visible)
        widget.viewer.add_labels(widget.labels[i], name=name + widget._segment, visible=visible)

    current_line = get_cell_line_by_coords(widget.well_data[WELL_NAMES[0]], 0, 0, widget.phases)
    plot_GUI(widget, widget.well_data, current_line)
    
    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)   
    widget.exportButton.setEnabled(True)
    widget.segmentationButton.setEnabled(True)
    widget.segmentationButton.setText("Segment")

def plot_GUI(widget, well_data, current_line):
    """
    Removes the previous plot and adds a new one to the viewer.
    Also adds a double click callback to the layers to update the plot with the selected cell.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    @param well_data: The well data with ONLY the cell signals. Passed to the callback function.
    @param current_line: The base line to plot. The callback function will update this line with the selected cell.
    """
    if widget.docked_plot is not None:
        try:
            # Attempt to remove the docked plot if it exists
            widget.viewer.window.remove_dock_widget(widget.docked_plot)
        except Exception as e:
            pass
        
    # create mpl figure with subplots
    mpl_fig = plt.figure()
    ax = mpl_fig.add_subplot(111)   # 1 row, 1 column, 1st plot
    (line,) = ax.plot(widget.time, current_line)
    # add the figure to the viewer as a FigureCanvas widget
    widget.docked_plot = widget.viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Cell signal plot')
    widget.docked_plot.setMinimumSize(200, 300)

    add_double_click_callbacks_to_layers(widget, well_data, ax)


def add_double_click_callbacks_to_layers(widget, well_data, ax):
    """
    Updates the plot with the selected cell when the user double clicks on a layer.
    @param widget: Needs the widget to access the GUI elements and save the data to the widget.
    @param well_data: The well data with ONLY the cell signals.
    @param ax: The matplotlib axis to plot the data.
    """
    for layer in widget.viewer.layers:
        @layer.mouse_double_click_callbacks.append
        def update_plot_on_double_click(layer, event):
            try:
                # get the well name from the layer name
                name = layer.name.split(' ')[0]
                ax.clear()
                # get the x and y coordinates of the double click event, it is different if the image is upscaled
                if widget.scaling_factor == 1:
                    x = int(event.position[1])
                    y = int(event.position[2])
                else:
                    x = int(event.position[0]/widget.scaling_factor)
                    y = int(event.position[1]/widget.scaling_factor)
                # squeeze the x and y coordinates to the well size
                x = max(0, min(x, 79))
                y = max(0, min(y, 79))
                # get the cell signal of the selected cell
                current_line = get_cell_line_by_coords(well_data[name], x, y, widget.phases)
                (line,) = ax.plot(widget.time, current_line)
                ax.set_title(f"Well: {name}, Cell: [{x} {y}]")
                line.figure.canvas.draw()
            except IndexError:
                pass

# GUI helper functions
def invert_coords(coords):
    # Invert the coordinates for napari plotting
    return np.array([[y, x] for x, y in coords])

def lin_indices(original_length, subsampled_length):
    """ 
    Creates linearly spaced indices for subsampling the biosensor data.
    @param original_length: The length of the original data.
    @param subsampled_length: The length of the subsampled data.
    @return: The linearly spaced indices.
    """
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]

def clear_layers(viewer):
    # Remove all layers from the viewer (is there a better way to do this?)
    viewer.layers.select_all()
    viewer.layers.remove_selected()

def loaded_params_to_GUI(widget):
    """
    After loading in the parameters from the previous run we need to set the values in the GUI so it is clear what was used and can be changed.
    But only if the parameters are loaded in.
    """
    if widget.preprocessing_params_loaded:
        widget.preprocessing_params = widget.preprocessing_params_loaded
        widget.horizontalFlip.setChecked(widget.preprocessing_params['flip'][0])
        widget.verticalFlip.setChecked(widget.preprocessing_params['flip'][1])
        widget.rangeTypeSelect.setCurrentIndex(widget.preprocessing_params['signal_range']['range_type'])
        widget.rangeMin.setValue(widget.preprocessing_params['signal_range']['ranges'][0])
        if widget.preprocessing_params['signal_range']['ranges'][1] is None:
            widget.rangeMax.setValue(len(widget.full_phases)+1)
        else:
            widget.rangeMax.setValue(widget.preprocessing_params['signal_range']['ranges'][1])
        widget.threshold.setValue(widget.preprocessing_params['drift_correction']['threshold'])
        widget.filterMethod.setCurrentText(widget.preprocessing_params['drift_correction']['filter_method'])
    # in the segmentation widget there are no localization parameters
    if hasattr(widget, 'thresholdRangeMin') and widget.localization_params_loaded:
        widget.localization_params = widget.localization_params_loaded
        widget.thresholdRangeMin.setValue(widget.localization_params['threshold_range'][0])
        widget.thresholdRangeMax.setValue(widget.localization_params['threshold_range'][1])
        widget.neighbourhoodSize.setValue(widget.localization_params['neighbourhood_size'])
        widget.errorMaskFiltering.setChecked(widget.localization_params['error_mask_filtering'])

def get_filter_points(viewer, bg_name=" background points"):
    """
    Returns the filter points from the viewer. The filter points are the background points.
    @param viewer: The napari viewer.
    @param bg_name: The name of the background point layers after the well name.
    @return: The filter points.
    """
    filter_ptss = {}
    for name in WELL_NAMES:
        filter_ptss[name] = invert_coords(np.round(viewer.layers[name + bg_name].data)).astype(np.uint8)
    return filter_ptss

def get_selected_cells(viewer, remaining_wells, peaks_name):
    """
    Returns the selected cells from the viewer. The selected cells are the peaks.
    @param viewer: The napari viewer.
    @param remaining_wells: The remaining wells after the peak detection. That are not deleted by the user.
    @param peaks_name: The name of the peak layers after the well name.
    @return: The selected cells.
    """
    selected_ptss = {}
    for name in remaining_wells:
        selected_ptss[name] = invert_coords(np.round(viewer.layers[name + peaks_name].data)).astype(np.uint8)
    return selected_ptss

def get_remaining_wells_from_layers(viewer):
    """
    Returns the remaining wells from the viewer. 
    The remaining wells are the wells that are not deleted by the user and will be shown and exported.
    This function can be more complex but it depends on the user interaction with the viewer. Later it can be extended.
    @param viewer: The napari viewer.
    @return: The remaining wells.
    """
    remaining_wells = []
    for layer in viewer.layers:
        well_name = layer.name.split()[0]
        if well_name not in remaining_wells:
            remaining_wells.append(well_name)
    return remaining_wells

def get_cell_line_by_coords(well_data, x, y, phases):
    """
    @param well_data: The well data. ONLY the biosensor data. so its a 3D numpy array.
    @param x: The x coordinate of the selected cell. Must be between 0 and 79!
    @param y: The y coordinate of the selected cell. Must be between 0 and 79!
    @param phases: The phases of the well data.
    @return: The cell signal of the selected cell.
    """
    current_line = well_data[:, x, y].copy()
    # Set the phases to nan so in the plot it will be a gap
    if len(phases) > 0:
        for p in phases:
            current_line[p] = np.nan
    return current_line

# Parameter update functions
def update_preprocessing_params(widget):
    return {
        'flip': [widget.horizontalFlip.isChecked(), widget.verticalFlip.isChecked()],
        'signal_range' : {
        'range_type': widget.rangeTypeSelect.currentIndex(),
        'ranges': [widget.rangeMin.value(), widget.rangeMax.value() if widget.rangeMax.value() != len(widget.full_phases)+1 else None], 
        },
        'drift_correction': {
        'threshold': widget.threshold.value(),
        'filter_method': widget.filterMethod.currentText(),
        'background_selector': widget.background_selector,
        }
    }

def update_localization_params(widget):
    return {
        'threshold_range': [widget.thresholdRangeMin.value(), widget.thresholdRangeMax.value()],
        'neighbourhood_size': widget.neighbourhoodSize.value(),
        'error_mask_filtering': widget.errorMaskFiltering.isChecked()
    }

def update_export_params(widget):
    return {
        'coordinates': widget.coordinates.isChecked(),
        'preprocessed_signals': widget.preprocessedSignals.isChecked(),
        'raw_signals': widget.rawSignals.isChecked(),
        'average_signal': widget.averageSignal.isChecked(),
        'breakdown_signal': widget.breakdownSignal.isChecked(),
        'max_well': widget.maxWell.isChecked(),
        'plot_signals_with_well': widget.plotSignalsWithWell.isChecked(),
        'plot_well_with_coordinates': widget.plotWellWithCoordinates.isChecked(),
        'plot_cells_individually': widget.plotCellsIndividually.isChecked(),
        'signal_parts_by_phases': widget.signalPartsByPhases.isChecked(),
        'max_centered_signals': widget.maxCenteredSignals.isChecked()
    }
