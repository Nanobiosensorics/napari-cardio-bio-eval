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


# thread worker functions
@thread_worker
def load_data_thread(widget):
    path = widget.dirLineEdit.text()
    widget.RESULT_PATH = os.path.join(path, 'result')
    if not os.path.exists(widget.RESULT_PATH):
        os.mkdir(widget.RESULT_PATH)

    widget.raw_wells, widget.full_time, widget.full_phases = load_data(path, flip=widget.preprocessing_params['flip'])
    widget.filter_params, widget.preprocessing_params_loaded, widget.localization_params_loaded = load_params(widget.RESULT_PATH)

    # we load in the parameters from the previous run if they exist
    # and set the values in the GUI so it is clear what was used and can be changed
    loaded_params_to_GUI(widget)

    widget.rangeLabel.setText(f'Phases: {[(n+1, p) for n, p in enumerate(widget.full_phases)]}, Time: {len(widget.full_time)}')
    widget.rangeTypeSelect.currentIndexChanged.connect(widget.range_type_changed)
    # Enable the range selection
    widget.rangeTypeSelect.setEnabled(True)
    widget.rangeMin.setEnabled(True)
    widget.rangeMax.setEnabled(True)
    # Set the extremes of the range
    widget.range_type_changed()

    widget.processButton.setEnabled(True)

@thread_worker
def preprocess_data_thread(widget):
    widget.preprocessing_params = update_preprocessing_params(widget)

    widget.well_data, widget.time, widget.phases, widget.filter_ptss, widget.selected_range = preprocessing(widget.preprocessing_params, widget.raw_wells, widget.full_time, widget.full_phases, widget.filter_params)

@thread_worker
def peak_detection_thread(widget):
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

    widget.remaining_wells = get_remaining_wells_from_layers(widget.viewer)

@thread_worker
def segmentation_thread(self):
    self.model = torch.jit.load(self.modelPath.text())
    biosensor = []
    # the indices length may vary later but for now it is 8
    bio_len = 8
    for name in WELL_NAMES: # remaining_wells?
        biosensor.append(self.well_data[name][lin_indices(self.well_data[name].shape[0], bio_len)])

    biosensor_tensor = torch.tensor(np.array(biosensor)).float() 

    with torch.no_grad():
        output = self.model(biosensor_tensor)

    self.image_size = output.shape[2]
    self.scaling_factor = self.image_size // 80

    output = torch.sigmoid(output).squeeze().detach().numpy()
    self.bin_output = (output > 0.5).astype(int)

    self.cell_centers = []
    self.labels = []
    for i in range(len(WELL_NAMES)):
        pred = self.bin_output[i].squeeze().astype(np.uint8)
        _, labels, _, centers = cv2.connectedComponentsWithStats(pred, connectivity=8)
        self.cell_centers.append(centers[1:])
        self.labels.append(labels)

@thread_worker
def export_results_thread(widget):
    export_results(widget.export_params, widget.RESULT_PATH, widget.selected_ptss, widget.filter_ptss, widget.well_data, 
                   widget.time, widget.phases, widget.raw_wells, widget.full_time, widget.full_phases, widget.selected_range)


# GUI functions
def manual_background_selection(widget):
    widget.background_selector = True

    if widget.docked_plot is not None:
        widget.viewer.window.remove_dock_widget(widget=widget.docked_plot)
        widget.docked_plot = None

    clear_layers(widget.viewer)

    for name in WELL_NAMES:
        visible = (name == WELL_NAMES[0])
        # if the peak detection happened once the well_data contains more data: the wells, the selected points and the filter points
        # so we need to select the first element of the tuple
        if hasattr(widget, 'cell_selector') and widget.cell_selector:
            widget.viewer.add_image(widget.well_data[name][0], name=name, colormap='viridis', visible=visible)
        else:
            widget.viewer.add_image(widget.well_data[name], name=name, colormap='viridis', visible=visible)
        widget.viewer.add_points(invert_coords(widget.filter_ptss[name]), name=name + widget._bg_points, size=1, face_color='orange', visible=visible)

    # Once the background selection is started new data cant be loaded
    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)

def load_and_preprocess_data_GUI(widget):
    clear_layers(widget.viewer)
    for name in WELL_NAMES:
        visible = (name == WELL_NAMES[0])
        widget.viewer.add_image(widget.well_data[name], name=name, colormap='viridis', visible=visible)

    widget.backgroundSelectorButton.setEnabled(True)

def peak_detection_GUI(widget):
    clear_layers(widget.viewer)
    for name in widget.remaining_wells:
        visible = (name == widget.remaining_wells[0])
        widget.viewer.add_image(widget.well_data[name][0], name=name, colormap='viridis', visible=visible)
        # invert the coordinates of the peaks to plot in napari (later invert back for other plots)
        widget.viewer.add_points(invert_coords(widget.well_data[name][1]), name=name + widget._peaks, size=1, face_color='red', visible=visible)
        # filter points for background selection
        # widget.viewer.add_points(invert_coords(widget.well_data[name][-1]), name=name + ' filter', size=1, face_color='blue', visible=visible)

    current_line = get_cell_line_by_coords(widget.well_data[widget.remaining_wells[0]][0], 0, 0, widget.phases)
    well_data = {key: value[0] for key, value in widget.well_data.items()}

    plot_GUI(widget, well_data, current_line)
    
    # Once the peak detection is started new data cant be loaded
    widget.backgroundSelectorButton.setEnabled(True)
    widget.peakButton.setEnabled(True)
    widget.exportButton.setEnabled(True)
    widget.loadButton.setEnabled(False)
    widget.processButton.setEnabled(False)

def SRUNet_segmentation_GUI(self):
    clear_layers(self.viewer)
    for i in range(len(WELL_NAMES)):
        visible = (i == 0)
        name = WELL_NAMES[i]            
        well_tensor = torch.tensor(self.well_data[name][-1]).unsqueeze(0).unsqueeze(0)
        upscaled_well = torch.nn.functional.interpolate(well_tensor, size=(self.image_size, self.image_size), mode='nearest').squeeze(0).squeeze(0).numpy()
        self.viewer.add_image(upscaled_well, name=name, colormap='viridis', visible=visible)
        self.viewer.add_labels(self.labels[i], name=name + self._segment, visible=visible)

    current_line = get_cell_line_by_coords(self.well_data[WELL_NAMES[0]], 0, 0, self.phases)
    plot_GUI(self, self.well_data, current_line)

    self.loadButton.setEnabled(False)
    self.processButton.setEnabled(False)   
    self.exportButton.setEnabled(True)
    self.segmentationButton.setEnabled(True)
    self.segmentationButton.setText("Segment")

def UNet_segmentation_GUI(self):
    clear_layers(self.viewer)
    for i in range(len(WELL_NAMES)):
        visible = (i == 0)
        name = WELL_NAMES[i]
        self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)
        self.viewer.add_labels(self.labels[i], name=name + self._segment, visible=visible)

    current_line = get_cell_line_by_coords(self.well_data[WELL_NAMES[0]], 0, 0, self.phases)
    plot_GUI(self, self.well_data, current_line)
    
    self.loadButton.setEnabled(False)
    self.processButton.setEnabled(False)   
    self.exportButton.setEnabled(True)
    self.segmentationButton.setEnabled(True)
    self.segmentationButton.setText("Segment")

def plot_GUI(widget, well_data, current_line):
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
    for layer in widget.viewer.layers:
        @layer.mouse_double_click_callbacks.append
        def update_plot_on_double_click(layer, event):
            try:
                name = layer.name.split(' ')[0]
                ax.clear()

                if widget.scaling_factor == 1:
                    x = int(event.position[1])
                    y = int(event.position[2])
                else:
                    x = int(event.position[0]/widget.scaling_factor)
                    y = int(event.position[1]/widget.scaling_factor)

                x = max(0, min(x, 79))
                y = max(0, min(y, 79))

                current_line = get_cell_line_by_coords(well_data[name], x, y, widget.phases)
                (line,) = ax.plot(widget.time, current_line)
                ax.set_title(f"Well: {name}, Cell: [{x} {y}]")
                line.figure.canvas.draw()
            except IndexError:
                pass

# GUI helper functions
def invert_coords(coords):
    return np.array([[y, x] for x, y in coords])

def lin_indices(original_length, subsampled_length):
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]

def clear_layers(viewer):
    viewer.layers.select_all()
    viewer.layers.remove_selected()

def loaded_params_to_GUI(widget):
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
    
    if hasattr(widget, 'thresholdRangeMin') and widget.localization_params_loaded:
        widget.localization_params = widget.localization_params_loaded
        widget.thresholdRangeMin.setValue(widget.localization_params['threshold_range'][0])
        widget.thresholdRangeMax.setValue(widget.localization_params['threshold_range'][1])
        widget.neighbourhoodSize.setValue(widget.localization_params['neighbourhood_size'])
        widget.errorMaskFiltering.setChecked(widget.localization_params['error_mask_filtering'])

def get_filter_points(viewer, bg_name):
    filter_ptss = {}
    for name in WELL_NAMES:
        filter_ptss[name] = invert_coords(np.round(viewer.layers[name + bg_name].data)).astype(np.uint8)
    return filter_ptss

def get_selected_cells(viewer, remaining_wells, peaks_name):
    selected_ptss = {}
    for name in remaining_wells:
        selected_ptss[name] = invert_coords(np.round(viewer.layers[name + peaks_name].data)).astype(np.uint8)
    return selected_ptss

def get_remaining_wells_from_layers(viewer):
    remaining_wells = []
    for layer in viewer.layers:
        well_name = layer.name.split()[0]
        if well_name not in remaining_wells:
            remaining_wells.append(well_name)
    return remaining_wells

def get_cell_line_by_coords(well_data, x, y, phases):
        # x and y must be between 0 and 80!
        current_line = well_data[:, x, y].copy()
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
