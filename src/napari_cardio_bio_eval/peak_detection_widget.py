import os
import napari
import matplotlib.pyplot as plt
import numpy as np

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar)

from nanobio_core.epic_cardio.processing import RangeType, load_data, load_params, preprocessing, localization, save_params
from nanobio_core.epic_cardio.defs import WELL_NAMES
from export_and_plot.export import export_results

from napari.qt.threading import thread_worker
from matplotlib.backends.backend_qt5agg import FigureCanvas


class PeakDetectionWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._peaks = " peaks"
        self._bg_points = " background points"
        self.docked_plot = None
        self.background_selector = False
        self.full_phases = []
        self.initUI()

    def initUI(self):
        self.layout = QFormLayout(self)

        # Directory selection
        dataLoadingLabel = QLabel('Data loading:')
        dataLoadingLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(dataLoadingLabel)

        self.browseBox = QHBoxLayout()
        self.dirLineEdit = QLineEdit(self)
        self.browseButton = QPushButton('Browse', self)
        self.browseButton.clicked.connect(self.open_file_name_dialog)
        self.browseBox.addWidget(self.dirLineEdit)
        self.browseBox.addWidget(self.browseButton)
        self.layout.addRow(QLabel('Select Directory:'), self.browseBox)

        # Parameter inputs
        self.flipBox = QHBoxLayout()
        self.horizontalFlip = QCheckBox('Horizontal', self)
        self.flipBox.addWidget(self.horizontalFlip)
        self.verticalFlip = QCheckBox('Vertical', self)
        self.flipBox.addWidget(self.verticalFlip)
        self.layout.addRow(QLabel('Fliping:'), self.flipBox)

        # Data loading button
        self.loadButton = QPushButton('Load Data', self)
        self.loadButton.clicked.connect(self.load_data)
        self.layout.addRow(self.loadButton)
        # Range type selection, the change function is added in the load_data function after the data is loaded
        self.layout.addRow(QLabel('Select signal range:'))
        self.rangeLabel = QLabel('Phases: , Time: ')
        self.layout.addRow(self.rangeLabel)
        self.rangeTypeSelect = QComboBox(self)
        self.rangeTypeSelect.addItems(['Measurement phase', 'Individual point'])
        self.rangeTypeSelect.setEnabled(False)
        self.rangeTypeSelect.setCurrentIndex(1)
        self.layout.addRow(QLabel('Range type:'), self.rangeTypeSelect)
        # Range thresholds the minimum is always 0, the maximum is set in the load_data function when the data is loaded
        self.rangesBox = QHBoxLayout()
        self.rangeMin = QSpinBox(self)
        self.rangeMin.setMinimum(0)
        self.rangeMin.setValue(0)
        self.rangeMin.setEnabled(False)
        self.rangeMax = QSpinBox(self)
        self.rangeMax.setMinimum(0)
        self.rangeMax.setEnabled(False)
        self.rangesBox.addWidget(self.rangeMin)
        self.rangesBox.addWidget(self.rangeMax)
        self.layout.addRow(QLabel('Range:'), self.rangesBox)

        self.layout.addRow(QLabel('Drift correction:'))
        self.threshold = QSpinBox(self)
        self.threshold.setMinimum(25)
        self.threshold.setMaximum(500)
        self.threshold.setValue(75)
        self.layout.addRow(QLabel('Threshold:'), self.threshold)

        self.filterMethod = QComboBox(self)
        self.filterMethod.addItems(['mean', 'median'])
        self.layout.addRow(QLabel('Filter method:'), self.filterMethod)

        # Data processing button
        self.processButton = QPushButton('Preprocess Data', self)
        self.processButton.setEnabled(False)
        self.processButton.clicked.connect(self.preprocess_data)
        self.layout.addRow(self.processButton)

        # Manual Background selection
        manBGsel = QLabel('Manual background selection if needed:')
        manBGsel.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; }")
        self.layout.addRow(manBGsel)
        self.backgroundSelectorButton = QPushButton('Select Background Points Manually', self)
        self.backgroundSelectorButton.clicked.connect(self.manual_background_selection)
        self.backgroundSelectorButton.setEnabled(False)
        self.layout.addRow(self.backgroundSelectorButton)

        # Peak detection parameters
        peakDetLabel = QLabel('Peak detection parameters:')
        peakDetLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(peakDetLabel)

        self.thresholdBox = QHBoxLayout()
        self.thresholdRangeMin = QSpinBox(self)
        self.thresholdRangeMin.setMinimum(25)
        self.thresholdRangeMin.setMaximum(5000)
        self.thresholdRangeMin.setValue(75)
        self.thresholdRangeMax = QSpinBox(self)
        self.thresholdRangeMax.setMinimum(25)
        self.thresholdRangeMax.setMaximum(5000)
        self.thresholdRangeMax.setValue(3000)
        self.thresholdBox.addWidget(self.thresholdRangeMin)
        self.thresholdBox.addWidget(self.thresholdRangeMax)
        self.layout.addRow(QLabel('Threshold range:'), self.thresholdBox)

        self.neighbourhoodSize = QSpinBox(self)
        self.neighbourhoodSize.setMinimum(1)
        self.neighbourhoodSize.setMaximum(10)
        self.neighbourhoodSize.setValue(3)
        self.layout.addRow(QLabel('Neighbourhood size:'), self.neighbourhoodSize)

        self.errorMaskFiltering = QCheckBox('Error Mask Filtering', self)
        self.errorMaskFiltering.setChecked(True)
        self.layout.addRow(self.errorMaskFiltering)
        # Peak detection button
        self.peakButton = QPushButton('Detect Signal Peaks', self)
        self.peakButton.clicked.connect(self.peak_detection)
        self.peakButton.setEnabled(False)
        self.layout.addRow(self.peakButton)

        # Export parameters
        dataLoadingLabel = QLabel('Exporting options:')
        dataLoadingLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(dataLoadingLabel)

        self.coordinates = QCheckBox('Coordinates', self)
        self.coordinates.setChecked(True)
        self.layout.addRow(self.coordinates)

        self.preprocessedSignals = QCheckBox('Preprocessed Signals', self)
        self.preprocessedSignals.setChecked(True)
        self.layout.addRow(self.preprocessedSignals)

        self.rawSignals = QCheckBox('Raw Signals', self)
        self.rawSignals.setChecked(True)
        self.layout.addRow(self.rawSignals)

        self.averageSignal = QCheckBox('Average Signal', self)
        self.layout.addRow(self.averageSignal)
        self.breakdownSignal = QCheckBox('Breakdown Signal', self)
        self.layout.addRow(self.breakdownSignal)

        self.maxWell = QCheckBox('Max Well', self)
        self.maxWell.setChecked(True)
        self.layout.addRow(self.maxWell)

        self.plotSignalsWithWell = QCheckBox('Plot Signals with Well', self)
        self.plotSignalsWithWell.setChecked(True)
        self.layout.addRow(self.plotSignalsWithWell)

        self.plotWellWithCoordinates = QCheckBox('Plot Well with Coordinates', self)
        self.plotWellWithCoordinates.setChecked(True)
        self.layout.addRow(self.plotWellWithCoordinates)

        self.plotCellsIndividually = QCheckBox('Plot Cells Individually', self)
        self.layout.addRow(self.plotCellsIndividually)
        self.signalPartsByPhases = QCheckBox('Signal Parts by Phases', self)
        self.layout.addRow(self.signalPartsByPhases)
        self.maxCenteredSignals = QCheckBox('Max Centered Signals', self)
        self.layout.addRow(self.maxCenteredSignals)
        # Export button
        self.exportButton = QPushButton('Export Data', self)
        self.exportButton.clicked.connect(self.export_data)
        self.exportButton.setEnabled(False)
        self.layout.addRow(self.exportButton)
        # Export progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(1)
        self.layout.addRow(self.progressBar)

    def range_type_changed(self):
        if self.rangeTypeSelect.currentIndex() == 0:
            num_of_phases = len(self.full_phases)
            self.rangeMin.setMaximum(num_of_phases)
            self.rangeMax.setMaximum(num_of_phases + 1)
            self.rangeMax.setValue(num_of_phases + 1)
        else:
            frame_count = len(self.full_time)
            self.rangeMin.setMaximum(frame_count)
            self.rangeMax.setMaximum(frame_count)
            self.rangeMax.setValue(frame_count)

    def open_file_name_dialog(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dirLineEdit.setText(directory)

    def load_data(self):
        self.preprocessing_params = update_preprocessing_params(self)

        self.cell_selector = False
        loader = self.load_data_thread()
        loader.start()

    @thread_worker
    def load_data_thread(self):
        set_buttons_enabled(self, False)
        path = self.dirLineEdit.text()
        self.RESULT_PATH = os.path.join(path, 'result')
        if not os.path.exists(self.RESULT_PATH):
            os.mkdir(self.RESULT_PATH)

        self.raw_wells, self.full_time, self.full_phases = load_data(path, flip=self.preprocessing_params['flip'])
        self.filter_params, self.preprocessing_params_loaded, self.localization_params_loaded = load_params(self.RESULT_PATH)

        # we load in the parameters from the previous run if they exist
        # and set the values in the GUI so it is clear what was used and can be changed
        if self.preprocessing_params_loaded:
            self.preprocessing_params = self.preprocessing_params_loaded
            self.horizontalFlip.setChecked(self.preprocessing_params['flip'][0])
            self.verticalFlip.setChecked(self.preprocessing_params['flip'][1])
            self.rangeTypeSelect.setCurrentIndex(self.preprocessing_params['signal_range']['range_type'])
            self.rangeMin.setValue(self.preprocessing_params['signal_range']['ranges'][0])
            if self.preprocessing_params['signal_range']['ranges'][1] is None:
                self.rangeMax.setValue(len(self.full_phases)+1)
            else:
                self.rangeMax.setValue(self.preprocessing_params['signal_range']['ranges'][1])
            self.threshold.setValue(self.preprocessing_params['drift_correction']['threshold'])
            self.filterMethod.setCurrentText(self.preprocessing_params['drift_correction']['filter_method'])

        if self.localization_params_loaded:
            self.localization_params = self.localization_params_loaded
            self.thresholdRangeMin.setValue(self.localization_params['threshold_range'][0])
            self.thresholdRangeMax.setValue(self.localization_params['threshold_range'][1])
            self.neighbourhoodSize.setValue(self.localization_params['neighbourhood_size'])
            self.errorMaskFiltering.setChecked(self.localization_params['error_mask_filtering'])

        self.rangeLabel.setText(f'Phases: {[(n+1, p) for n, p in enumerate(self.full_phases)]}, Time: {len(self.full_time)}')
        self.rangeTypeSelect.currentIndexChanged.connect(self.range_type_changed)

        # Enable the range selection
        self.rangeTypeSelect.setEnabled(True)
        self.rangeMin.setEnabled(True)
        self.rangeMax.setEnabled(True)
        if self.rangeTypeSelect.currentIndex() == 0:
            self.rangeMin.setMaximum(len(self.full_phases))
            self.rangeMax.setMaximum(len(self.full_phases)+1)
            self.rangeMax.setValue(len(self.full_phases)+1)
        else:
            self.rangeMin.setMaximum(len(self.full_time))
            self.rangeMax.setMaximum(len(self.full_time))
            self.rangeMax.setValue(len(self.full_time))

        self.processButton.setEnabled(True)

    def preprocess_data(self):
        preprocessor = self.preprocess_data_thread()
        preprocessor.finished.connect(self.load_and_preprocess_data_GUI)
        preprocessor.start()

    @thread_worker
    def preprocess_data_thread(self):
        self.preprocessing_params = update_preprocessing_params(self)
        # It means that the range is set to the last phase or time point, it would be out of index
        # if self.rangeMax.value() == len(self.full_phases)+1:
        #     self.preprocessing_params['signal_range']['ranges'] = [self.rangeMin.value(), None]

        self.well_data, self.time, self.phases, self.filter_ptss, self.selected_range = preprocessing(self.preprocessing_params, self.raw_wells, self.full_time, self.full_phases, self.filter_params)

    def load_and_preprocess_data_GUI(self):
        clear_layers(self.viewer)
        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)

        self.peakButton.setEnabled(True)
        self.backgroundSelectorButton.setEnabled(True)

    def manual_background_selection(self):
        self.background_selector = True
        
        if self.docked_plot is not None:
            self.viewer.window.remove_dock_widget(widget=self.docked_plot)
            self.docked_plot = None

        clear_layers(self.viewer)

        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            # if the peak detection happened once the well_data contains more data: the wells, the selected points and the filter points
            # so we need to select the first element of the tuple
            if self.cell_selector:
                self.viewer.add_image(self.well_data[name][0], name=name, colormap='viridis', visible=visible)
            else:
                self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)
            self.viewer.add_points(invert_coords(self.filter_ptss[name]), name=name + self._bg_points, size=1, face_color='orange', visible=visible)

        # Once the background selection is started new data cant be loaded
        self.loadButton.setEnabled(False)
        self.processButton.setEnabled(False)

    def peak_detection(self):
        self.localization_params = update_localization_params(self)

        self.cell_selector = True
        peak_detector = self.peak_detection_thread()
        peak_detector.finished.connect(self.peak_detection_GUI) 
        peak_detector.start()

    @thread_worker
    def peak_detection_thread(self):
        set_buttons_enabled(self, False)
        if self.background_selector:
            self.filter_ptss = get_filter_points(self.viewer, self._bg_points)

        self.preprocessing_params = update_preprocessing_params(self)
        self.localization_params = update_localization_params(self)

        self.background_selector = False

        # From here the well data contains the wells, the selected points and the filter points (which are the background points)
        self.well_data = localization(self.preprocessing_params, self.localization_params,
                                      self.raw_wells, self.selected_range, self.filter_ptss)

        self.remaining_wells = get_remaining_wells_from_layers(self.viewer)

    def peak_detection_GUI(self):
        clear_layers(self.viewer)
        # visualize the data with peaks
        for name in self.remaining_wells:
            visible = (name == self.remaining_wells[0])
            self.viewer.add_image(self.well_data[name][0], name=name, colormap='viridis', visible=visible)
            # invert the coordinates of the peaks to plot in napari (later invert back for other plots)
            self.viewer.add_points(invert_coords(self.well_data[name][1]), name=name + self._peaks, size=1, face_color='red', visible=visible)
            # filter points for background selection
            # self.viewer.add_points(invert_coords(self.well_data[name][-1]), name=name + ' filter', size=1, face_color='blue', visible=visible)

        current_line = get_cell_line_by_coords(self.well_data[self.remaining_wells[-1]][0], 0, 0, self.phases)

        if self.docked_plot is not None:
            self.viewer.window.remove_dock_widget(widget=self.docked_plot)
            
        # create mpl figure with subplots
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)   # 1 row, 1 column, 1st plot
        (line,) = ax.plot(self.time, current_line)
        # add the figure to the viewer as a FigureCanvas widget
        self.docked_plot = self.viewer.window.add_dock_widget(FigureCanvas(mpl_fig), name='Cell signal plot')
        self.docked_plot.setMinimumSize(200, 300)

        # add a double click callback to all of the layers
        # the well name in the layer name is important to get the selected layer
        for layer in self.viewer.layers:
            @layer.mouse_double_click_callbacks.append
            def update_plot_on_double_click(layer, event):
                try:
                    name = layer.name.split(' ')[0]
                    ax.clear()
                    current_line = get_cell_line_by_coords(self.well_data[name][0], int(event.position[1]), int(event.position[2]), self.phases)
                    (line,) = ax.plot(self.time, current_line)
                    ax.set_title(f"Well: {name}, Cell: [{int(event.position[1])} {int(event.position[2])}]")
                    line.figure.canvas.draw()
                except IndexError:
                    pass
        
        # Once the peak detection is started new data cant be loaded
        set_buttons_enabled(self, True)
        self.loadButton.setEnabled(False)
        self.processButton.setEnabled(False)

    def export_data(self):
        self.export_params = update_export_params(self)
        self.progressBar.setMaximum(0)
        self.remaining_wells = get_remaining_wells_from_layers(self.viewer)
        self.selected_ptss = get_selected_cells(self.viewer, self.remaining_wells, self._peaks)

        for name in self.remaining_wells:
            self.well_data[name] = (self.viewer.layers[name].data, self.selected_ptss[name], self.well_data[name][-1])

        self.preprocessing_params = update_preprocessing_params(self)
        self.localization_params = update_localization_params(self)

        save_params(self.RESULT_PATH, self.well_data, self.preprocessing_params, self.localization_params)

        exporter = export_res(self)
        exporter.finished.connect(lambda: self.progressBar.setMaximum(1))
        exporter.start()

@thread_worker
def export_res(widget):
    export_results(widget.export_params, widget.RESULT_PATH, widget.selected_ptss, widget.filter_ptss, widget.well_data, 
                   widget.time, widget.phases, widget.raw_wells, widget.full_time, widget.full_phases, widget.selected_range)

def clear_layers(viewer):
    viewer.layers.select_all()
    viewer.layers.remove_selected()

def invert_coords(coords):
    return np.array([[y, x] for x, y in coords])

def set_buttons_enabled(widget, state):
    # state is a boolean
    widget.backgroundSelectorButton.setEnabled(state)
    widget.peakButton.setEnabled(state)
    widget.exportButton.setEnabled(state)

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
    # kell ez a bonyolult verzió?
    # peak_layers = [layer.name for layer in viewer.layers if 'peaks' in layer.name]
    # remaining_wells = [layer.name for layer in viewer.layers if len(layer.name.split()) == 1]
    # if len(peak_layers) == 0:
    #     return remaining_wells
    # remaining_wells = [well for well in remaining_wells if any(peak.startswith(well + " peaks") for peak in peak_layers)]

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