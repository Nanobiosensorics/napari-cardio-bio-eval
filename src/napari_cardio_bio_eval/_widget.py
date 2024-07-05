import os
import napari
import matplotlib.pyplot as plt
import numpy as np

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar, QFrame)

from nanobio_core.epic_cardio.processing import RangeType, load_data, load_params, preprocessing, localization, save_params
from nanobio_core.epic_cardio.defs import WELL_NAMES
from nanobio_core.kiertekelo.export import export_results

from napari.qt.threading import thread_worker
from matplotlib.backends.backend_qt5agg import FigureCanvas


class CardioBioEvalWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
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
        self.browseButton.clicked.connect(self.openFileNameDialog)
        self.browseBox.addWidget(self.dirLineEdit)
        self.browseBox.addWidget(self.browseButton)
        self.layout.addRow(QLabel('Select Directory:'), self.browseBox)

        # Parameter inputs
        self.horizontalFlip = QCheckBox('Horizontal Flip', self)
        self.layout.addRow(self.horizontalFlip)
        self.verticalFlip = QCheckBox('Vertical Flip', self)
        self.layout.addRow(self.verticalFlip)

        self.layout.addRow(QLabel('Drift correction:'))
        self.threshold = QSpinBox(self)
        self.threshold.setMinimum(0)
        self.threshold.setMaximum(100)
        self.threshold.setValue(75)
        self.layout.addRow(QLabel('Threshold:'), self.threshold)

        self.filterMethod = QComboBox(self)
        self.filterMethod.addItems(['mean', 'median'])
        self.layout.addRow(QLabel('Filter method:'), self.filterMethod)

        # Data loading button
        self.loadButton = QPushButton('Load and Preprocess Data', self)
        self.loadButton.clicked.connect(self.loadAndPreprocessData)
        self.layout.addRow(self.loadButton)

        # Manual Background selection
        manBGsel = QLabel('Manual background selection if needed:')
        manBGsel.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; }")
        self.layout.addRow(manBGsel)
        self.backgroundSelectorButton = QPushButton('Select Background Points Manually', self)
        self.backgroundSelectorButton.clicked.connect(self.manualBackgroundSelection)
        self.layout.addRow(self.backgroundSelectorButton)

        # Peak detection parameters
        peakDetLabel = QLabel('Peak detection parameters:')
        peakDetLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(peakDetLabel)

        self.neighbourhood_size = QSpinBox(self)
        self.neighbourhood_size.setMinimum(0)
        self.neighbourhood_size.setMaximum(10)
        self.neighbourhood_size.setValue(3)
        self.layout.addRow(QLabel('Neighbourhood size:'), self.neighbourhood_size)
        self.errorMaskFiltering = QCheckBox('Error Mask Filtering', self)
        self.errorMaskFiltering.setChecked(True)
        self.layout.addRow(self.errorMaskFiltering)
        # Peak detection button
        self.peakButton = QPushButton('Detect Signal Peaks', self)
        self.peakButton.clicked.connect(self.peakDetection)
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
        self.exportButton.clicked.connect(self.exportData)
        self.layout.addRow(self.exportButton)
        # Export progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(1)
        self.layout.addRow(self.progressBar)

    def openFileNameDialog(self):
        # TODO check the selected directory for the needed files if neccecary (in data_load function it does)
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dirLineEdit.setText(directory)

    def loadAndPreprocessData(self):
        self.preprocessing_params = {
            'flip': [self.horizontalFlip.isChecked(), self.verticalFlip.isChecked()],
            'signal_range' : {
            'range_type': RangeType.MEASUREMENT_PHASE,
            'ranges': [0, None],
            },
            'drift_correction': {
            'threshold': self.threshold.value(),
            'filter_method': self.filterMethod.currentText(),
            'background_selector': False,
            }
        }

        loader = self.loadAndPreprocessData_thread()
        loader.finished.connect(self.loadAndPreprocessData_GUI)
        loader.start()

    # Manual Background selection here if the automatic is bad
    def manualBackgroundSelection(self):
        self.preprocessing_params['drift_correction']['background_selector'] = True

        self.clear_layers()

        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)
            self.viewer.add_points(self.invert_coords(self.filter_ptss[name]), name=name + ' bg points', size=1, face_color='orange', visible=visible)


    def peakDetection(self):
        self.localization_params = {
            'threshold_range' : [.075*1000, 3*1000],
            'neighbourhood_size': self.neighbourhood_size.value(),
            'error_mask_filtering': self.errorMaskFiltering.isChecked()
        }

        peak_detector = self.peakDetection_thread()
        peak_detector.finished.connect(self.peakDetection_GUI)
        peak_detector.start()

    def exportData(self):
        self.export_params = {
            'coordinates': self.coordinates.isChecked(),
            'preprocessed_signals': self.preprocessedSignals.isChecked(),
            'raw_signals': self.rawSignals.isChecked(),
            'average_signal': self.averageSignal.isChecked(),
            'breakdown_signal': self.breakdownSignal.isChecked(),
            'max_well': self.maxWell.isChecked(),
            'plot_signals_with_well': self.plotSignalsWithWell.isChecked(),
            'plot_well_with_coordinates': self.plotWellWithCoordinates.isChecked(),
            'plot_cells_individually': self.plotCellsIndividually.isChecked(),
            'signal_parts_by_phases': self.signalPartsByPhases.isChecked(),
            'max_centered_signals': self.maxCenteredSignals.isChecked()
        }
        self.progressBar.setMaximum(0)
        self.remaining_wells = self.remaining_wells_from_layers()
        self.selected_ptss = self.get_selected_points()

        for name in self.remaining_wells:
            self.well_data[name] = (self.viewer.layers[name].data, self.selected_ptss[name], self.well_data[name][-1])

        save_params(self.RESULT_PATH, self.well_data, self.preprocessing_params, self.localization_params)

        exporter = self.export_res()
        exporter.finished.connect(lambda: self.progressBar.setMaximum(1))
        # exporter.finished.connect(lambda: self.viewer.notifications.show('Export finished!'))
        exporter.start()

    def clear_layers(self):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

    def invert_coords(self, coords):
        return np.array([[y, x] for x, y in coords])

    def set_buttons_enabled(self, state):
        self.loadButton.setEnabled(state)
        self.backgroundSelectorButton.setEnabled(state)
        self.peakButton.setEnabled(state)
        self.exportButton.setEnabled(state)

    def get_filter_points(self):
        filter_ptss = {}
        for name in WELL_NAMES:
            filter_ptss[name] = self.invert_coords(np.round(self.viewer.layers[name + ' bg points'].data)).astype(np.uint8)
        return filter_ptss

    def get_selected_points(self):
        selected_ptss = {}
        for name in self.remaining_wells:
            selected_ptss[name] = self.invert_coords(np.round(self.viewer.layers[name + ' peaks'].data)).astype(np.uint8)
        return selected_ptss

    def remaining_wells_from_layers(self):
        remaining_wells = []
        for layer in self.viewer.layers:
            # if 'peaks' not in layer.name or 'bg' not in layer.name:
            if  len(layer.name.split()) == 1:
                remaining_wells.append(layer.name)
        return remaining_wells        

    @thread_worker
    def loadAndPreprocessData_thread(self):
        self.set_buttons_enabled(False)
        self.loadButton.setEnabled(True)
 
        path = self.dirLineEdit.text()
        self.RESULT_PATH = os.path.join(path, 'result')
        if not os.path.exists(self.RESULT_PATH):
            os.mkdir(self.RESULT_PATH)

        self.raw_wells, self.full_time, self.full_phases = load_data(path, flip=self.preprocessing_params['flip'])
        self.filter_params, _, _ = load_params(self.RESULT_PATH)
        self.well_data, self.time, self.phases, self.filter_ptss, self.selected_range = preprocessing(self.preprocessing_params, self.raw_wells, self.full_time, self.full_phases, self.filter_params)
        
        self.set_buttons_enabled(True)

    def loadAndPreprocessData_GUI(self):
        self.clear_layers()
        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)

    @thread_worker
    def peakDetection_thread(self):
        self.set_buttons_enabled(False)

        if self.preprocessing_params['drift_correction']['background_selector']:
            self.filter_ptss = self.get_filter_points()

        # Here the well data contains tha wells, the selected points and the filter points (which are the background points)
        self.well_data = localization(self.preprocessing_params, self.localization_params, 
                                    self.raw_wells, self.selected_range, 
                                    self.filter_ptss)

        self.remaining_wells = self.remaining_wells_from_layers()
        self.set_buttons_enabled(True)

    def peakDetection_GUI(self):
        self.clear_layers()
        # visualize the data with peaks
        for name in self.remaining_wells:
            visible = (name == self.remaining_wells[0])
            self.viewer.add_image(self.well_data[name][0], name=name, colormap='viridis', visible=visible)
            # invert the coordinates of the peaks to plot in napari (later invert back for other plots)
            self.viewer.add_points(self.invert_coords(self.well_data[name][1]), name=name + ' peaks', size=1, face_color='red', visible=visible)
            # filter points for background selection
            # self.viewer.add_points(self.invert_coords(self.well_data[name][-1]), name=name + ' filter', size=1, face_color='blue', visible=visible)

        current_line = self.get_cell_line_by_coords(self.remaining_wells[0], 0, 0)

        # create mpl figure with subplots
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)   # 1 row, 1 column, 1st plot
        (line,) = ax.plot(self.time, current_line)
        # add the figure to the viewer as a FigureCanvas widget
        docked_plot = self.viewer.window.add_dock_widget(FigureCanvas(mpl_fig))
        docked_plot.setMinimumSize(200, 300)

        for layer in self.viewer.layers:
            @layer.mouse_double_click_callbacks.append
            def update_plot_on_double_click(layer, event):
                try:
                    name = layer.name.split(' ')[0]
                    ax.clear()
                    current_line = self.get_cell_line_by_coords(name, int(event.position[1]), int(event.position[2]))
                    (line,) = ax.plot(self.time, current_line)
                    ax.set_title(f"Well: {name}, Cell: [{int(event.position[1])} {int(event.position[2])}]")
                    line.figure.canvas.draw()
                except IndexError:
                    pass

    @thread_worker
    def export_res(self):
        export_results(self.export_params, self.RESULT_PATH, self.selected_ptss, self.filter_ptss, #backgroung selectorból
                        self.well_data, self.time, self.phases, self.raw_wells, self.full_time, self.full_phases, self.selected_range)

    def get_cell_line_by_coords(self, well_name, x, y):
        current_line = self.well_data[well_name][0][:, x, y].copy()
        if len(self.phases) > 0:
            for p in self.phases:
                current_line[p] = np.nan
        return current_line
