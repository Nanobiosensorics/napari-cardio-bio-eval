import os
import napari
import matplotlib.pyplot as plt
import numpy as np
import torch

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar)

from nanobio_core.epic_cardio.processing import RangeType, load_data, load_params, preprocessing, localization, save_params
from nanobio_core.epic_cardio.defs import WELL_NAMES
from export_and_plot.export import export_results

from napari.qt.threading import thread_worker
from matplotlib.backends.backend_qt5agg import FigureCanvas


class SegmentationWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._segment = " segment"
        self._bg_points = " background points"
        self.docked_plot = None
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
        self.browseButton.clicked.connect(self.select_data_dir_dialog)
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

        # Segmentation
        self.modelPath = QLineEdit(self)
        self.browseModelLayout = QHBoxLayout()
        self.browseModelButton = QPushButton('Browse', self)
        self.browseModelButton.clicked.connect(self.select_model_dialog)
        self.browseModelLayout.addWidget(self.modelPath)
        self.browseModelLayout.addWidget(self.browseModelButton)
        self.layout.addRow(QLabel('Select segmentation model:'), self.browseModelLayout)

        self.segmentationButton = QPushButton('Segment', self)
        self.segmentationButton.setEnabled(False)
        self.segmentationButton.clicked.connect(self.segmentation)
        self.layout.addRow(self.segmentationButton)

        exportLabel = QLabel('Export:')
        exportLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(exportLabel)
        # Export button
        self.exportButton = QPushButton('Export segments', self)
        self.exportButton.clicked.connect(self.export_data)
        self.exportButton.setEnabled(False)
        self.layout.addRow(self.exportButton)
        # Export progress bar
        self.progressBar = QProgressBar(self)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(1)
        self.layout.addRow(self.progressBar)

    def select_data_dir_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.dirLineEdit.setText(directory)

    def select_model_dialog(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select segmentation model", filter="Model files (*.pth)")
        if model_path:
            self.modelPath.setText(model_path)

    def load_data(self):
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

        self.cell_selector = False
        loader = self.load_data_thread()
        loader.start()

    @thread_worker
    def load_data_thread(self):
        self.set_buttons_enabled(False)
        path = self.dirLineEdit.text()
        self.RESULT_PATH = os.path.join(path, 'result')
        if not os.path.exists(self.RESULT_PATH):
            os.mkdir(self.RESULT_PATH)

        self.raw_wells, self.full_time, self.full_phases = load_data(path, flip=self.preprocessing_params['flip'])
        self.filter_params, _, _ = load_params(self.RESULT_PATH)

        self.rangeLabel.setText(f'Phases: {[(n+1, p) for n, p in enumerate(self.full_phases)]}, Time: {len(self.full_time)}')
        self.rangeTypeSelect.currentIndexChanged.connect(self.range_type_changed)
        # Enable the range selection
        self.rangeTypeSelect.setEnabled(True)
        self.rangeMin.setEnabled(True)
        self.rangeMax.setEnabled(True)
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
        if self.rangeTypeSelect.currentIndex() == 0:
            self.preprocessing_params['signal_range']['range_type'] = RangeType.MEASUREMENT_PHASE
        else:
            self.preprocessing_params['signal_range']['range_type'] = RangeType.INDIVIDUAL_POINT

        # It means that the range is set to the last phase or time point, it would be out of index
        if self.rangeMax.value() == len(self.full_phases)+1:
            self.preprocessing_params['signal_range']['ranges'] = [self.rangeMin.value(), None]
        else:
            self.preprocessing_params['signal_range']['ranges'] = [self.rangeMin.value(), self.rangeMax.value()]

        self.well_data, self.time, self.phases, self.filter_ptss, self.selected_range = preprocessing(self.preprocessing_params, self.raw_wells, self.full_time, self.full_phases, self.filter_params)

    def load_and_preprocess_data_GUI(self):
        self.clear_layers()
        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)

        self.backgroundSelectorButton.setEnabled(True)
        self.segmentationButton.setEnabled(True)

    def segmentation(self):
        self.segmentationButton.setEnabled(False)
        self.segmentationButton.setText("Segmenting...")

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

        # print(self.scaling_factor, self.image_size)

        if self.scaling_factor == 1:
            self.GUI_UNet()
        else: 
            self.GUI_SRUNet()

    def GUI_SRUNet(self):
        self.clear_layers()
        for i in range(len(WELL_NAMES)):
            visible = (i == 0)
            name = WELL_NAMES[i]            
            well_tensor = torch.tensor(self.well_data[name][-1]).unsqueeze(0).unsqueeze(0)
            upscaled_well = torch.nn.functional.interpolate(well_tensor, size=(self.image_size, self.image_size), mode='nearest').squeeze(0).squeeze(0).numpy()
            self.viewer.add_image(upscaled_well, name=name, colormap='viridis', visible=visible)
            self.viewer.add_labels(self.bin_output[i], name=name + self._segment, visible=visible)
        self.GUI_plot()

    def GUI_UNet(self):
        self.clear_layers()
        for i in range(len(WELL_NAMES)):
            visible = (i == 0)
            name = WELL_NAMES[i]
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)
            self.viewer.add_labels(self.bin_output[i], name=name + self._segment, visible=visible)
        self.GUI_plot()

    def GUI_plot(self):
        if self.docked_plot is not None:
            self.viewer.window.remove_dock_widget(widget=self.docked_plot)
        
        current_line = self.get_cell_line_by_coords(WELL_NAMES[-1], 0, 0)

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

                    if self.scaling_factor == 1:
                        x = int(event.position[1])
                        y = int(event.position[2])
                    else:
                        x = int(event.position[0]/self.scaling_factor)
                        y = int(event.position[1]/self.scaling_factor)

                    x = max(0, min(x, 79))
                    y = max(0, min(y, 79))

                    current_line = self.get_cell_line_by_coords(name, x, y)
                    (line,) = ax.plot(self.time, current_line)
                    ax.set_title(f"Well: {name}, Cell: [{x} {y}]")
                    line.figure.canvas.draw()
                except IndexError:
                    pass
        
        # Once the peak detection is started new data cant be loaded
        self.set_buttons_enabled(True)
        self.loadButton.setEnabled(False)
        self.processButton.setEnabled(False)
        self.segmentationButton.setEnabled(True)
        self.segmentationButton.setText("Segment")

    def export_data(self):
        self.progressBar.setMaximum(0)
        self.remaining_wells = self.get_remaining_wells_from_layers()

        segments = {}
        for name in self.remaining_wells:
            segments[name] = self.viewer.layers[name + self._segment].data

        # Save to disk
        with open('well_segments.npz', 'wb') as f:
            np.savez(self.RESULT_PATH, **segments)

        # # Later on, load from disk
        # loaded = np.load('well_segments.npz')

        # # Access data
        # for i in range(1, len(segments)+1):
        #     loaded_segments = loaded['well'+str(i)]

        self.progressBar.setMaximum(1)

    def clear_layers(self):
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

    def set_buttons_enabled(self, state):
        # state is a boolean
        self.backgroundSelectorButton.setEnabled(state)
        self.exportButton.setEnabled(state)

    def get_filter_points(self):
        filter_ptss = {}
        for name in WELL_NAMES:
            filter_ptss[name] = invert_coords(np.round(self.viewer.layers[name + self._bg_points].data)).astype(np.uint8)
        return filter_ptss

    def get_selected_cells(self):
        selected_ptss = {}
        for name in self.remaining_wells:
            selected_ptss[name] = invert_coords(np.round(self.viewer.layers[name + self._peaks].data)).astype(np.uint8)
        return selected_ptss

    def get_remaining_wells_from_layers(self):
        peak_layers = [layer.name for layer in self.viewer.layers if 'segment' in layer.name]
        remaining_wells = [layer.name for layer in self.viewer.layers if len(layer.name.split()) == 1]
        if len(peak_layers) == 0:
            return remaining_wells
        remaining_wells = [well for well in remaining_wells if any(peak.startswith(well + self._peaks) for peak in peak_layers)]
        return remaining_wells

    def get_cell_line_by_coords(self, well_name, x, y):
        # x and y must be between 0 and 80!
        current_line = self.well_data[well_name][:, x, y].copy()
        if len(self.phases) > 0:
            for p in self.phases:
                current_line[p] = np.nan
        return current_line

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

def invert_coords(coords):
        return np.array([[y, x] for x, y in coords])

def lin_indices(original_length, subsampled_length):
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]
