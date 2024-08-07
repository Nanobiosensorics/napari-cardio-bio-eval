import os
import napari
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar)

from napari_cardio_bio_eval.widget_utils import *
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
        self.background_selector = False
        self.docked_plot = None
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
        
        # Manual Background selection
        manBGsel = QLabel('Manual background selection if needed:')
        manBGsel.setStyleSheet("QLabel { font-size: 10pt; font-weight: bold; }")
        self.layout.addRow(manBGsel)
        self.backgroundSelectorButton = QPushButton('Select Background Points Manually', self)
        self.backgroundSelectorButton.clicked.connect(self.bg_selection)
        self.backgroundSelectorButton.setEnabled(False)
        self.layout.addRow(self.backgroundSelectorButton)

        # Segmentation
        self.modelPath = QLineEdit(self)
        self.browseModelLayout = QHBoxLayout()
        self.browseModelButton = QPushButton('Browse', self)
        self.browseModelButton.clicked.connect(self.select_model_dialog)
        self.browseModelLayout.addWidget(self.modelPath)
        self.browseModelLayout.addWidget(self.browseModelButton)
        self.layout.addRow(QLabel('Select segmentation model:'), self.browseModelLayout)

        self.showCellCenters = QCheckBox('Show cell centers', self)
        self.layout.addRow(self.showCellCenters)

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

    def load_data(self):
        self.preprocessing_params = update_preprocessing_params(self)
        loader = load_data_thread(self)
        loader.start()

    def preprocess_data(self):
        preprocessor = preprocess_data_thread(self)
        preprocessor.finished.connect(self.show_wells_GUI)
        preprocessor.start()

    def show_wells_GUI(self):
        load_and_preprocess_data_GUI(self)
        self.segmentationButton.setEnabled(True)

    def bg_selection(self):
        manual_background_selection(self)

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
        if self.showCellCenters.isChecked():
            self.cell_centers = get_cell_centers(self.bin_output)

        # print(self.scaling_factor, self.image_size)

        if self.scaling_factor == 1:
            self.GUI_UNet()
        else: 
            self.GUI_SRUNet()

    def GUI_SRUNet(self):
        clear_layers(self.viewer)
        for i in range(len(WELL_NAMES)):
            visible = (i == 0)
            name = WELL_NAMES[i]            
            well_tensor = torch.tensor(self.well_data[name][-1]).unsqueeze(0).unsqueeze(0)
            upscaled_well = torch.nn.functional.interpolate(well_tensor, size=(self.image_size, self.image_size), mode='nearest').squeeze(0).squeeze(0).numpy()
            self.viewer.add_image(upscaled_well, name=name, colormap='viridis', visible=visible)
            self.viewer.add_labels(self.bin_output[i], name=name + self._segment, visible=visible)
            if self.showCellCenters.isChecked():
                self.viewer.add_points(invert_coords(self.cell_centers[i]), name=name + ' cell centers', size=self.scaling_factor/2, face_color='red', visible=visible)
        self.GUI_plot()

    def GUI_UNet(self):
        clear_layers(self.viewer)
        for i in range(len(WELL_NAMES)):
            visible = (i == 0)
            name = WELL_NAMES[i]
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)
            self.viewer.add_labels(self.bin_output[i], name=name + self._segment, visible=visible)
            if self.showCellCenters.isChecked():
                self.viewer.add_points(invert_coords(self.cell_centers[i]), name=name + ' cell centers', size=1, face_color='red', visible=visible)
        self.GUI_plot()

    def GUI_plot(self):
        if self.docked_plot is not None:
            self.viewer.window.remove_dock_widget(widget=self.docked_plot)
        
        current_line = get_cell_line_by_coords(self.well_data[WELL_NAMES[0]], 0, 0, self.phases)

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

                    current_line = get_cell_line_by_coords(self.well_data[name], x, y, self.phases)
                    (line,) = ax.plot(self.time, current_line)
                    ax.set_title(f"Well: {name}, Cell: [{x} {y}]")
                    line.figure.canvas.draw()
                except IndexError:
                    pass
        
        # Once the peak detection is started new data cant be loaded
        self.exportButton.setEnabled(True)
        self.loadButton.setEnabled(False)
        self.processButton.setEnabled(False)
        self.segmentationButton.setEnabled(True)
        self.segmentationButton.setText("Segment")

    def export_data(self):
        self.progressBar.setMaximum(0)
        self.remaining_wells = get_remaining_wells_from_layers(self.viewer)

        segments = {}
        for name in self.remaining_wells:
            if self.showCellCenters.isChecked():
                segments[name] = (self.viewer.layers[name + self._segment].data, self.viewer.layers[name + ' cell centers'].data)
            else:
                segments[name] = self.viewer.layers[name + self._segment].data

        # segments['size'] = self.image_size # kell? mert shapeből úgy is kiolvasható

        np.savez(os.path.join(self.RESULT_PATH, 'well_segments'), **segments)

        # # Later on, load from disk
        # segments = np.load('well_segments.npz')
        # well_segments = {key: segments[key] for key in segments.files}
        # for key in segments.files:
        #     print(key, segments[key].shape)
        
        self.progressBar.setMaximum(1)

    



def get_cell_centers(output):
    cell_centers = []
    for i in range(len(WELL_NAMES)):
        pred = output[i].squeeze().astype(np.uint8)
        _, _, _, centers = cv2.connectedComponentsWithStats(pred, connectivity=8)
        cell_centers.append(centers[1:])
    return cell_centers

def lin_indices(original_length, subsampled_length):
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]
