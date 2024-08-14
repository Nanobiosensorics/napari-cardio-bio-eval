import os
import napari
import numpy as np

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar)

from napari_cardio_bio_eval.widget_utils import *
from nanobio_core.epic_cardio.processing import RangeType, load_data, load_params, preprocessing, localization, save_params
from nanobio_core.epic_cardio.defs import WELL_NAMES
from nanobio_core.epic_cardio.data_correction import correct_well
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
        segmentLabel = QLabel('Segmentation:')
        segmentLabel.setStyleSheet("QLabel { font-size: 11pt; font-weight: bold; }")
        self.layout.addRow(segmentLabel)
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

    # ez után is megint draw error van
    def bg_selection(self):
        self.background_selector = True
        manual_background_selection(self)

    def segmentation(self):
        self.segmentationButton.setEnabled(False)
        self.segmentationButton.setText("Segmenting...")

        if self.background_selector:
            self.filter_ptss = get_filter_points(self.viewer)
            self.background_selector = False

            slicer = slice(self.selected_range[0], self.selected_range[1])
            for name in WELL_NAMES:
                well_tmp = self.raw_wells[name]
                well_tmp = well_tmp[slicer]
                well_corr, _, _ = correct_well(well_tmp, 
                                                    coords= self.filter_ptss[name],
                                                    threshold=self.preprocessing_params['drift_correction']['threshold'],
                                                    mode=self.preprocessing_params['drift_correction']['filter_method'])
                
                self.well_data[name] = well_corr

        segmentation = segmentation_thread(self)
        segmentation.finished.connect(self.segmentation_finished)
        segmentation.start()

    def segmentation_finished(self):
        if self.scaling_factor == 1:
            UNet_segmentation_GUI(self)
        else: 
            SRUNet_segmentation_GUI(self)

    def export_data(self):
        self.progressBar.setMaximum(0)
        self.remaining_wells = get_remaining_wells_from_layers(self.viewer)

        segments = {}
        for name in self.remaining_wells:
            segments[name] = self.viewer.layers[name + self._segment].data

        # segments['size'] = self.image_size # kell? mert shapeből úgy is kiolvasható

        np.savez(os.path.join(self.RESULT_PATH, 'well_segments'), **segments)

        # # Later on, load from disk
        # segments = np.load('well_segments.npz')
        # well_segments = {key: segments[key] for key in segments.files}
        # for key in segments.files:
        #     print(key, segments[key].shape)
        
        self.progressBar.setMaximum(1)
        print("Export finished")
