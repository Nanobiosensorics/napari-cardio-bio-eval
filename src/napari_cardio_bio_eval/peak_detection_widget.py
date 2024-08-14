import napari

from qtpy.QtWidgets import (QWidget, QHBoxLayout, QFormLayout, 
                            QPushButton, QLineEdit, QFileDialog, 
                            QLabel, QSpinBox, QComboBox, QCheckBox, 
                            QProgressBar)

from napari_cardio_bio_eval.widget_utils import *
from nanobio_core.epic_cardio.processing import save_params


class PeakDetectionWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self._peaks = " peaks"
        self._bg_points = " background points"
        self.docked_plot = None
        self.background_selector = False
        self.cell_selector = False
        self.full_phases = []
        self.scaling_factor = 1
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

    # I think it needs to be a member function because it needs to be given to the rangeTypeSelect and can't be None
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

    def select_data_dir_dialog(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if directory:
            self.dirLineEdit.setText(directory)

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
        self.peakButton.setEnabled(True)

    def bg_selection(self):
        manual_background_selection(self)

    def peak_detection(self):
        self.localization_params = update_localization_params(self)

        self.cell_selector = True
        peak_detector = peak_detection_thread(self)
        peak_detector.finished.connect(self.show_peaks_GUI)
        peak_detector.start()

    def show_peaks_GUI(self):
        peak_detection_GUI(self)

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

        exporter = export_results_thread(self)
        exporter.finished.connect(lambda: self.progressBar.setMaximum(1))
        exporter.start()
