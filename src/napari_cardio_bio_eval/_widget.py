"""
This module contains four napari widgets declared in
different ways:

- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QWidget,QHBoxLayout, QFormLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QSpinBox, QComboBox, QCheckBox
import os
from magicgui.widgets import CheckBox, Container, create_widget
from skimage.util import img_as_float
from nanobio_core.epic_cardio.processing import *
from nanobio_core.epic_cardio.defs import WELL_NAMES
from nanobio_core.kiertekelo.export import export_results
import napari
from napari.qt.threading import thread_worker



# if we want even more control over our widget, we can use
# magicgui `Container`
class ImageThreshold(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        # use create_widget to generate widgets from type annotations
        self._image_layer_combo = create_widget(
            label="Image", annotation="napari.layers.Image"
        )
        self._threshold_slider = create_widget(
            label="Threshold", annotation=float, widget_type="FloatSlider"
        )
        self._threshold_slider.min = 0
        self._threshold_slider.max = 1
        # use magicgui widgets directly
        self._invert_checkbox = CheckBox(text="Keep pixels below threshold")

        # connect your own callbacks
        self._threshold_slider.changed.connect(self._threshold_im)
        self._invert_checkbox.changed.connect(self._threshold_im)

        # append into/extend the container with your widgets
        self.extend(
            [
                self._image_layer_combo,
                self._threshold_slider,
                self._invert_checkbox,
            ]
        )

    def _threshold_im(self):
        image_layer = self._image_layer_combo.value
        if image_layer is None:
            return

        image = img_as_float(image_layer.data)
        name = image_layer.name + "_thresholded"
        threshold = self._threshold_slider.value
        if self._invert_checkbox.value:
            thresholded = image < threshold
        else:
            thresholded = image > threshold
        if name in self._viewer.layers:
            self._viewer.layers[name].data = thresholded
        else:
            self._viewer.add_labels(thresholded, name=name)

class CardioBioEvalWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.initUI()

    def initUI(self):
        self.layout = QFormLayout(self)

        # Directory selection
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

        self.backgroundSelector = QCheckBox(self)
        self.layout.addRow(QLabel('Background selector:'), self.backgroundSelector)

        # Data loading button
        self.loadButton = QPushButton('Load and Preprocess Data', self)
        self.loadButton.clicked.connect(self.loadAndPreprocessData)
        self.layout.addRow(self.loadButton)

        # Peak detection parameters
        self.neighbourhood_size = QSpinBox(self)
        self.neighbourhood_size.setMinimum(0)
        self.neighbourhood_size.setMaximum(10)
        self.neighbourhood_size.setValue(3)
        self.layout.addRow(QLabel('Neighbourhood size:'), self.neighbourhood_size)
        self.errorMaskFiltering = QCheckBox('Error Mask Filtering', self)
        self.errorMaskFiltering.setChecked(True)
        self.layout.addRow(self.errorMaskFiltering)
        # Peak detection button
        self.peakButton = QPushButton('Peak Detection', self)
        self.peakButton.clicked.connect(self.peakDetection)
        self.layout.addRow(self.peakButton)

        # Export parameters
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

        self.exportButton = QPushButton('Export Data', self)
        self.exportButton.clicked.connect(self.exportData)
        self.layout.addRow(self.exportButton)


    def openFileNameDialog(self):
        # TODO check the selected directory for the needed files
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dirLineEdit.setText(directory)

    def loadAndPreprocessData(self):
        path = self.dirLineEdit.text()
        self.preprocessing_params = {
            'flip': [self.horizontalFlip.isChecked(), self.verticalFlip.isChecked()],
            'signal_range' : {
            'range_type': RangeType.MEASUREMENT_PHASE,
            'ranges': [0, None],
            },
            'drift_correction': {
            'threshold': self.threshold.value(),
            'filter_method': self.filterMethod.currentText(),
            'background_selector': self.backgroundSelector.isChecked(), #manual background selection is not implemented yet
            }
        }

        self.RESULT_PATH = os.path.join(path, 'result')
        if not os.path.exists(self.RESULT_PATH):
            os.mkdir(self.RESULT_PATH)

        self.raw_wells, self.full_time, self.full_phases = load_data(path, flip=self.preprocessing_params['flip'])
        self.filter_params, _, _ = load_params(self.RESULT_PATH)
        self.well_data, self.time, self.phases, self.filter_ptss, self.selected_range = preprocessing(self.preprocessing_params, self.raw_wells, self.full_time, self.full_phases, self.filter_params)

        for name in WELL_NAMES:
            visible = (name == WELL_NAMES[0])
            self.viewer.add_image(self.well_data[name], name=name, colormap='viridis', visible=visible)

    def peakDetection(self):
        self.localization_params = {
            'threshold_range' : [.075*1000, 3*1000],
            'neighbourhood_size': self.neighbourhood_size.value(),
            'error_mask_filtering': self.errorMaskFiltering.isChecked()
        }

        # if self.preprocessing_params['drift_correction']['background_selector']:
        #     background_selector = WellArrayBackgroundSelector(well_data, filter_params, False)

        # if self.preprocessing_params['drift_correction']['background_selector']:
        #     filter_ptss = background_selector.selected_coords

        self.well_data = localization(self.preprocessing_params, self.localization_params, 
                                self.raw_wells, self.selected_range, 
                                {} if not self.preprocessing_params['drift_correction']['background_selector'] else background_selector.selected_coords)

        self.remaining_wells = self.remaining_wells_from_layers()

        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        
        # visualize the data with peaks
        for name in self.remaining_wells:
            visible = (name == self.remaining_wells[0])
            self.viewer.add_image(self.well_data[name][0], name=name, colormap='viridis', visible=visible)
            # invert the coordinates of the peaks to plot in napari (later invert back for other plots)
            self.viewer.add_points(self.invert_coords(self.well_data[name][1]), name=name + ' peaks', size=1, face_color='red', visible=visible)
            # filter points for background selection
            # self.viewer.add_points(self.invert_coords(self.well_data[name][-1]), name=name + ' filter', size=1, face_color='blue', visible=visible)
        

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

        self.remaining_wells = self.remaining_wells_from_layers()
        self.selected_ptss = self.selected_points()

        for name in self.remaining_wells:
            self.well_data[name] = (self.viewer.layers[name].data, self.selected_ptss[name], self.well_data[name][-1])

        exporter = self.export_res()
        exporter.finished.connect(lambda: print('Export finished'))
        exporter.start()

        save_params(self.RESULT_PATH, self.well_data, self.preprocessing_params, self.localization_params)
        

    def invert_coords(self, coords):
        return np.array([[y, x] for x, y in coords])

    def selected_points(self):
        selected_ptss = {}
        for name in self.remaining_wells:
            selected_ptss[name] = self.invert_coords(np.round(self.viewer.layers[name + ' peaks'].data)).astype(np.uint8)
        return selected_ptss

    def remaining_wells_from_layers(self):
        remaining_wells = []
        for layer in self.viewer.layers:
            if 'peaks' not in layer.name:
                remaining_wells.append(layer.name)
        return remaining_wells        

    @thread_worker
    def export_res(self):
        export_results(self.export_params, self.RESULT_PATH, self.selected_ptss, self.filter_ptss, #backgroung selectorból
                        self.well_data, self.time, self.phases, self.raw_wells, self.full_time, self.full_phases, self.selected_range)