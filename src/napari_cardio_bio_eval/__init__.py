__version__ = "0.0.1"

import napari
from napari_cardio_bio_eval.peak_detection_widget import PeakDetectionWidget
from napari_cardio_bio_eval.segmentation_widget import SegmentationWidget

__all__ = (
    "PeakDetectionWidget",
    "SegmentationWidget",
)

# viewer = napari.Viewer()
# widget = PeakDetectionWidget(viewer)
# widget = SegmentationWidget(viewer)
# viewer.window.add_dock_widget(widget, area="right")
# widget.dirLineEdit.setText("C:/Users/wittd/Desktop/epic_data")

# napari.run()