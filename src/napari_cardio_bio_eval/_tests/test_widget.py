import numpy as np

from napari_cardio_bio_eval._widget import (
    CardioBioEvalWidget,
)


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_CardioBioEvalWidget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)), name='Image')

    my_widget = CardioBioEvalWidget(viewer)

    assert viewer.layers[0].name == 'Image'

