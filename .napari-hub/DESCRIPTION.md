# Description

This plugin provides a widget which can load, preprocess, annotate and export cardio biosensor data.  

# Installation

Napari needs to be set up on your machine in order to install and use this plugin. If you do not have napari installed it can be done following this [article](https://napari.org/stable/tutorials/fundamentals/installation.html).

Then the plugin can be installed via the plugin manager or pip.

### Napari plugin manager

Search for `napari-cardio-bio-eval` and click install. 
After completed napari needs to be restarted to activate the plugin.

kép

### Pip package manager

You can install the plugin in the environment where napari is set up with pip.
```
pip install napari-cardio-bio-eval
```
If you have a conda environment use anaconda prompt.

# Usage

You can open the plugin's widget from the **Plugins** menu after the installation of the plugin.

kép

## Data loading and preprocessing

At the top of the widget, you need to select the directory, which contains the data you want to examine and process. To successfully load the data the directory have to contain the following files:
- *_wl_power.file: which contains the starting values of the measurement
- DRM directory: which contains the difference from the previous measurement point 
- *_avg.file: which contains additional biosensor data

#### Import parameters:  
- Flipping: horizontal and vertical mirroring of the biosensor recording
- Signal range type: measurement phase, individual point
- Ranges???
- Drift correction threshold: 25-500
- Filter method: mean or median

Beside the source directory you can set the import parameters and then click the ***Load and Preprocess Data*** button. After a few seconds the well images will appear on the viewer.  

Each well has its own layer. You can turn the layers visible or invisible by clicking  on the small eye icon next to each layer. If you do not need any of the wells then you can delete its layer and it won't appear in the next steps and also won't be exported.

After you see the wells you can proceed to the next step or if the background correction is not good enough you can click the ***Select Background Points Manually*** button and it will show the automatically selected background points for each well, which you can move to real background coordinates and in the next step, in another background correction step, these points will be used by the algorithm. After the first export these points will be saved so if the same directory is loaded a second time the preprocessing will use these points.

## Selecting the cells

In this step you can also set some parameters for the peak detection algorithm and then click the ***Detect Signal Peaks*** button to start the process. After a few seconds the wells with the potential cells will show on the window.

#### Detection parameters:  
- Threshold range: 25-5000
- Neighbourhood size: 1-10
- Error mask filtering:

Here you can delete, add or move the points on each points layer. There are keyboard shortcuts for easier use!

Additionally by double clicking on any point of the image you can examine the time-signal diagram of the selected point under the widget. Be sure to select the correct layer!

After selecting the needed cells and wells (and deleting the unnecessary ones) you can export plots and additional values about them.

## Exporting

You can select what kind of data do you want to export and click the ***Export Data*** button. The data will be exported to the source directory into a *result* sub-directory.

#### Export options:
- Coordinates: the coordinates of the selected cells
- Preprocessed signals: 
- Raw signals:
- Average signal: 
- Breakdown signal: 
- Max well: 
- Plot signals with well:  
- Plot well with coordinates: 
- Plot cells individually: 
- Signal parts by phases: 
- Max centered signals: 
