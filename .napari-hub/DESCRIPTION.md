# Description

This plugin provides a widget which can load, preprocess, annotate and export cardio biosensor data.  


## Data loading and preprocessing

At the top of the widget you can select the directory which contains the data you want to examine and process. To successfully load the data the directory needs to contain:
- *_wl_power.file: which contains the starting values of the measurement
- DRM directory: which contains the difference from the previous measurement point 
- *_avg.file: which contains additional biosensor data

Beside the source directory you can set the import parameters and then click the *Load and Preprocess Data* button. After a few seconds the well images will appear on the viewer.  
Each well has its own layer. You can turn the layers visible or invisible by clicing on the small eye icon next to each layer. If you don not need any of the wells then you can delete it's layer and it won't appear in the next steps and won't be in the export.

After you see the wells you can proceed to the next step or if the background correction is not good enough you can click the *Manual Background Selection* button and it will show the automatically selected background points for each well, which you can move to real background coordinates and in the next step, in another background correction step, these points will be used by the algorithm.

## Selecting the cells

In this step you can also set some parameters for the peak detection algorithm and then click the *Peak Detection* button to start the process. After a few seconds the wells with the potential cells will show on the window.

Here you can delete, add or move the points on each point layer and by double clicking on any point of the image you can examine the time-signal diagramm of the selected point under the widget. Be sure to select the correct layer!

After selecting the needed cells and wells (and deleting the unnecesery ones) you can export plots and additional values about them.

## Exporting

You can select what kind of data do you want to export and click the *Export Data* button. The data will be exported to the source directory into a *result* folder.

