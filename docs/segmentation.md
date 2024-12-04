Hurray! Now you have your features extracted and ready for the main action! 😊  
The Segmentation widget is a long widget with several panels, but don't worry we'll go through all of them, from top to bottom!  

## Inputs and Labels' statistics
![Inputs](assets/segmentation_widget/seg_1.png){width="360" align=right}
### Inputs
1. **Input Layer**: To set which napari layer is your input image layer
2. **Feature Storage**: Select your previously extracted features `HDF5` file here.  
    ***Note***: You need to select the storage file for this particular input image, obviously!
3. **Ground Truth Layer**: To select your *Labels* layer
4. **Add Layer** button: To add a new GT layer to napari layers

### Labeling Statistics
5. **Analyze** button: To get info about number of classes and labels you've added so far.

!!! note
    - You can have as many *Labels* layer as you want. But **only the selected** one will be used for training the RF model.  
    - You can also drag & drop your previously saved labels into the napari and select that layer.

## Train Model
![Inputs](assets/segmentation_widget/seg_2.png){width="360" align=left}
