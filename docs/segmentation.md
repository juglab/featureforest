Hurray! Now you have your features extracted and ready for the main action! 😊  
The Segmentation widget is a long widget with several panels, but don't worry we'll go through all of them, from top to bottom!  

## Inputs and Labels' statistics
![Inputs](assets/segmentation_widget/seg_1.png){width="360" align=right}
### Inputs
1. **Input Layer**: To set which napari layer is your input image layer
2. **Feature Storage**: Select your previously extracted features `HDF5` file here.  
    ***Note***: You need to select the storage file for the selected input image, obviously!
3. **Ground Truth Layer**: To select your *Labels* layer
4. **Add Layer** button: To add a new GT layer to napari layers

### Labeling Statistics
5. **Analyze** button: To get info about number of classes and labels you've added so far.

!!! note
    - You can have as many *Labels* layer as you want. But **only the selected** one will be used for training the RF model.  
    - You can also drag & drop your previously saved labels into the napari and select that layer.


## Train Model
![Inputs](assets/segmentation_widget/seg_2.png){width="360" align=right}
### Train Model (Random Forest)
1. **Number of Trees**: To set number of trees (estimators) in the forest
2. **Max depth**: The maximum depth of a tree
3. **Train** button: To extract the training data and train the **RF** model
4. **Load Model** button: Using this, you can load a previously trained and saved model.
5. **Save Model** button: To save the current RF model

!!! tip
    - Setting a high value for the `Max depth` would overfit your **RF** model over the training data. So, it won't perform well on test images.
    But if you're doing the segmentation over the entire stack (or a single image), you may try higher values.

<div class="clear"></div>

## Prediction
![Inputs](assets/segmentation_widget/seg_3.png){width="360" align=right}
### Prediction
###### Segmentation Layer:
1. **New Layer**: If checked, the segmentation result will show up on a new layer in napari
2. **Layer Dropdown**: You can select which layer should be used as the layer for the segmentation result
3. **Add/Replace Segmentation** option: Based on your choice, this will add new segmentation to the previous result, or completely replace the result (Default).
###### Buttons:
4. **Predict Slice** button: To generate the segmentation mask for the *current* slice
5. **Predict Whole Stack** button: to start the prediction process for the whole loaded stack
6. **Stop** button: Just for your safety!😉 this will stop the prediction process.


## Post-processing
![Inputs](assets/segmentation_widget/seg_4.png){width="360"}

-


## Export
![Inputs](assets/segmentation_widget/seg_5.png){width="360"}

-


## Run Prediction Pipeline
![Inputs](assets/segmentation_widget/seg_6.png){width="360"}

-

