# Plugin Widgets

This plugin provides four widgets, one for extracting the SAM embeddings, and three widgets for segmentations using different methods.

## SAM Embeddings Extractor Widgets
SAM embeddings will be used for running prediction in two other widgets: **SAM-RF Widget**, and **SAM Prompt Segmentation Widget**.  
Therefore, we provided a widget to extract embeddings of the loaded input stack, and to save them on disk in the *HDF5* format.  

![extractor widget](images/extractor_widget.png)

1. **Input layer combo box:** to select input image (stack)
2. **Set storage:** to select where the storage should be saved.
3. **Extract Embeddings button:** to start extracting SAM embeddings
4. **Stop button:** to stop the extraction process.
5. **Process progress bar:** showing extraction progress.
<br><br>

## SAM Predictor Widget
This widget just simply uses the user prompts, and give them to SAM predictor model to get the segmentations. Prompts can be in form of points (only positive), or boxes.  
This widget works nicely for objects having more clear boundaries. It also can be used as a complementary method over the **SAM-RF Widget**.

![sam predictor widget](images/sam_predictor_widget.png)

1. **Input layer combo box:** to select input image (stack).
2. **Prompt layer combo box:** to select the input prompt layer.
3. **Add Point Layer button:** to add a new point prompt layer.
4. **Add Box Layer button:** to add a new box prompt layer (basically it's a napari shapes layer).
5. **New Layer checkbox:** if checked, the segmentations result will be added into a new layer.
6. **Segmentation layer combo box:** if *New Layer* is unchecked, then user must select the segmentations layer.
7. **Segmentation layer options:**
    - **Add Segmentations:** the result will be added to the selected layer. In other words, pixels which segmented as non-background will be added to the selected layer.
    - **Replace Segmentations:** the selected layer content will be replaced with the result.
8. **Predict Prompts button:** to do the prediction using SAM's predictor.
<br><br>

## SAM-RF Widget
This widget is designed to do segmentation while using the SAM embeddings features instead of the image features, along with the user-provided sparse labels using a *Random Forest* model.  
The amount of required labels for having a almost nice-looking segmentations compared to the number of pixels are super low.  
The provided *postprocessing* methods can create an even more accurate and cleaner annotations.

![sam-rf widget 1](images/sam_rf_widget_1.png)

1. **Input layer combo box:** to select input image (stack).
2. **Embedding Storage Select button:** to select the embedding storage file.
3. **Ground Truth Layer:** to select or add a new ground truth layer (napari Labels layer).
4. **Analyze button:** to check number of user-provided labels for each class.
5. **Random Forest Number of trees:** to set number of trees for the RF model.
6. **Random Forest Max depth:** to set maximum depth for each tree in the RF model. pass 0 to set it as *unlimited*.
7. **TrainRF Model button:** to start training of the RF model.
8. **Load Model button:** to load an already trained RF model.
9. **Save Model button:** to save the trained RF model.
<br>

![sam-rf widget 1](images/sam_rf_widget_2.png)

10. **New Layer checkbox:** if checked, the segmentations result will be added into a new layer.
11. **Segmentation layer combo box:** if *New Layer* is unchecked, then user must select the segmentations layer.
12. **Segmentation layer options:**
    - **Add Segmentations:** the result will be added to the selected layer. In other words, pixels which segmented as non-background will be added to the selected layer.
    - **Replace Segmentations:** the selected layer content will be replaced with the result.
13. **Postprocess Segmentation checkbox:** if checked, the segmentations result will be postprocessed.
14. **Area Threshold textbox:** if postprocess checked, then the area threshold will be used to eliminate small segmented objects with area below the set threshold. The higher the area threshold, the more segmented objects will be eliminated.
15. **Use SAM Predictor checkbox:** to use *SAM predictor* model to predict final segmentations using the RF model prediction as input prompts (prompts will be bounding boxes around RF segmented objects).
16. **Predict Slice button:** to run the prediction for the current image slice.
17. **Predict Whole Stack button:** to run prediction for the whole loaded stack.
18. **Stop button:** to stop the prediction process (whole stack prediction migth take a long time).
19. **Prediction Progressbar:** to show the prediction progress.
