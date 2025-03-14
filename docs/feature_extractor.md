After selecting your image stack, you need to extract the *features*. Later, these image features will be used as inputs for training a Random Forest model, 
and predicting annotation masks.  

!!! info
    In deep learning, the output of an Encoder model is called <i>embeddings</i> or <i>features</i>.

You can bring up the *Feature Extractor widget* from the napari **Plugins** menu:  

![plugins menu](assets/plugins_menu.png){width="360"}

## Widget Tools Description
![Feature Extractor](assets/extractor_widget/extractor.png){width="360"}

1. **Image Layer**: To select your current image stack.
2. **Encoder Model**: Sets which model you want to use for feature extraction.  
    The **FF** plugins, by default, comes with `MobileSAM`, `SAM (huge)`, `μSAM_LM (base)`, `μSAM_EM_Organelles (base)`, `DINOv2`, `SAM2 (large)`, and `SAM2 (base)` models. It is also possible to introduce a new model by adding a new [*model adapter*](./model_adapter.md) class.
3. **Features Storage File**: Where you want to save the features as an `HDF5` file.
4. **Extract Features** button: Will run the feature extraction process.
5. **Stop** button: To stop the extraction process!

The extraction process might take some time based on number of image slices and the image resolution. This is due to the fact that in **FF** we turn an image into overlapping patches, then pass those patches to the encoder model to get the features. Why we do this? We need to aquire a feature vector per each pixel and not for the whole image. 

## Model Selection
Our experiments tell us usually the `SAM2 (large)` model works the best. However, for less complicated images, using `MobileSAM` or `DINOv2` might also result in a good segmentation as they are lighter and faster.  

!!! note
    When you use a model for the first time, the model's weight will be downloaded from their repository.  So, you might hit a little delay at the first use of model.  

Once you have your image features extracted, you can use the [**Segmentation**](./segmentation.md) widget to generate your image masks.
