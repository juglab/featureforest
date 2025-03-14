*Feature Forest* (***FF***) uses vision foundation models for the feature extraction. 
`MobileSAM`, `SAM (huge)`, `μSAM_LM (base)`, `μSAM_EM_Organelles (base)`, `DINOv2`, `SAM2 (large)`, and `SAM2 (base)` models are already included into the plugin.

!!! tip
    Based on our experiences, we found `SAM2` works better, especially for complex images. 
    If you have less complex stack to segment, you might get a decent result using `DINOv2` or even `MobileSAM` while they are lighter models.


### Adapting a new Model
New and powerful deep Learning models appear quite often, and we always want to try them over our data and tasks.  In **FF**, we provided an `BaseModelAdapter` class to adapt any model of interest.  
For adding a new model, you need to subclass the `BaseModelAdapter`, and set a few attributes and implement a few methods especially the `get_features_patches` method which does the feature extraction. The input for the model will be provided as image patches, but this will be handled by the plugin, so you can think of it as a batch of images.  
You also need to take care of input/output transformation by setting the right sizes for the `self.input_transforms` to make the right size input for the model, and the `self.embedding_transform` for resizing the model output back to the patch size.  
You can check the code for the *`SAM2` model adapter* [here](https://github.com/juglab/featureforest/blob/main/src/featureforest/models/SAM2/adapter.py).  
