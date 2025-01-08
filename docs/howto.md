

Microscopic images usually come with a large stack: many high-resolution slices!

There are two ways to utilize this plugin over a large stack:

- **One Model for All**: Training an RF model on a small sub-stack, then predicting over the entire stack.
- **Divide And Conquer**: Dividing the large stack into several sub-stacks, then train an RF model for each.

## One Model For All
As for the first step, we recommend making a small sub-stack to train a Random Forest (RF) model using our plugin. This sub-stack can have about 20 slices selected across the whole stack (not just the beginning or last few slices). This way, when you extract and save the sub-stack's features, the storage file won't occupy too much space on the hard drive.  

!!! tip
    If the image resolution is high, it's better to down-scale images into a resolution of below 1200 pixels for the largest dimension.

After the training, you can save the RF model, and later apply it on the entire stack.  

## Divide And Conquer
Extracted features saved as an `HDF5` file can take up a huge space on the disk. In this method, to prevent disk space overflow, you can divide your large stack into several sub-stacks. Then use the plugin for each, separately.   
Although, you can try one trained model over another sub-stack, Random Forest model can not be fine-tuned. By using this method, you can achieve better annotations at the expense of spending more time on training several models.
