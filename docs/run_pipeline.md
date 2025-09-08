After training your **RF** model in napari, you can use `run_pipeline.py` to run the pipeline on a new set of images without using napari gui and only by commandline.  
This allows you to run the whole pipeline on HPC or other servers as a batch job.

## Usage
```bash
python run_pipeline.py -h
```

```bash
FeatureForest run-pipeline script

options:
  -h, --help            show this help message and exit
  --data DATA           Path to the input image
  --outdir OUTDIR       Path to the output directory
  --rf_model RF_MODEL   Path to the trained RF model
  --feat_model {SAM2_Large,SAM2_Base,μSAM_LM,μSAM_EM_Organelles,Cellpose_cyto3,MobileSAM,SAM,DinoV2}
                        Name of the model for feature extraction
  --no_patching         If true, no patching will be used during feature extraction
  --smoothing_iterations SMOOTHING_ITERATIONS
                        Post-processing smoothing iterations; default=25
  --area_threshold AREA_THRESHOLD
                        Post-processing area threshold to remove small regions; default=50
  --post_sam            to use SAM2 for generating final masks
  --only_extract        to only extract features to zarr file without running prediction pipeline
```

For example if you just want to extract features from a stack, using SAM2 Large model with no patching:
```bash
python run_pipeline.py \
--data /path/to/input.tif \
--outdir /path/to/output/ \
--feat_model SAM2_Large \
--no_patching \
--only_extract
```

Another example, to run the whole pipeline (feature extraction, prediction, post-processing) on a stack using a trained RF model:
```bash
python run_pipeline.py \
--data /path/to/input.tif \
--outdir /path/to/output/ \
--rf_model /path/to/trained/rf_model.bin \
--feat_model SAM2_Large
--area_threshold 10 \
--post_sam
```
