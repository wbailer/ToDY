# ToDY toolchain

This directory contains the toolchain that has been used to generate additional annotations for the [Skyfinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/), and a script to prepare data directories for training.

## Dependencies

### Pytorch Image Models (TIMM)

[TIMM](https://github.com/rwightman/pytorch-image-models/tree/master/timm) is used as framework for training our classifiers. For other classifier frameworks, the data preparation script may need to be adjusted.

## Steps to run the pipeline


### Add annotations

Run `prepare_classes.py` with the following arguments:

```
--annotate --dataroot <Skyfinder root directory> 
```

`dataroot` denotes the root directory containing Skyfinder metadata. A new file `skyfinder_annotations.csv` will be created there.

### Sample images

This mode can be used to sample images for manual verification or to sample a final set to be used. Optionally, data can be augmented by cropping different images with less sky region. The number of samples per class will be balanced, i.e. each class will have as many samples as the fewest number of samples of any of the classes.


Run `prepare_classes.py` with the following arguments:

```
--sample_images --dataroot <Skyfinder root directory> --outdir <output directory> 
```

`dataroot` denotes the root directory containing Skyfinder metadata.

`outdir` denotes the output directory for annotations (and images, if `--augment` is used).

Optional arguments:

`--noiseth`: keep only images below specified noise threshold

`--timeth`: keep only images below specified time distance between image and metadata time

`--emptyth`: if <1, discard empty images

`--min_height`: minimum height of sampled (or augmented images)

`--augment`: generate additional images with less sky region than original image

`--cvat`: write file copy script and XML snippets to patch annotations for [CVAT annotation tool](https://cvat.org/)

`--oversample`: multiplication factor for the number of samples determined (this is especially useful when drawing samples for manual verification)


### Update from CVAT export

If manual annotations in CVAT have been done, import those annotations. 

Run `prepare_classes.py` with the following arguments:

```
--update --dataroot <Skyfinder root directory> --outdir <output directory> -cvatxml <CVAT 1.1 XML export>
``` 

The following update procedure is applied:
- if an image if not in the CVAT export, no annotations will be modified
- if an image has any time of day annotations in the CVAT export, the time of day annotation will be replaced
- if an image has any season annotations in the CVAT export, the season annotation will be replaced
- if an image has no labels in the CVAT export, time of day and season annotations will all be set to 0 (image is excluded)

### Get dataset statistics


To print statistics, run `prepare_classes.py` with the following arguments:

```
--stats --dataroot <Skyfinder root directory> 
``` 

Optional arguments:

`--noiseth`: keep only images below specified noise threshold

`--timeth`: keep only images below specified time distance between image and metadata time

`--emptyth`: if <1, discard empty images


### Create TIMM input directories

Run `create_timm_skyfinder.py` with the following arguments to create directories with the annotations files for one task.

```
season|tod <annotationfile> <outputdir> <sourcedir> <valshare>
```

`annotationfile` denotes the CSV file listing sampled data

`outputdir` denotes root of the directory structure where training and validation files will be stored

`sourcedir` denotes the root of the Skyfinder image directory

`valshare` denotes the share of data used for validation (e.g., 0.1 = 90% training, 10% validation data)

## License

The tools provided here are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

