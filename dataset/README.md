# ToDY Dataset

Release of the additional annotation files of the ToDY Dataset. The dataset is derived from the [Skyfinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/). 


## Annotations

The annotations contain per image annotations for time of day and season. In addition, some image and data quality metrics that were used in the process are provided. 

### Time of day class definitions

| Class | Definition |
| ----------- | ----------- |
| night | night time | 
| twilight | civil twilight, before sunrise/after sunset |
| sunrise | sun above horizon, until fully above horizon |
| sunset | sun above horizon, after being fully above horizon |
| fulldaylight | sun completely above horizon |
| day | day time, i.e. fulldaylight, sunrise or sunset (not used as a separate class, can be derived from classifications) |

### Season class definitions

| Class | 
| ----------- | 
| spring |  
| summer | 
| fall | 
| winter | 

### Column headings in the annotation files

Columns starting with S_ contain binary season annotations, columns starting with D_ contain binary time of day annotations

In addition, image properties are contained:
- I_Height: height of the image in pixels
- I_Width: width of the image in pixels
- I_LowestSky: the row index of the lowest pixel that belongs to the sky region
- I_Lowest09Sky: the row index determined as the 0.9 quantile of the lowest sky pixels of each column

The following quality attributes are contained:
- IQ_Empty: image seems to be empty (more than 50% of pixels have the same RGB value)
- IQ_Noise: a noise score for the image (0 .. no noise, ~1.0 moderate noise, >1.0 increasing noise)
- DQ_TimeDiffers: time difference in minutes between the timestamp in the file name and the time in the metadata (TZ offset corrected), indication on how reliable the time specified in the metadata is


### Annotation files

`skyfinder_annotations.csv` contains all additional annotations for the Skyfinder images.

`skyfinder_sampled_season.csv` contains a balanced sampled set of annotations for season classification.

`skyfinder_sampled_tod.csv` contains a balanced sampled set of annotations for time of day classification.

The following file lists are of the train/test splits are provided for reference: `season_train.txt, season_val.txt, tod_train.txt, tod_val.txt`. Note that in order to have unique files names they are composed of `<camid>_<filename>`.

## License

The license terms of the [Skyfinder dataset](https://cs.valdosta.edu/~rpmihail/skyfinder/) apply to the images. The additional annotates provided here are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
