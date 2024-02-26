# SPARK 2024 Utils

This repository contains all you need to start playing around with the SPARK 2024 Dataset.

## Stream 1: spacecraft semantic segmentation

### Dataset structure

Please first create a `data/` folder in `stream-1/`, then download the training and validation datasets in the newly created folder (see dedicated email for download link). After unziping the `*.zip` archives, the tree structure of `data/` must finally follow the one below:

<pre>

└───data/  
    ├───images/
    │	├───object_1/
    │	│	│
    │	│	├───train/
    │	│	│	└───img1....
    │	│	└───val/
    │	│		└───img1....
    │	│
    │	└───object_2/
    │		│
    │		├───train/
    │		│	└───img1....
    │		└───val/
    │			└───img1....
    │	
    │	
    ├───mask/
    │	├───object_1/
    │	│	│
    │	│	├───train/
    │	│	│	└───mask1....
    │	│	└───val/
    │	│		└───mask1....
    │	│
    │	└───object_2/
    │		│
    │		├───train/
    │		│	└───img1....
    │		└───val/
    │			└───img1....
    │	
    ├───train.csv
    └───val.csv
</pre>

The `stream-1/visualize_data.ipynb` notebook contains basic functions to load and display dataset samples.

The correspondences between class names and indexes are given below.

| Class name                | Index    |
|---------------------------|----------|
| `VenusExpress`            | 0        |
| `Cheops`                  | 1        |
| `LisaPathfinder`          | 2        |
| `ObservationSat1`         | 3        |
| `Proba2`                  | 4        |
| `Proba3`                  | 5        |
| `Proba3ocs`               | 6        |
| `Smart1`                  | 7        |
| `Soho`                    | 8        |
| `XMM Newton`              | 9        |


### Evaluation

For this stream, semantic segmentation algorithms will be evaluated using the following procedure.
 
First of all, we consider that each image contains one or two items to be segmented, i.e. a spacecraft body (red-labeled pixels in the `mask\` pictures) and a set of solar panels (blue-labeled pixels). In some of these images, the number of blue-labeled pixels is negligible compared to the number of red-labeled pixels (less than 5%). Therefore, in such pictures, we discard the solar panel class and measure the segmentation performance only on the spacecraft body class.
 
Let's consider the spacecraft body class first.
 
For every picture, we compute the intersection-over-union score (IoU) between the predicted and groundtruth masks. Then, we compute the proportion of correctly segmented images, i.e. the percentage of images in the dataset for which the IoU is above a certain threshold (e.g., 0.5), and we average the proportions over different thresholds: 0.5, 0.55, ... 0.95, to give more importance to more accurate results.
 
We then do the same for every picture featuring solar panels, and we finally average these two scores.

Please note that neither the classification nor the object detection performance will be taken into account in the evaluation, so that your method does not need to perform any of these tasks.

## Stream 2: spacecraft trajectory estimation

### Dataset structure

Please first create a `Data/` folder in `stream-2/`, then download the training and validation datasets in the newly created folder (see dedicated email for download link). After unziping the `*.zip` archives, the tree structure of `Data/` must finally follow the one below:

<pre>
└───stream-2/
    ├───Data/  
        ├───train/
            ├───images/
                ├───GTXXX/
                ├───...
            ├───train.csv
        ├───val/
            ├───images/
                ├───GTXXX/
                ├───...
            ├───val.csv
</pre>

The `stream-2/visualize_data_spark.py` script contains basic functions to load and display dataset samples.

### Evaluation

For this stream, both position and orientation accuracies will be evaluated. The metric is largely inspired by the [SPEED+ Challenge](https://kelvins.esa.int/pose-estimation-2021/). More precisely, we are going to sum the relative position error and the geodesic orientation error for each frame, then average these scores over all the frames and trajectories.

