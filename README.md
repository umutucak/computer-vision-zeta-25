# SPARK 2024 Utils

This repository contains all you need to start playing around with the SPARK 2024 Dataset.

## Stream 1: spacecraft semantic segmentation

Please first create a `data/` folder in `stream-1/`, then download the training and validation datasets in the newly created folder (see dedicated email for download link). After unziping the `*.zip` archives, the tree structure of `data/` must finally follow the one below:

<pre>
└───stream-1/
    ├───data/  
        ├───train/
        ├───val/
        ├───train.csv
        ├───val.csv
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


For this stream, semantic segmentation accuracy will be evaluated in addition to classification performance. More details about the metric are coming.

## Stream 2: spacecraft trajectory estimation

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

For this stream, both position and orientation accuracies will be evaluated. The metric is largely inspired by the [SPEED+ Challenge](https://kelvins.esa.int/pose-estimation-2021/). More precisely, we are going to sum the relative position error and the geodesic orientation error for each frame, then average these scores over all the frames and trajectories.

