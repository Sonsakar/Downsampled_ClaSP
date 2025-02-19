

# Evaluating Downsampling algorithms for visualization as a preprocessing step for Time Series Segmentation with ClaSP

This GitHub repository provides the notebooks and code used for the implementation of the tests applied for the master thesis "Evaluating Downsampling algorithms for visualization as a preprocessing step for Time Series Segmentation with ClaSP":

The task of segmentation is a prominent problem of time series classification. The Classification Score Profile (ClaSP) is a segmentation tool, that outperforms other state-of-the-art segmentation tools regarding the Covering score of the predicted segments or change points. However, due to its complexity it requires higher runtimes, especially on large data samples. Reducing the length of the time series by sampling the datapoints can provide a solution to this issue. In this thesis an introduction to the concept of downsampling in the context of time series as well as an overview of six downsampling algorithms is provided. The Downsampled ClaSP is proposed, as a solution for the known issues regarding runtime and complexity of time series segmentation using ClaSP. For this method an ablation study on design choices for the parameters of the downsampling and segmentation is performed, resulting in data-adaptive and, if possible, predictable approaches for each parameter. The main results show that ClaSP can, depending on the data, achieve not significantly lower or even higher Covering scores on preprocessed data using downsampling than on original data.

## Related Repositories

* [claspy](https://github.com/ermshaua/claspy) is used for the time series segmentation with ClaSP
* [tsdownsample](https://github.com/predict-idlab/tsdownsample) for downsampling using the MinMax, M4, LTTB, MinMaxLTTB and EveryNth algorithm
* [LTD](https://github.com/FarisYang/LTTB-LTD-py/tree/main) implements the Largest-Triangle-Dynamic Downsampler. A slighty adapted version of this implementation is included in this repository
* The [Time Series Segmentation Benchmark](https://github.com/ermshaua/time-series-segmentation-benchmark) and the [Human Acitivity Segmentation Challenge](https://github.com/patrickzib/human_activity_segmentation_challenge) have been used as benchmarks for this thesis

## Notebooks

| Notebook | MT Chapter   | Description |
| :--- | :--- | :--- |
| `0a_Downsample_All` | 2.2 | Downsamples all time series in TSSB and HASC using all Downsampling algorithms |
| `0b_ClaSP_All`| 2.1 | Segments previously downsampled and original time series using ClaSP |
| `1_Splitting` | 3.1 and 5.1 | Splits time series at change points, downsamples splits, concatenates downsampled splits and saves offsets as change points |
| `2_Upscaling` | 3.2 and 5.2 | Tests different change point upscaling approaches |
| `3a_Window_Size` | 3.3 and 5.3 | Tests different window size approaches for Downsampled ClaSP |
| `3b_Compression_Ratio` | 3.3 and 5.4 | Tests different compression ratio approaches for Downsampled ClaSP |
| `4_Main_Experiment` | 6.1 | Runs the main experiment with the best parameters determined previously |
| `5a_Downsampling_Performance` | 6.2 | Tests the impact of the downsampling performance on the segmentation performance |
| `5b_Group_Performance` | 6.2 | Groups segmentation results by types of data |

## Prerequisites and Instructions

The requirements.txt includes all packages needed to run all notebooks. 
```bash
pip install -r requirements.txt
```

The consecutive notebooks requires data generated by their predecessor. Therefore it is not possible to run a notebook before the required data has been generated. The .pkl files are not provided with this repository due to their file size. However, notebook will write and read the results and inputs needed automatically from/to the provided directory structure. 



