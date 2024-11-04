# ISBI2025-ADASSO

Code for ISBI 2025 submission "*Uncovering Spatiotemporal Differences in Cortical Activity Corresponding to Two Tasks Using Data-Driven ADASSO Algorithm*"

## Prerequisites

1. Install the environment by `conda env create -f environment.yml` and check the env by `conda env list`.

## How to run the code
> [!TIP]
> Code is written as a python package, run the moudule with prefix `-m`, for example `python -m src.bool_mat`.

To reproduce the results using the proposed ADASSO algorithm, follow these steps:
    
1. Preprocess the raw data, split the trails into epochs.
2. Booleanize the read-value matrix into boolean matrix by `python -m src.bool_mat`.
3. Calculate the basis vectors B and occurence matrix C and D by `python -m src.calc_bcd`.
4. Once you have B, C, and D, you can perform analysis and generate plots based on these results.

## Problems

If you have any issue regards to the code or paper, feel free to contact Siwei by putting this into your terminal `echo c20yNDcwQHNjYXJsZXRtYWlsLnJ1dGdlcnMuZWR1Cg== | base64 --decode`.
