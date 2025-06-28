# Origin - Destination Survey Monterrey 2019

## Description

This repository enables usage of the origin destination survey within the CFC data pipelines.

It documents processing to the original data and defines several alternate versions for custom applications.

Specifically, it contains:
1. Stores and documents code to clean and preprocess the original survey into a consistent data set.
    1. The most recent clean version is located in `data/outputs/od-mty-2019/`
1. Stores the definition of traffic analysis zones used in the OD survey.
1. Code to translate to GTAModel schema.
    1. The most recent OD version with the GTAModel schema is in `data/outputs/od-mty-2019-gta/`
1. Assignment of census geometries to TAZ. Geoemtry is either AGEB or Locality, depending on the length of CVEGEO id. Assignment file is store in `data/outputs/taz_assigment.yaml`. A PDF report of the assignment is also provided.

## Usage

Install the dependencies using `uv` is recommended, this will also install the processing functions as a package. Then, you can use SnakeMake to generate the outputs.

To generate census geometry assignments:
```{.sh}
uv run snakemake -c 1 generate_assignments
```

## TODO
- [ ] Cleanup trip legs. Trip legs are still inconsistent.

## Authors
- Gonzalo G. Peraza Mues