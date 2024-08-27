# ISLAND: Informing Brightness and Surface Temperature Through a Land Cover-based Interpolator

## Public Available Dataset
Please refer to the following Design-Safe link for the public available dataset:
https://doi.org/10.17603/ds2-3rf5-sd58


## Generate Your Own Land Surface Temperature (LST) Maps
Below are the instructions on how to generate LST maps for any region in the CONUS.
### Download data from EarthEngine
1. Identify a region, call it `city`
2. Navigate to `./jupyter/8 - ROI Extractor.ipynb ` to obtain `scene_id` associated with a LADNSAT image. The notebook will also ask you to you 4 coordinates defining a polygon within that LANDSAT image. 
3. Navigate to `./data/us_cities.csv`, append a new row in the format of `city | state | scene_id | coordinates` that you obtained from previous steps. 
4. Navigate to the root directory for this project, run `python region_sampler -c CITY`, replace `city` with value from above. This will strip all relevant data from Google EarthEngine.

### Process data
Once data is downloaded, run `python main -c CITY`, where `CITY` is defined as above. This will generate both timelapses of brightness temperature (BT) and land surface temperature (LST).  

### Data structure
Once data is downloaded and processed, you should see files being populated at `./data/` and sorted by region.

    .
    ├── ...
    ├── Houston                 # name of the region
    │   ├── analysis            # directory storing analysis outputs, if any
    │   ├── bt_series           # brightness temperature input GeoTIFF files
    │   ├── bt_series_png       # brightness temperature input, rescaled PNG
    │   ├── cloud               # cloud bitmask
    │   ├── output              # placeholder directory storing all ISLAND outputs
    │   ├── output_bt           # temporary directory, storing all BT outputs of ISLAND
    │   ├── output_st           # temporal directory, storing all LST outputs of ISLAND
    │   ├── output_referenced   # directory for geo-referenced output GeoTIFF files from ISLAND
    │   │   ├── bt              # geo-referenced BT output
    │   │   ├── st              # geo-referenced LST output
    │   ├── qa_series           # quality assessment bitmask
    │   ├── shadow              # cloud shadow bitmask
    │   ├── TOA_RGB             # top-of-atmosphere rescaled RGB images
    │   ├── metadata.csv        # a csv file containing list of Landsat 8 revisit dates for this site
    │   ├── nlcd_YYYYMMDD_color.tif # GeoTIFF file containing NLCD land cover lables, color mapped to RGB
    │   ├── nlcd_YYYYMMDD.tif   # GroTIFF file containing NLCD land cover labels
    │   └── ...                 
    └── ...

