# [RSASE 2024] ISLAND: Interpolating Land Surface Temperature using land cover

## Public Available Dataset
Please refer to the following Design-Safe link for the public available dataset:
https://doi.org/10.17603/ds2-3rf5-sd58


## Generate Your Own Land Surface Temperature (LST) Maps
Below are the instructions on how to generate LST maps for any region in the CONUS.
### Download data from EarthEngine
1. Identify a region, e.g., `CITY`
2. Navigate to `./jupyter/8 - ROI Extractor.ipynb ` to obtain `scene_id` associated with a LADNSAT image. The notebook will also ask you to you 4 coordinates defining a polygon within that LANDSAT image. 
3. Navigate to `./config/us_cities.csv`, append a new row in the format of `CITY | state | scene_id | coordinates` that you obtained from previous steps. 
4. Navigate to the root directory for this project, run `python region_sampler -c CITY`, replace `CITY` with value from above. This will strip all relevant data from Google EarthEngine.

### Process data
Once data is downloaded, run `python main -c CITY`, where `CITY` is defined as above. This will generate both timelapses of brightness temperature (BT) and land surface temperature (LST).  

### Data structure
Once data is downloaded and processed, you should see files being populated at `./data/` and sorted by region.

    .
    ├── ...
    ├── CITY                    # name of the region
    │   ├── analysis            # directory storing analysis outputs, if any
    │   ├── cloud               # cloud bitmask
    │   ├── lst                 # land surface temperature (LST) input
    │   ├── output              # placeholder directory storing intermediate (non geo-referenced) ISLAND outputs
    │   ├── output_referenced   # directory for geo-referenced output GeoTIFF files from ISLAND
    │   │   ├── lst             # geo-referenced LST output
    │   ├── qa_series           # quality assessment bitmask
    │   ├── shadow              # cloud shadow bitmask
    │   ├── TOA_RGB             # top-of-atmosphere rescaled RGB images (optional)
    │   ├── metadata.csv        # a csv file containing list of Landsat 8 revisit dates for this site
    │   ├── nlcd_YYYY_color.tif # GeoTIFF file containing NLCD land cover lables, color mapped to RGB
    │   ├── nlcd_YYYY.tif       # GroTIFF file containing NLCD land cover labels
    │   └── ...                 
    └── ...

