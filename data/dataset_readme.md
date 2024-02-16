# ISLAND Public Dataset

This dataset contains interpolated brightness temperature (BT) and land surface temperature (LST) for 20 US regions with the highest population, according to the 2020 US Census. ISLAND generates these interpolated BT and LST files from Landsat 8 thermal infrared (TIR) inputs, NLCD land cover dataset 2019 release, and ASTER GED emissivity. 

The interpolated BT and LST files has a spatial resolution of 30 m and a revisit cycle (i.e. temporal resolution) of 16 days. The dataset spans Jan 1st 2017 to July 1st 2022.

Notice that some output files are missing because the input Landsat TIR image is missing. If over 99% of pixels are contaminated by either cloud or cloud shadow, the corresponding ISLAND BT and LST outputs will be empty images, where all pixel values are zero.

The file structure of this dataset is as follows:

    .
    ├── NAME OF THE REGION
    │   ├── bt    # brightness temperature
    │   │   ├── bt_YYYYMMDD.tif
    │   │   ├── ...
    │   ├── st    # land surface temperature
    │   │   ├── st_YYYYMMDD.tif
    │   │   ├── ...
    │   └── ...                 
    └── ...

Each .tif file is a GeoTIFF file where YYYY,MM, DD refers to the year, month, and date, respectively.