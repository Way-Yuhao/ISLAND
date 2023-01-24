# Landcover-based Surface Temperature Interpolator

## Resources 
* Overleaf document: https://www.overleaf.com/project/63c6f44f8443b76167d1829e
* Outline of the manuscript: https://docs.google.com/document/d/1SHeuhIw01oC5Mv04Tol6JW15W0kIbSSt-10vKv3W4N8/edit?usp=sharing
* Google Drive Folder (permission requried): https://drive.google.com/drive/folders/11hxVhBQRy96oryDJ2QIbtyFIEgMj020L 
* Zotero library: https://www.zotero.org/groups/4916357

## Public Available Dataset

| City | Brightness Temperature Ready | Surface Temperature Ready |
| --- | :---: | :---:|
| New York | :heavy_check_mark: | |
| Los Angeles | :heavy_check_mark: | |
| Chicago | :heavy_check_mark: | |
| Houston | :heavy_check_mark: |:heavy_check_mark:|
| Phoenix | :heavy_check_mark: | |
| Philadelphia |:heavy_check_mark:| |
| San Antonio |:heavy_check_mark:|:heavy_check_mark: |
| San Diego |:heavy_check_mark:|emis 20171227 missing|
| Dallas |:heavy_check_mark:|:heavy_check_mark:|
| San Jose |download incomplete| |
| Austin |:heavy_check_mark: need to move files| |
| Jacksonville |no temporal reference for 20180816| |
| Fort Worth |download incomplete| |
| Columbus |download incomplete| |
| Indianapolis |:heavy_check_mark:|:heavy_check_mark:|
| Charlotte |download incomplete| |
| San Francisco |:heavy_check_mark: |:heavy_check_mark: |
| Seattle | | |
| Denver | :heavy_check_mark:| |
| Olkahoma City | | |


## Generate Your Own Timelapse of Surface Temperature

### Download data from EarthEngine
1. Identify a region, call it `city`
2. Navigate to `./jupyter/8 - ROI Extractor.ipynb ` to obtain `scene_id` associated with a LADNSAT image. The notebook will also ask you to you 4 coordinates defining a polygon within that LANDSAT image. 
3. Navigate to `./data/us_cities.csv`, append a new row in the format of `city | state | scene_id | coordinates` that you obtained from previous steps. 
4. Navigate to the root directory for this project, run `python region_sampler -c CITY`, replace `city` with value from above. This will strip all relevant data from Google EarthEngine.

### Process data
Once data is downloaded, run `python main -c CITY`, where `CITY` is defined as above. This will generate both timelapses of brightness temperature (BT) and surface temperature (ST).  

## TODO
- [ ] Geo-reference our public results
- [x] Resolve key error for NLCD ocean labels
- [ ] Show comparison with baselines
- [ ] Plot correlation between urban hotspots and SVI
- [ ] finalize journal to publish 
- [ ] Incorporate LANDSAT 9 to improve temporal continuity
