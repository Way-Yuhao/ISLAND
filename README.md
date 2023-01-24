# Landcover-based Surface Temperature Interpolator

## Resources 
* Overleaf document: https://www.overleaf.com/project/63c6f44f8443b76167d1829e
* Outline of the manuscript: https://docs.google.com/document/d/1SHeuhIw01oC5Mv04Tol6JW15W0kIbSSt-10vKv3W4N8/edit?usp=sharing
* Google Drive Folder (permission requried): https://drive.google.com/drive/folders/11hxVhBQRy96oryDJ2QIbtyFIEgMj020L 
* Zotero library: https://www.zotero.org/groups/4916357

## Data
Here is a list of cities where we've deployed our model during an observation time window of Jan 2017 to Jun 2022. These data are to be released to the public once completed. Notice that the data in this dataset is only derived from LANDSAT 8 and does not include LANDSAT 9, meaning that the time gap between consecutive frames is 16 days for each region. 
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
| San Jose |:heavy_check_mark:|:heavy_check_mark:|
| Austin |:heavy_check_mark: need to move files| |
| Jacksonville |no temporal reference for 20180816| |
| Fort Worth |:heavy_check_mark:|:heavy_check_mark:|
| Columbus |download incomplete| |
| Indianapolis |:heavy_check_mark:|:heavy_check_mark:|
| Charlotte |download incomplete| |
| San Francisco |:heavy_check_mark: |:heavy_check_mark: |
| Seattle | | |
| Denver | :heavy_check_mark:| |
| Olkahoma City | | |

Note that some LANDSAT 8 frames are missing.

## TODO
- [ ] Geo-reference our public results
- [x] Resolve key error for NLCD ocean labels
- [ ] Show comparison with baselines
- [ ] Plot correlation between urban hotspots and SVI
- [ ] finalize journal to publish 
- [ ] Incorporate LANDSAT 9 to improve temporal continuity
