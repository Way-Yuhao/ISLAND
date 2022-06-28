# Continuous Urban Surface Temperature Mapping
## TODO
- [x] consider adding class balanced weights
- [x] consider adding parallel GPU support
- [ ] What is the temporal locality of labels?
- [ ] Is TRAD only for BAND 10?
- [ ] IsST_TRAD corrected for atmospheric conditions?
- [ ] Change Thermal Band input
- [ ] What algorithm did USGS use to compute surface reflectance?
- [ ] switching from CLOUD DISTANCE to (cloud confidence & cloud shadow confidence)
- [ ] Look into QA_PIXEL and QA_RADSAT
- [ ] Is L1 (TOA) products more accessible than L2 (Surface) products?
- [ ] Plot NDVI change over time
- [ ] Plot NVID/PV value for concrete and bare soil
- [ ] Plot ASTER GED over time
- [ ] Incorporate LANDSAT 8 to improve temporal continuity
- [ ] Seg Net: pixel based vs. object based