__author__ = 'yuhao liu'

NUM_CLASSES = 6  # including y=0 (cloud)
VALIDATION_SPLIT = 0.2  # percentage of training data reserved for validation
VIS_PARAM = {
    'label_min': 1,
    'label_max': 5,
    'label_palette': [
        # implicityly: 0, Black
        [255, 255, 0],  # 1, Yellow
        [218, 165, 32],  # 2, Gold
        [0, 150, 0],  # 3, Dark Green
        [150, 255, 150],  # 4, Light Green
        [0, 0, 255],  # 5, Blue
    ]
}

################### URBAN BOUNDARY ##########################
# coordinates of rectangle surrounding Houston
HOUSTON_BOUNDING_BOX = [[[-95.690165, 29.5937], [-95.690165, 30.266005],
                         [-94.900379, 30.266005], [-94.900379, 29.5937]]]

LANDSAT8_HOUSTON_REFERENCE_DATE = '20130411'
LANDSAT9_HOUSTON_REFERENCE_DATE = '20211031'
NULLIFY_DATES_AFTER = '20220501'

CROP_HEIGHT_COORD = .0675 * 2  # degrees in latitude
CROP_WIDTH_COORD = .0775 * 2  # degrees in longitude
CROP_SHAPE = (512, 512)  # (h, w) in pixels

################ SATELLITE SOURCES #########################

LANDSAT8_META = {
    'name': 'USGS Landsat 8 Level 2, Collection 2, Tier 1',
    'id': "LANDSAT/LC08/C02/T1_L2/LC08_025039_{}",
    'selected_bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_TRAD', 'ST_CDIST'],
    'rgb_bands': ['SR_B4', 'SR_B3', 'SR_B2'],
    'scales': [2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 0.00341802, 0.01],
    'offsets': [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149, 0.],
    'cloud_dist_bound': 1e-7,  # upper bound (saturation limit) for cloud distance, in kilometer
    'avg_cloud_cover': .379  # from Jan 2018 to Feb 2020
}

NLCD_2019_META = {
    'name': 'NLCD 2019: USGS National Land Cover Database, 2019 release',
    'collection_id': 'USGS/NLCD_RELEASES/2019_REL/NLCD',
    'filter': '2019',
    'selected_bands': ['landcover'],
    'scales': [1],  # no scaling
    'offsets': [0],  # no offset
    'lut': {
        '11': '466b9f',
        '12': 'd1def8',
        '21': 'dec5c5',
        '22': 'd99282',
        '23': 'eb0000',
        '24': 'ab0000',
        '31': 'b3ac9f',
        '41': '68ab5f',
        '42': '1c5f2c',
        '43': 'b5c58f',
        '51': 'af963c',
        '52': 'ccb879',
        '71': 'dfdfc2',
        '72': 'd1d182',
        '73': 'a3cc51',
        '74': '82ba9e',
        '81': 'dcd939',
        '82': 'ab6c28',
        '90': 'b8d9eb',
        '95': '6c9fb8'},
    'class_names': {
        '11': 'Open water',
        '12': 'Perennial ice/snow',
        '21': 'Developed, open space',
        '22': 'Developed, low intensity',
        '23': 'Developed, medium intensity',
        '24': 'Developed high intensity',
        '31': 'Barren land',  # (rock / sand / clay)',
        '41': 'Deciduous forest',
        '42': 'Evergreen forest',
        '43': 'Mixed forest',
        '51': 'Dwarf scrub',
        '52': 'Shrub / scrub',
        '71': 'Grassland / herbaceous',
        '72': 'Sedge/herbaceous',
        '73': 'Lichens',
        '74': 'Moss',
        '81': 'Pasture / hay',
        '82': 'Cultivated crops',
        '90': 'Woody wetlands',
        '95': 'Emergent herbaceous wetlands'}
}


# LANDSAT8_META_REFERENCE = {
#     'name': 'USGS Landsat 8 Level 2, Collection 2, Tier 1',
#     'id': "LANDSAT/LC08/C02/T1_L2/LC08_025039_20190514",
#     'selected_bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_TRAD'],
#     'rgb_bands': ['SR_B4', 'SR_B3', 'SR_B2'],
#     'scales': [2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 2.75e-05, 0.00341802],
#     'offsets': [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 149]
# }

################ CONSOLE STDOUT #########################

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
