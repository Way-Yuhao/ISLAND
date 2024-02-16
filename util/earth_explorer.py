import json
from landsatxplore.earthexplorer import EarthExplorer
from landsatxplore.api import API
# ee = EarthExplorer(username='yuhao.liu@rice.edu', password='LIUyh0929000')
#
# # # ee.download('LT51960471995178MPS00', output_dir='./data')
# # ee.download('LC08_025039_20180204', output_dir='./data')
# #
# # """
# # LT5_196047_1995178MPS00
# # """
# ee.logout()


def construct_scene_id(location_id):
    pass


def try_search():
    # Initialize a new API instance and get an access key
    api = API(username='yuhao.liu@rice.edu', password='jyfjeb-wysfyd-Dovri8')

    # Search for Landsat TM scenes
    scenes = api.search(
        dataset='landsat_tm_c1',
        latitude=50.85,
        longitude=-4.35,
        start_date='1995-01-01',
        end_date='1995-10-01',
        max_cloud_cover=10
    )

    print(f"{len(scenes)} scenes found.")

    # Process the result
    for scene in scenes:
        print(scene['acquisition_date'].strftime('%Y-%m-%d'))
        # Write scene footprints to disk
        fname = f"{scene['landsat_product_id']}.geojson"
        with open(fname, "w") as f:
            json.dump(scene['spatial_coverage'].__geo_interface__, f)

    api.logout()


def init():
    ee = EarthExplorer(username='yuhao.liu@rice.edu', password='jyfjeb-wysfyd-Dovri8')
    return ee


def download():
    ee = init()
    # ee.download('LC80250392018323LGN00', output_dir='./data')
    ee.logout()
    return


def search():
    pass


def main():
    download()


if __name__ == '__main__':
    main()
