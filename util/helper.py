import sys
import os
from os import path as p
import traceback
import warnings
import functools
from config import bcolors
import time
import datetime as dt
import uuid
import pandas as pd
from matplotlib import pyplot as plt
import requests
from dotenv import load_dotenv

# Global variables
slack_alert_msg_printed = False

def yprint(msg):
    """
    Print to stdout console in yellow.
    :param msg:
    :return:
    """
    sys.stdout.flush()
    print(f"{bcolors.WARNING}{msg}{bcolors.ENDC}")
    sys.stdout.flush()


def rprint(msg):
    """
    Print to stdout console in red.
    :param msg:
    :return:
    """
    sys.stdout.flush()
    print(f"{bcolors.FAIL}{msg}{bcolors.ENDC}")
    sys.stdout.flush()


def pjoin(*args):
    """
    Joins paths for OS file system while ensuring the corrected slashes are used for Windows machines
    :param args:
    :return:
    """
    path = os.path.join(*args).replace("\\", "/")
    return path


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def timer(func):
    """
    Decorator to measure the execution time of a function.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        func(*args, **kwargs)
        yprint('---------------------------------')
        stop_time = time.monotonic()
        yprint(f'Processing time = {dt.timedelta(seconds=stop_time - start_time)}')

    return inner


class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise

    def write(self, x): pass


def get_season(now):
    Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (dt.date(Y, 1, 1), dt.date(Y, 3, 20))),
               ('spring', (dt.date(Y, 3, 21), dt.date(Y, 6, 20))),
               ('summer', (dt.date(Y, 6, 21), dt.date(Y, 9, 22))),
               ('autumn', (dt.date(Y, 9, 23), dt.date(Y, 12, 20))),
               ('winter', (dt.date(Y, 12, 21), dt.date(Y, 12, 31)))]
    # now = dt.datetime.strptime(str(now), '%Y%m%d').date()
    if isinstance(now, dt.datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def hash_(str_=None, bits=4):
    """
    modifies output filename by appending a hashcode. If input (str_) is provided,
    it must contain substring '{}' as a placeholder for hascode.
    """
    hashcode = uuid.uuid4().hex[:bits]
    if str_ is not None:
        return str_.format(hashcode)
    else:
        return hashcode


def parse_csv_dates(city_name):
    root_ = f'./data/{city_name}/'
    assert p.exists(root_)
    df = pd.read_csv(p.join(root_, 'metadata.csv'))
    assert df is not None, 'csv file not found.'
    dates = df['date'].values.tolist()
    dates = [str(d) for d in dates]
    return dates


def get_scene_id(city_name):
    """
    Gets a 6-digit scene_id for a city. Requires metadata to be already saved in
    '../data/us_cities.csv'
    :param city_name:
    :return:
    """
    cities_list_path = "../data/us_cities.csv"
    # print(f'Parsing metadata from {cities_list_path}')
    cols = list(pd.read_csv(cities_list_path, nrows=1))
    cities_meta = pd.read_csv(cities_list_path, usecols=[i for i in cols if i != 'notes'])
    row = cities_meta.loc[cities_meta['city'] == city_name]
    if row.empty:
        raise IndexError(f'City {city_name} is not specified in {cities_list_path}')
    scene_id = str(row.iloc[0]['scene_id'])
    if len(scene_id) == 5:
        scene_id = '0' + scene_id
    return scene_id


def save_cmap(img, out_path, palette='inferno', vmin=None, vmax=None):
    """
    Saves a color-mapped image without boarder. Original resolution of the image
    is retained.
    :param img: numpy array
    :param out_path:
    :param palette:
    :param vmin:
    :param vmax:
    :return:
    """
    cmap = plt.get_cmap(palette)
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    out = cmap(norm(img))
    plt.imsave(out_path, out)
    plt.close()
    return


def alert(message):
    """
    Sends a message to a designated slack channel, which a SLACK_WEBHOOK_URL to be set in .env file.
    If webhook URL is not found, the message is printed to stdout in red.
    :param message:
    :return:
    """
    global slack_alert_msg_printed # print out error msg only once
    load_dotenv('../env/.env')
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    if webhook_url is None:
        if not slack_alert_msg_printed:
            msg = 'To send alerts to slack, set SLACK_WEBHOOK_URL in /env/.env file.'
            yprint(msg)  # Assuming yprint is a typo and meant print. Adjust as necessary for your logging method.
            yprint('Message routed to stdout')
            slack_alert_msg_printed = True  # Mark the warning as printed
        rprint(message)
        return
    else:
        data = {'text': message,
                'username': 'Webhook Alert',
                'icon_emoji': ':robot_face:'}
        response = requests.post(webhook_url, json=data)
        if response.status_code != 200:
            raise ValueError(
                'Request to slack returned an error %s, the response is:\n%s'
                % (response.status_code, response.text)
            )
        return


def monitor(func):
    """
    Decorator to monitor the execution of a function. If the function fails, the stack trace is printed.
    If the function succeeds, a message is sent to a designated slack channel.
    :param func:
    :return:
    """
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
            alert(f'Code executed successfully: {func.__name__}')
        except Exception as e:
            stack_trace = traceback.format_exc()
            title = 'Code failed'
            alert(f'*{title}*```{stack_trace}```')
            raise e
    return inner
