import argparse
import os
import time
from rich.progress import Progress
import multiprocessing as mp
from util.helper import capture_stdout

@capture_stdout
def foo(x):
    print(x)
    time.sleep(0.3)
    print(x)
    time.sleep(0.3)
    print(x)
    time.sleep(0.3)
    return x


def sequential():
    cycles = 10
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=cycles)
        for i in range(cycles):
            foo(i)
            progress.update(task_id, advance=1)


def parallel():
    cycles = 10
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=cycles)
        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(foo, [(i,) for i in range(cycles)])
            progress.update(task_id, completed=cycles)


if __name__ == '__main__':
    # sequential()
    parallel()