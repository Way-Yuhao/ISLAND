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
    cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=len(cycles))
        with mp.Pool(2) as pool:
            results = [pool.apply_async(foo, args=(i,), callback=lambda _: progress.update(task_id, advance=1)) for i in
                       cycles]
            # Wait for all tasks to complete
            for result in results:
                result.get()


if __name__ == '__main__':
    # sequential()
    parallel()