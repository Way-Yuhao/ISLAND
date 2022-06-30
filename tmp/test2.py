import numpy as np
import pandas as pd
import multiprocessing as mp
import time


def foo(i, arr, lock):
    with lock:
        arr.append(i)


def main():
    pool = mp.Pool(3)
    manager = mp.Manager()
    lock = manager.Lock()
    arr = manager.list()
    b = 123
    for i in range(10):
        r = pool.apply_async(foo, args=(i, arr, lock))
        # r.get()
    pool.close()
    pool.join()
    print(arr)


if __name__ == '__main__':
    main()
