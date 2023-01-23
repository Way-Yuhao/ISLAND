import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import cv2

def foo(i, arr, lock):
    with lock:
        arr.append(i)


def mp_main():
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


def main():
    # nlcd = cv2.imread('./data/Jacksonville/')
    arr = np.array([0, 0, 1, 0, 2])
    print(np.count_nonzero(arr == 0))


if __name__ == '__main__':
    main()
