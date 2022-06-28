import numpy as np
import pandas as pd


def main():
    a = np.array([1, 2, 3, 0, 0])
    b = np.array([3, 4, 3, 2, 2])
    print(np.average(a[a != 0]))
    d = {'col1': a, 'col2': b}
    df = pd.DataFrame(data=d)
    print(df)
    df.to_csv
if __name__ == '__main__':
    main()