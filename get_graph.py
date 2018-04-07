import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def main(c_account, c_kind, c_session):
    fname = 'report_' + c_account + '_' + c_kind + '_' + str(c_session)

    data = pd.read_table(fname, header=None)

    train = []
    test = []
    diff = []
    break_point = 0

    for i, tr in enumerate(data[0]):
        if tr != "train" and tr != "test":
            train.append(float(tr))
        if tr == "test":
            break_point = i
            break

    for i, te in enumerate(data[0]):
        if i > break_point and te != "diff":
            test.append(float(te))
        elif te == "diff":
            break

    assert len(train) == len(test)

    break_point *= 2

    for i, di in enumerate(data[0]):
        if i > break_point:
            diff.append(float(di))

    assert len(train) == len(diff)
    assert len(test) == len(diff)

    sns.set(color_codes=True)
    plt.plot(train)
    plt.plot(test)
    plt.plot(diff)
    plt.legend(['train', 'test', 'diff'], loc='best')
    plt.savefig('graph_'+ fname[7:]+'.png')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--account', type=str, default='GzuPark')
    args.add_argument('--kind', type=str, default='movie_phase1')
    args.add_argument('--session', type=int, default=128)
    config = args.parse_args()

    main(config.account, config.kind, config.session)
