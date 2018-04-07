import argparse
import get_graph


def load_data(file):
    with open(file, 'rt', encoding='utf-8') as f:
        data = f.readlines()
    result = []
    for i, d in enumerate(data):
        if i != 0 and i != 1:
            result.append(d.split())
    return result


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--account', type=str, default='GzuPark')
    args.add_argument('--kind', type=str, default='movie_phase1')
    args.add_argument('--session', type=int, default=129)
    config = args.parse_args()

    file_total = config.kind + '_submitted_total'
    file_session = config.kind + '_' + str(config.session)
    v_t = 'GzuPark/' + config.kind + '/' + str(config.session)

    submitted = load_data(file_total)

    total = []
    for cat in submitted:
        if cat[5] == v_t:
            total.append([cat[1], int(cat[6])])
    total = sorted(total, key=lambda x: x[1])

    result = []
    for res in total:
        result.append(float(res[0]))

    sess = load_data(file_session)

    session = []
    for cat in sess:
        for c in cat:
            if c[0] == 't':
                session.append(float(c[11:-1]))

    diff = []
    for i in range(len(session)):
        diff.append(result[i] - session[i])

    final = sum([['train'], session, ['test'], result, ['diff'], diff], [])

    fname = 'report_' + config.account + '_' + config.kind + '_' + str(config.session)
    with open(fname, 'w', encoding='utf-8') as f:
        f.writelines(["%s\n" % item  for item in final])

    get_graph.main(config.account, config.kind, config.session)
