import pandas as pd


def load_MIES(root_path):

    path = root_path / 'datasets/mies/data.csv'

    df = pd.read_csv(path, sep='\t')

    # drop users that somehow did no answered to intro, extro, no
    df = df[df['IE'] != 0]

    answers = df.filter(regex=r'Q\d+A')
    labels = df['IE']

    xs = answers.to_numpy()
    ys = labels.to_numpy()

    return xs, ys