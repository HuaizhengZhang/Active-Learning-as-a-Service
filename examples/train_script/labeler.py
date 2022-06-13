import json
import pandas as pd
from pathlib import Path


def get_label(response_path, output_path=None):
    response_path = Path(response_path)
    output_path = (output_path and Path(output_path)) or response_path.with_suffix('.csv')
    # create parent directory
    output_path.parent.mkdir(exist_ok=True)

    with open(response_path) as f:
        response: dict = json.load(f)

    image_paths = response['query_results']
    labels = list()
    # get label from the image_path
    for p in image_paths:
        name = p[:-4].split('/')[-1]
        label = name.split('_')
        labels.append(label)

    df = pd.DataFrame(zip(image_paths, labels), columns=('image_paths', 'labels'))
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    get_label(response_path='data/lc_cifar_5k_from_all.json')
    get_label(response_path='data/lc_cifar_100_from_1000.json')
    get_label(response_path='data/lc_cifar_200_from_1000.json')
