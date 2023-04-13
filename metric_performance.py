from pycocoevalcap.eval import calculate_metrics
import os
import numpy as np
import json
import argparse
from utils.logger_new import Logger


def create_dataset(array):
    dataset = {'annotations': []}

    for i, caption in enumerate(array):
        dataset['annotations'].append({
            'image_id': i,
            'caption': caption
        })
    return dataset


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def _init_writer(self):
    writer = open(os.path.join(self.result_path, self.metric_name), 'w')
    return writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--result_path', type=str,
    #                     default='./results/clean_test.json')
    parser.add_argument('--result_path', type=str,
                        default='./result_logs/20230412-2254trainresnet152/results')
    parser.add_argument('--caption_name', type=str,
                        default='debug.json')
    parser.add_argument('--metric_name', type=str,
                        default='metric.txt')
    args = parser.parse_args()
    print(args)

    test = load_json(os.path.join(args.result_path, args.caption_name))
    datasetGTS = {'annotations': []}
    datasetRES = {'annotations': []}

    for i, image_id in enumerate(test):
        array = []
        for each in test[image_id]['Pred Sent']:
            array.append(test[image_id]['Pred Sent'][each])
        pred_sent = '. '.join(array)

        array = []
        for each in test[image_id]['Real Sent']:
            sent = test[image_id]['Real Sent'][each]
            if len(sent) != 0:
                array.append(sent)
        real_sent = '. '.join(array)
        datasetGTS['annotations'].append({
            'image_id': i,
            'caption': real_sent
        })
        datasetRES['annotations'].append({
            'image_id': i,
            'caption': pred_sent
        })

    # 生成记录指标的文件metric.txt
    writer = _init_writer(args)
    writer.write("{}\n".format(args))

    rng = range(len(test))
    metric = calculate_metrics(rng, datasetGTS, datasetRES)

    # print(metric)
    for i in metric.items():
        writer.write(''.join(str(i))+'\n')
        print(''.join(str(i)))
    writer.close()