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


def _init_logger(args):
    logger = Logger(os.path.join(args.log_dir, 'tb_metric'))
    return logger


def _init_writer(args):
    writer = open(os.path.join(args.log_dir, args.metric_name), 'w')
    return writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--result_path', type=str,
    #                     default='./results/clean_test.json')
    parser.add_argument('--result_name', type=str,
                        default='.')
    parser.add_argument('--log_dir', type=str,
                        default='./result_logs/0418-2309densenet201')
    parser.add_argument('--result_dir', type=str,
                        default='/results')

    parser.add_argument('--caption_name', type=str,
                        default='debug.json')
    parser.add_argument('--metric_name', type=str,
                        default='metric.txt')
    args = parser.parse_args()
    print(args)

    logger = _init_logger(args)

    writer = _init_writer(args)
    writer.write("{}\n".format(args))

    result_path = args.log_dir + args.result_dir  # './result_logs/0418-2309densenet201/results'
    result_list = os.listdir(result_path)  # [].
    result_list.sort(key=lambda x:int(x[:-5]))
    for i, result_name in enumerate(result_list):
        args.result_name = os.path.join(result_path, result_name)
        test = load_json(args.result_name)

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

        rng = range(len(test))
        metric = calculate_metrics(rng, datasetGTS, datasetRES)

        for tag, value in metric.items():
            logger.scalar_summary(tag, value, int(result_name[:-5]))
        writer.write("[ Epoch {}]\tBleu1:{:.4f} Bleu2:{:.4f} Bleu3:{:.4f} Bleu4:{:.4f}\t\t"
                     "METEOR:{:.4f}\tROUGE_L:{:.4f}\tCIDEr:{:.4f}\n".format(result_name[:-5],
                                                                            metric['Bleu_1'],
                                                                            metric['Bleu_2'],
                                                                            metric['Bleu_3'],
                                                                            metric['Bleu_4'],
                                                                            metric['METEOR'],
                                                                            metric['ROUGE_L'],
                                                                            metric['CIDEr'],
                                                                            ))
        print("[ Epoch {}]\tBleu1:{:.4f} Bleu2:{:.4f} Bleu3:{:.4f} Bleu4:{:.4f}\t\t"
                     "METEOR:{:.4f}\tROUGE_L:{:.4f}\tCIDEr:{:.4f}\n".format(result_name[:-5],
                                                                            metric['Bleu_1'],
                                                                            metric['Bleu_2'],
                                                                            metric['Bleu_3'],
                                                                            metric['Bleu_4'],
                                                                            metric['METEOR'],
                                                                            metric['ROUGE_L'],
                                                                            metric['CIDEr'],
                                                                            ))
    writer.close()
