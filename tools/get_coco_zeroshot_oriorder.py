# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/vindr/zero-shot/instances_val_all_2.json')
    parser.add_argument('--cat_path', default='datasets/vindr/annotations/train_instances_coco.json')
    args = parser.parse_args()
    print('Loading', args.cat_path)
    cat = json.load(open(args.cat_path, 'r'))['categories']

    print('Loading', args.data_path)
    data = json.load(open(args.data_path, 'r'))
    data['categories'] = cat
    out_path = args.data_path[:-5] + '_oriorder.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))
