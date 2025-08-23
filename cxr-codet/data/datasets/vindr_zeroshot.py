# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .lvis_v1 import custom_register_lvis_instances
from detectron2.data import DatasetCatalog

categories_seen = [
        {'id': 0, 'name': 'Aortic enlargement'},
        {'id': 1, 'name': 'Atelectasis'},
        {'id': 2, 'name': 'Calcification'},
        {'id': 3, 'name': 'Cardiomegaly'},
        {'id': 4, 'name': 'Consolidation'},
        {'id': 5, 'name': 'ILD'},
        {'id': 6, 'name': 'Infiltration'},
        {'id': 7, 'name': 'Lung Opacity'},
        {'id': 8, 'name': 'Nodule/Mass'},
        {'id': 9, 'name': 'Other lesion'},
        {'id': 10, 'name': 'Pleural effusion'},
        {'id': 11, 'name': 'Pleural thickening'},
        {'id': 12, 'name': 'Pneumothorax'},
        {'id': 13, 'name': 'Pulmonary fibrosis'},
        {'id': 14, 'name': 'No finding'}
]

categories_unseen = [
        {'id': 0, 'name': 'Aortic enlargement'},
        {'id': 1, 'name': 'Atelectasis'},
        {'id': 2, 'name': 'Calcification'},
        {'id': 3, 'name': 'Cardiomegaly'},
        {'id': 4, 'name': 'Consolidation'},
        {'id': 5, 'name': 'ILD'},
        {'id': 6, 'name': 'Infiltration'},
        {'id': 7, 'name': 'Lung Opacity'},
        {'id': 8, 'name': 'Nodule/Mass'},
        {'id': 9, 'name': 'Other lesion'},
        {'id': 10, 'name': 'Pleural effusion'},
        {'id': 11, 'name': 'Pleural thickening'},
        {'id': 12, 'name': 'Pneumothorax'},
        {'id': 13, 'name': 'Pulmonary fibrosis'},
        {'id': 14, 'name': 'No finding'}
]



def _get_metadata(cat):
    if cat == 'all':
        return _get_builtin_metadata('vindr')
    elif cat == 'seen':
        id_to_name = {x['id']: x['name'] for x in categories_seen}
    else:
        assert cat == 'unseen'
        id_to_name = {x['id']: x['name'] for x in categories_unseen}

    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}



_PREDEFINED_SPLITS_VINDR = {
    "vindr_zeroshot_train": ("vindr/images", "vindr/zero-shot/instances_train_seen_2.json", 'seen'),
    "vindr_zeroshot_val": ("vindr/images", "vindr/zero-shot/instances_val_unseen_2.json", 'unseen'),
    "vindr_not_zeroshot_val": ("vindr/images", "vindr/zero-shot/instances_val_seen_2.json", 'seen'),
    "vindr_generalized_zeroshot_val": ("vindr/images", "vindr/zero-shot/instances_val_all_2_oriorder.json", 'all'),
    "vindr_zeroshot_train_oriorder": ("vindr/images", "vindr/zero-shot/instances_train_seen_2_oriorder.json", 'all'),
}

for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_VINDR.items():
    register_coco_instances(
        key,
        _get_metadata(cat),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )

_CUSTOM_SPLITS_COCO = {
    "vindr_caption_train_tags": ("vindr/images/", "vindr/annotations/train_captions_coco.json"),
    "vindr_caption_train_tags_634": ("vindr/images/", "vindr/annotations/vinbigdata_nested_captions.json")
    }
for key, (image_root, json_file) in _CUSTOM_SPLITS_COCO.items():
    custom_register_lvis_instances(
        key,
        _get_builtin_metadata('vindr'),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )

if 'vindr_zeroshot_train' in DatasetCatalog.list():
    print("-----Worked-----")
else:
    print("-----Failed------")



