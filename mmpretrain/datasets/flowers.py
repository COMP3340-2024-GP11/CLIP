# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union
import os
import numpy as np

from mmengine import fileio
from mmengine.logging import MMLogger

from mmpretrain.registry import DATASETS
from .base_dataset import BaseDataset
from .custom import CustomDataset


@DATASETS.register_module()
class Flowers(CustomDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    METAINFO = {'classes': [
        'daffodil', 'snowdrop', 'lilyValley', 'bluebell', 'crocys', 'iris', 'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'colts foot', 'dandelion', 'cowslip', 'buttercup', 'wind flower', 'pansy']}

    def __init__(self,
                 data_root: str = '',
                 split: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 **kwargs):
        kwargs = {'extensions': self.IMG_EXTENSIONS, **kwargs}

        if split:
            splits = ['train', 'val', 'test']
            assert split in splits, \
                f"The split must be one of {splits}, but get '{split}'"

            data_prefix = split if data_prefix == '' else data_prefix

            if ann_file == '':
                _ann_path = fileio.join_path(data_root, 'meta', f'{split}.txt')
                if fileio.exists(_ann_path):
                    ann_file = fileio.join_path('meta', f'{split}.txt')

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            **kwargs)

    def extra_repr(self) -> List[str]:
        """The extra repr information of the dataset."""
        body = [
            f'Root of dataset: \t{self.data_root}',
        ]
        return body