# dataset settings
dataset_type = 'Flowers'
data_preprocessor = dict(
    num_classes=17,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/flowers',
        split='train',
        ann_file='meta/train.txt',
        with_label=True,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/flowers',
        ann_file='meta/val.txt',
        split='val',
        with_label=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = dict(
    batch_size=32,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root='data/flower',
        ann_file='meta/test.txt',
        split='test',
        with_label=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_evaluator = val_evaluator
