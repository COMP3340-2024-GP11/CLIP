default_scope = 'mmpretrain'
# checkpoint saving
default_hooks = dict( 
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'), 

    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=100),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=True,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=32),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]