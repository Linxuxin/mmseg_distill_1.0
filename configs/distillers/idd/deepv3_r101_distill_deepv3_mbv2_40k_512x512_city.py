_base_ = [
    '../../deeplabv3/deeplabv3_m-v2-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth',
    init_student = False,
    use_logit = True,
    num_classes = 19,
    distill_cfg = [ dict(methods=[dict(type='InterClassLoss',
                                       name='loss_idd_f3',
                                       loss_weight=1.0,
                                       distance_type='cross_sim',
                                       student_channels=96,
                                       teacher_channels=1024,
                                       tau=0.1,
                                       decoupled=True,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='InterClassLoss',
                                       name='loss_idd_ap',
                                       loss_weight=1.0,
                                       distance_type='cross_sim',
                                       student_channels=128,
                                       teacher_channels=512,
                                       tau=0.1,
                                       decoupled=True,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='ContrastLoss',
                                       name='loss_cl',
                                       student_channels=512,
                                       teacher_channels=2048,
                                       loss_weight=0.5,
                                       # max_views=16384,
                                       # max_samples=16484,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='CIRKDLoss',
                                       name='loss_cirkd',
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='CIRKDMemLoss',
                                       name='loss_cirkd_mem',
                                       student_channels=128,
                                       teacher_channels=512,
                                       loss_weight=1.0,
                                       # neg_sample_mode='semi_random',
                                       # anchor_sample_mode='prob'
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='ContrastMemLoss',
                                       name='loss_cl_mem',
                                       student_channels=512,
                                       teacher_channels=2048,
                                       loss_weight=0.1,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fea1',
                                       student_channels=320,
                                       teacher_channels=2048,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='DecoupledKD',
                                       name='loss_dkd_pixel',
                                       tau=2.0,
                                       loss_weight1=2.0,
                                       loss_weight2=2.0,
                                       dim=0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DecoupledKD',
                                       name='loss_dkd_channel',
                                       tau=4.0,
                                       loss_weight1=4.0,
                                       loss_weight2=4.0,
                                       dim=1,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='IFVDLoss',
                                       name='loss_ifvd',
                                       loss_weight=200.0,
                                       student_channels=128,
                                       teacher_channels=512,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='DISTLoss',
                                       name='loss_dist',
                                       beta=2.0,
                                       gamma=2.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='PixelWiseLoss',
                                       name='loss_pi',
                                       loss_weight=1.0,
                                       )
                                ]
                        ),
                    # dict(methods=[dict(type='DecoupledKD',
                    #                    name='loss_dkd_features',
                    #                    tau=4.0,
                    #                    loss_weight1=4.0,
                    #                    loss_weight2=4.0,
                    #                    features=True,
                    #                    student_channels=512,
                    #                    teacher_channels=2048,
                    #                    loss_type='kl',
                    #                    # merge_label=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 10], [11, 12], [13, 16], [17,18]],
                    #                    )
                    #             ]
                    #     ),
                   ]
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

student_cfg = 'configs/deeplabv3/deeplabv3_m-v2-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/deeplabv3/deeplabv3_r101-d8_512x1024_80k_cityscapes.py'
