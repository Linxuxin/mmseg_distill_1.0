_base_ = [
    '../../pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    init_student = False,
    use_logit = True,
    use_adv = False,
    num_classes = 19,
    distill_cfg = [ dict(methods=[dict(type='InterClassLoss',
                                       name='loss_idd_f4',
                                       loss_weight=0.1,
                                       distance_type='cross_sim',
                                       student_channels=512,
                                       teacher_channels=2048,
                                       tau=0.1,
                                       decoupled=True,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='InterClassLoss',
                                       name='loss_idd_f3',
                                       loss_weight=1.0,
                                       distance_type='cross_sim',
                                       student_channels=256,
                                       teacher_channels=1024,
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
                                       student_channels=512,
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
                                       loss_weight=10.0,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='CriterionAdvForG',
                                       name='loss_adv_g',
                                       adv_type='hinge',
                                       loss_weight=0.001,
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

# data = dict(
#     test=dict(
#         img_dir='leftImg8bit/test',
#         ann_dir='gtFine/test'))

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

# optimizer = dict(
#     D_model=dict(type='Adam',
#                   lr=0.0004,
#                   betas=(0.9, 0.99)),
#     model=dict(type='SGD',
#                   lr=0.01,
#                   momentum=0.9,
#                   weight_decay=0.0005),
# )

student_cfg = 'configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
