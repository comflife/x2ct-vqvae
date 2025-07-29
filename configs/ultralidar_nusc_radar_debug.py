# Test configuration with minimal workers to debug data loading
# Copy from ultralidar_nusc_radar.py but with debug settings

model_type = "codebook_training"
_base_ = ["./_base_/default_runtime.py"]
batch_size = 3  # Reduced batch size for debugging
point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
voxel_size = [0.15625, 0.15625, 0.2]
class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", 
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
plugin = True
plugin_dir = "plugin/"
num_points = 30

model = dict(
    type="UltraLiDAR",
    model_type=model_type,
    pts_bbox_head=dict(
        type="CenterHead",
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type="CenterPointBBoxCoder",
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9,
        ),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        norm_bbox=True,
    ),
    voxelizer=dict(
        type="Voxelizer",
        x_min=point_cloud_range[0],
        x_max=point_cloud_range[3],
        y_min=point_cloud_range[1],
        y_max=point_cloud_range[4],
        z_min=point_cloud_range[2],
        z_max=point_cloud_range[5],
        step=voxel_size[0],
        z_step=voxel_size[2],
    ),
    vector_quantizer=dict(
        type="VectorQuantizer",
        n_e=1024,
        e_dim=1024,
        beta=0.25,
        cosine_similarity=False,
    ),
    lidar_encoder=dict(
        type="VQEncoder",
        img_size=640,
        codebook_dim=1024,
    ),
    lidar_decoder=dict(
        type="VQDecoder",
        img_size=(640, 640),
        num_patches=6400,
        codebook_dim=1024,
    ),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    ),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            nms_type=["rotate", "rotate", "rotate", "circle", "rotate", "rotate"],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]],
        )
    ),
)

# Data configuration with minimal workers for debugging
dataset_type = 'NuscDataset'  # Back to custom dataset that handles the annotation format
data_root = "/data1/nuScenes/"
file_client_args = dict(backend="disk")

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 4, 5],
        file_client_args=file_client_args,
    ),
    # TEMPORARILY REMOVE ANNOTATIONS TO FIX INFINITE LOOP
    # We'll add them back once the basic pipeline works
    # dict(
    #     type='LoadAnnotations3D',
    #     with_bbox_3d=True,
    #     with_label_3d=True,
    #     file_client_args=file_client_args,
    # ),
    dict(type='DefaultFormatBundle3D', class_names=class_names),  # Add back formatting
    dict(type='Collect3D', keys=['points']),  # Only points for now
]

test_pipeline = train_pipeline

input_modality = dict(use_lidar=False, use_camera=False, use_radar=True, use_map=False, use_external=False)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=1,  # Reduced from 8 to 1 for debugging
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train_radar.pkl',  # Use preprocessed radar annotations
        # ann_file=data_root + 'nuscenes_infos_val_radar_tiny.pkl',  # Use tiny dataset for debugging
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,  # Changed to True to skip annotation requirements
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_radar.pkl',  # Use preprocessed radar annotations
        # ann_file=data_root + 'nuscenes_infos_val_radar_tiny.pkl',  # Use preprocessed radar annotations
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val_radar.pkl',  # Use preprocessed radar annotations
        # ann_file=data_root + 'nuscenes_infos_val_radar_tiny.pkl',  # Use preprocessed radar annotations
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

# Simplified optimizer
optimizer = dict(
    type="AdamW",
    lr=6e-4,
    betas=(0.9, 0.95),
    weight_decay=0.0001,
)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=100, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3)
runner = dict(type="EpochBasedRunner", max_epochs=150)  # Only 150 epochs for testing

checkpoint = None
work_dir = "./work_dirs/nusc_radar_debug"
find_unused_parameters = True

# Simplified logging to avoid TensorBoard issues
log_config = dict(
    interval=1,  # Log every iteration
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

# Remove custom hooks that might cause issues
# custom_hooks = [
#     dict(type='NumClassCheckHook'),
#     dict(type='IterTimerHook'),
#     dict(
#         type='CheckpointHook',
#         interval=1,
#         by_epoch=True,
#         save_optimizer=True,
#         out_dir=work_dir),
# ]

# Simple evaluation configuration
# evaluation = dict(
#     interval=1,
#     pipeline=test_pipeline,
#     save_best='mAP',
#     rule='greater')

# Basic configuration
load_from = None
resume_from = None
workflow = [('train', 1)]
