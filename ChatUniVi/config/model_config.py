model_config_pretune = {
    "use_cluster": True,
    "freeze": False,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}

model_config_finetune = {
    "use_cluster": True,
    "freeze": False,
    "mm_tune": True,
    "vision_tune": False,

    "spatial_cluster_rate0": 64,  # 0.25
    "spatial_cluster_rate1": 32,  # 0.5
    "spatial_cluster_rate2": 16,  # 0.5

    "temporal_cluster_rate": 1/16,
}