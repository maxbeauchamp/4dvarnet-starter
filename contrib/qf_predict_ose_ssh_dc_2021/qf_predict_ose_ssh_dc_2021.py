import hydra_zen
import numpy as np
import pandas as pd
import qf_pipeline
import qf_hydra_recipes
import qf_predict_4dvarnet_starter
import qf_merge_patches
import qf_download_altimetry_constellation
import qf_alongtrack_metrics_from_map

b = hydra_zen.make_custom_builds_fn()

grid_cfg = qf_hydra_recipes.grid_recipe(params=dict(
    dict(
        input_path="data/prepared/inference_combined.nc",
        grid=dict(
            time=b(pd.date_range, start='${......params.min_time}', end='${......params.max_time}', freq="1D"),
            lat=b(np.arange, start='${......params.min_lat}', stop='${......params.max_lat}', step=0.05),
            lon=b(np.arange, start='${......params.min_lon}', stop='${......params.max_lon}', step=0.05),
        ),
        output_path="data/prepared/gridded.nc",
    )
))

s3cfg = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.config_path}', local_path='model_xp/config.yaml'))
s3ckpt = qf_hydra_recipes.get_s3_recipe(params=dict(remote_path='${....params.checkpoint_path}', local_path='model_xp/checkpoint.ckpt'))

predict_cfg = qf_predict_4dvarnet_starter.recipe(
        input_path= '${.._04_grid.params.output_path}',
        output_dir= 'data/inference/batches',
        params = dict(
            config_path="model_xp/config.yaml",
            ckpt_path="model_xp/my_checkpoint.ckpt",
            strides='${....params.strides}',
            check_full_scan=True,
        ),
    )
merge_cfg = qf_merge_patches.recipe(
    input_directory="data/inference/batches",
    output_path="method_outputs/merged_batches.nc",
    weight=b(qf_merge_patches.build_weight, patch_dims='${..._05_predict.patcher.patches}'),
    out_coords='${.._04_grid.params.grid}',
)

stages = {
    "_01_fetch_inference_data": qf_download_altimetry_constellation.recipe(),
    "_02_fetch_config": s3cfg,
    "_03_fetch_checkpoint": s3ckpt,
    "_04_grid": grid_cfg,
    "_05_predict": predict_cfg,
    "_06_merge_batches": merge_cfg,
    "_07_compute_metrics": qf_alongtrack_metrics_from_map.recipe(),
}



b = hydra_zen.make_custom_builds_fn(populate_full_signature=True)
params = dict(
    all_sats=["alg", "h2ag", "j2g", "j2n", "j3", "s3a"],
    min_time="2016-12-01",
    max_time="2018-02-01",
    min_lon=-66.0,
    max_lon=-54.0,
    min_lat=32.0,
    max_lat=44.0,
    strides={},
    config_path='melody/quentin_cloud/starter_jobs/new_baseline/.hydra/config.yaml',
    checkpoint_path='melody/quentin_cloud/starter_jobs/new_baseline/base/checkpoints/val_mse=3.01245-epoch=551.ckpt',
)


inference_data_pipeline, recipe = qf_pipeline.register_pipeline(
    "dc_ose_2021_4dvarnet", stages=stages, params=params
)
