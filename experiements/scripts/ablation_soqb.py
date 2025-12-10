from rfdetr import RFDETRBase

# Experiment 3: SOQB Only (No Density Init)
# Test if boosting query features works without density guidance
model = RFDETRBase(
    enable_density_init=False,
    density_loss_coef=0.0,
    enable_small_object_query_boost=True,
    soqb_boost_factor=2.0 
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=12,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/ablation_soqb',
)
