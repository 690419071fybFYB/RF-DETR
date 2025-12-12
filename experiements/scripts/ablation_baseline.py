from rfdetr import RFDETRBase

# Experiment 1: Baseline (No Density Init, No SOQB)
# "Baseline" - pure RF-DETR without our custom modules
model = RFDETRBase(
    enable_density_init=False,
    density_loss_coef=0.0,
    enable_small_object_query_boost=False,
    soqb_boost_factor=2.0 
)

model.train(
    dataset_file='coco',
    dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
    coco_path='/home/fyb/datasets/RSOD_cocoFormat',
    epochs=60,
    batch_size=6,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/ablation_baseline_6bs',
    early_stop_patience=5,
    early_stop_threshold=0.001,
)
