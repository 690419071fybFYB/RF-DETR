from rfdetr import RFDETRBase

# Experiment 4: Combined (Density Init + SOQB)
# Test if the combination yields better results than individual parts
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=1.0,
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
    output_dir='/home/fyb/mydir/rf-detr/experiements/results/ablation_combined',
)
