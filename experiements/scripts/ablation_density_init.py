from rfdetr import RFDETRBase
from torchinfo import summary
# Experiment 2: Density Init Only (No SOQB)
# This was our previous "Best Baseline"
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=1.0,
    enable_small_object_query_boost=False,
    soqb_boost_factor=2.0 
)
summary(model.model.model,input_size=(1,3,560,560),depth=7)
# model.train(
#     dataset_file='coco',
#     dataset_dir='/home/fyb/datasets/RSOD_cocoFormat',
#     coco_path='/home/fyb/datasets/RSOD_cocoFormat',
#     epochs=12,
#     batch_size=4,
#     grad_accum_steps=4,
#     lr=1e-4,
#     output_dir='/home/fyb/mydir/rf-detr/experiements/results/ablation_density_init_debug',
# )
