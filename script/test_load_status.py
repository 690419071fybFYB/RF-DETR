from rfdetr import RFDETRBase
from torchinfo import summary
model = RFDETRBase(
    enable_density_init=True,
    density_loss_coef=1.0,
    enable_small_object_query_boost=True, # Example: Disable boost
    soqb_boost_factor=2.0 
)
summary(model.model.model, input_size=(1, 3, 560, 560),depth=7)
