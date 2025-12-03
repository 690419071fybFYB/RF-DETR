from rfdetr import RFDETRBase
import torch
from torchinfo import summary
# Instantiate the model
model_wrapper = RFDETRBase()
model = model_wrapper.model.model

# Print the model structure
summary(model, (1, 3, 560, 560))

