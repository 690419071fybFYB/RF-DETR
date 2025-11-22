from rfdetr import RFDETRBase
from torchinfo import summary

rf = RFDETRBase()
summary(rf.model.model, input_size=(1, 3, 560, 560),depth=7)
