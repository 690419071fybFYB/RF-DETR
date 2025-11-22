from rfdetr import RFDETRBase
from torchinfo import summary
model = RFDETRBase()
summary(model.model.model, input_size=(1, 3, 560, 560),depth=3)


