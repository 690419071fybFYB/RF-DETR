
import time
from rfdetr.util.misc import MetricLogger, SmoothedValue

print("Testing MetricLogger with tqdm...")
logger = MetricLogger(delimiter="  ")
logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

data_loader = range(10)

print("Starting loop...")
for i in logger.log_every(data_loader, print_freq=2, header="Test Epoch"):
    time.sleep(0.1)
    logger.update(loss=0.5)

print("\nLoop finished.")
