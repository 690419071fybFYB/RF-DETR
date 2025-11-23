from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator("/home/fyb/mydir/rf-detr/script/DIOR_RF_CSD_SOQB_FPN")
ea.Reload()
print("Scalars:", ea.Tags().get("scalars", []))
