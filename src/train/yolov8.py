import argparse
import os

from ultralytics import YOLO

import config


def main(lr0, lrf, optim):
    # Parameters
    data_path = os.path.join(config.DATASET_ROOT_DIR, "data.yaml")
    epochs = 20
    batch = 12
    imgsz = [1024, 576]
    save = True
    save_period = 5
    optimizer = optim
    pretrained = True
    lr0 = lr0
    lrf = lrf
    yolo_version = "l"
    name = f"yolov8{yolo_version}_reduced_{imgsz}_{optimizer}_lr0_{lr0}_lrf_{lrf * lr0}_bs_{batch}_ep_{epochs}"

    # Load the model
    model = YOLO(f"yolov8{yolo_version}.pt")

    # Training
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        save=save,
        optimizer=optimizer,
        save_period=save_period,
        project="YOLOv8",
        name=name,
        pretrained=pretrained,
        verbose=True,
        lr0=lr0,
        lrf=lrf
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lr0", type=float, help="Initial learning rate")
    parser.add_argument("lrf", type=float, help="Final learning rate")
    parser.add_argument("optim", type=str, help="Optimization algorithm")
    args = parser.parse_args()

    main(args.lr0, args.lrf, args.optim)
