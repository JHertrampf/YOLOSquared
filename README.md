# Motorcycle Helmet Detection 

The largest portion of the code in this repository originates from a project that was part of the 2023 edition of the "Deep Learning Lab" at the Computer Vision Chair [1] (Computer Science Chair 13) at RWTH Aachen University.

This repository includes a training and especially an inference script, for the detection of motorcycles and motorcycle helmets, using a two-stage approach inspired by [2]. In each stage a YOLOv8 model [3] is applied, first to detect motorcycles, and then to detect heads that do or do not wear motorcycle helmets in each motorcycle patch. Furthermore, motorcycles are tracked through subsequent frames using a tracking algorithm adapted from [4]. This enables the per motorcycle detection of traffic rule violations, either due to a too large number of passengers, or because at least one of the passengers does not wear a helmet. This project was motivated by the idea of constructing a system to improve traffic safety in regions were motorcycles are an especially prominent form of transportation.

## The Inference Architecture
![image](https://github.com/JHertrampf/YOLOSquared/assets/95368554/04bdd716-5b39-43f2-bc78-58feb194dc5f)

## Usage
<b>Inference</b>:
 - Place trained models into src/inference/stage1 and src/inference/stage2 respectively, and data into the data directory.
 - Consider src/inference/inference_config.py for the different modes, sources, image sizes, etc.
 - run 'python3 src/inference/inference.py'

<b>Training</b> yolo models: run 'python3 src/train/yolov8.py'

## Example Results
<img width="700" alt="no_helmet" src="https://github.com/JHertrampf/YOLOSquared/assets/95368554/0ff7c305-d82e-41f9-9059-4c4d0f19a2bb">
<img width="700" alt="helmet" src="https://github.com/JHertrampf/YOLOSquared/assets/95368554/7ad24547-a751-4ca3-85ed-8d3732039b63">

These Screenshots show the tracking mode, which indicates tracks and bounding boxes for the detected motorcycles, together with a label "violation" (with reason, e.g. H for no helmet) or "legal", depending on the number of detected passengers and helmets. Additionally, the currently achieved frames per second are displayed on the top right.


## Acknowledgements
We thank Tristan Höfer and Syed Ayaz, with whom we worked on the Deep Learning Lab project, which this repository originates from.

## References
[1] https://www.vision.rwth-aachen.de/

[2] Jia, W., Xu, S., Liang, Z., Zhao, Y., Min, H., Li, S., & Yu, Y. (2021). Real‐time automatic helmet detection of motorcyclists in urban traffic using improved YOLOv5 detector. IET Image Processing, 15(14), 3623-3637.

[3] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

[4] Wojke, N., Bewley, A., & Paulus, D. (2017, September). Simple online and realtime tracking with a deep association metric. In 2017 IEEE international conference on image processing (ICIP) (pp. 3645-3649). IEEE.
