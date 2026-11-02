# Data and model notices

## Pretrained encoder

`facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, released under Apache-2.0 by Meta. The distributed ONNX encoder converts the pretrained weights to dynamic per-channel INT8 matrix weights; it is not newly trained. See `MODEL_LICENSE.txt` and `reports/export.json`.

## Dataset and example images

MVTec AD, MVTec Software GmbH. Original publication: Paul Bergmann, Michael Fauser, David Sattlegger and Carsten Steger, “MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019. [Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [Paper](https://doi.org/10.1109/CVPR.2019.00982).

The dataset is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). `web/samples/` and `web/references/` contain resized examples from bottle, metal-nut and zipper categories; reference previews use JPEG compression. Dataset-derived images and feature banks (`web/data/*.f32`, `artifacts/bottle.npz`, `artifacts/metal_nut.npz`, `artifacts/zipper.npz`) retain those terms, separate from the application-code MIT license. Source paths are recorded in `web/samples.json` and `web/data/*.json`. The full dataset is not redistributed.

The screenshot shows those examples inside the application. Do not use the data-derived assets for commercial inspection without the appropriate dataset permission.
