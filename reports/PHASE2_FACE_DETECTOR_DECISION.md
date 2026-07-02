# Phase 2 Face Detector Decision

## Local Environment

- Python: 3.11.9
- TensorFlow: 2.20.0
- Keras: 3.10.0
- OpenCV: 4.10.0.84, installed
- MediaPipe: not installed
- MTCNN: not installed
- RetinaFace: not installed
- GPU available to TensorFlow in this environment: no

## Options Reviewed

| Detector | Installation compatibility | CPU speed | Accuracy and small faces | Multiple faces | Model size / download | Browser-extension path | Licensing notes | Phase 2 fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenCV YuNet | OpenCV API is present, but no YuNet ONNX file is present locally. OpenCV documents `FaceDetectorYN` as a DNN face detector with score and NMS thresholds. | Good on CPU. | Better than Haar and can detect roughly 10x10 to 300x300 faces according to the OpenCV Zoo README. | Yes. | Requires adding an ONNX model file. | Reasonable later via OpenCV.js or a small ONNX runtime path, but still an added asset. | OpenCV Zoo model includes its own license file. | Best long-term OpenCV option, but not zero-change in this environment. |
| MediaPipe Face Detector | Not installed locally. Official docs support Python, Web, Android, and iOS. | Official Pixel 6 benchmark lists short-range BlazeFace at 2.94 ms CPU. | Good selfie/webcam detector; full-range model exists. | Yes. | Requires installing MediaPipe and model/task assets. | Strong browser story. | Google docs state code samples are Apache 2.0, content CC BY 4.0. | Attractive for later extension, but it would change the current environment. |
| MTCNN | Not installed locally. PyPI lists Python >=3.10 and TensorFlow >=2.12, so it is probably compatible with Python 3.11/TensorFlow 2.20. | Slower than lightweight OpenCV/BlazeFace options because it cascades three neural nets. | Robust for frontal faces and landmarks. | Yes. | Requires new package install; PyPI wheel is about 1.9 MB. | Weak browser-extension story. | MIT on PyPI. | Good research detector, not the lightest deployment path. |
| RetinaFace | Not installed locally. PyPI package is TensorFlow-based. | Heavier than Haar/YuNet/BlazeFace. | Strong crowd and difficult-face performance. | Yes. | Package is small but typically downloads/uses pretrained weights. | Weak browser-extension story. | MIT on PyPI. | Accuracy is appealing, but too heavy for this MVP phase. |

## Selected Detector

Selected for the Phase 2 Python prototype: **OpenCV Haar cascade `haarcascade_frontalface_default.xml` as the local OpenCV fallback**.

The preferred OpenCV-family detector for a production follow-up is YuNet, but this phase avoids downloading a new model asset or changing the working audit environment. The Haar cascade ships with the already-installed `opencv-python` package, runs on CPU, supports multiple frontal faces, exposes bounding boxes, and is enough to test whether automatic face cropping destabilizes the existing deepfake classifier.

This choice is intentionally conservative. Haar confidence is not calibrated; the implementation exposes OpenCV's cascade reject weight as a normalized detector ranking signal and documents that limitation in code and reports.

## Sources Checked

- OpenCV `FaceDetectorYN` documentation: https://docs.opencv.org/4.x/df/d20/classcv_1_1FaceDetectorYN.html
- OpenCV Zoo YuNet README: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- MediaPipe Face Detector documentation: https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
- MTCNN PyPI package page: https://pypi.org/project/mtcnn/
- RetinaFace PyPI package page: https://pypi.org/project/retina-face/
