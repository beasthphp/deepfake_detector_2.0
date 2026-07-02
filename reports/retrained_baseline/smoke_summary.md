# Smoke Baseline Training Summary

- Mode: `smoke`
- Seed: 69
- Batch size: 32
- Epochs completed: 1
- Augmentation enabled: `True`
- Full train counts: `{'fake': 50000, 'real': 50000}`
- Full validation counts: `{'fake': 10000, 'real': 10000}`
- Used train counts: `{'fake': 500, 'real': 500}`
- Used validation counts: `{'fake': 100, 'real': 100}`
- Train batch: `{'image_shape': [32, 256, 256, 3], 'label_shape': [32], 'label_distribution': {'0': 12, '1': 20}, 'image_dtype': 'float32', 'image_min': 0.0, 'image_max': 1.0}`
- Validation batch: `{'image_shape': [32, 256, 256, 3], 'label_shape': [32], 'label_distribution': {'0': 18, '1': 14}, 'image_dtype': 'float32', 'image_min': 0.0, 'image_max': 1.0}`
- Checkpoint path: `D:\ddp\models\experiments\retrained_custom_cnn_smoke_best.keras`
- Best validation epoch: `1`
- Best validation ROC-AUC: `0.6857000589370728`
- Elapsed seconds: `32.43`

## Final Epoch Metrics

- accuracy: 0.484000
- auc: 0.480808
- loss: 0.935248
- precision: 0.479381
- recall: 0.372000
- val_accuracy: 0.570000
- val_auc: 0.685700
- val_loss: 0.690659
- val_precision: 0.750000
- val_recall: 0.210000
- learning_rate: 0.001000

## Model Summary

```text
Model: "retrained_custom_cnn"
+--------------------------------------------------------------------------+
| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------+------------------------+---------------|
| conv2d (Conv2D)                 | (None, 254, 254, 32)   |           896 |
|---------------------------------+------------------------+---------------|
| max_pooling2d (MaxPooling2D)    | (None, 127, 127, 32)   |             0 |
|---------------------------------+------------------------+---------------|
| conv2d_1 (Conv2D)               | (None, 125, 125, 64)   |        18,496 |
|---------------------------------+------------------------+---------------|
| max_pooling2d_1 (MaxPooling2D)  | (None, 62, 62, 64)     |             0 |
|---------------------------------+------------------------+---------------|
| conv2d_2 (Conv2D)               | (None, 60, 60, 128)    |        73,856 |
|---------------------------------+------------------------+---------------|
| max_pooling2d_2 (MaxPooling2D)  | (None, 30, 30, 128)    |             0 |
|---------------------------------+------------------------+---------------|
| flatten (Flatten)               | (None, 115200)         |             0 |
|---------------------------------+------------------------+---------------|
| dense (Dense)                   | (None, 128)            |    14,745,728 |
|---------------------------------+------------------------+---------------|
| dropout (Dropout)               | (None, 128)            |             0 |
|---------------------------------+------------------------+---------------|
| dense_1 (Dense)                 | (None, 1)              |           129 |
+--------------------------------------------------------------------------+
 Total params: 14,839,105 (56.61 MB)
 Trainable params: 14,839,105 (56.61 MB)
 Non-trainable params: 0 (0.00 B)
```
