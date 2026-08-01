# AgroFedVision Architecture

AgroFedVision is designed as a modular AI platform for precision agriculture, not as a single disease classifier. The platform accepts whichever agricultural signals are available, routes them to independent AI modules, and merges their outputs into a complete agricultural report.

## Design Goals

- Work with partial inputs: image-only, sensor-only, UAV-only, or any combination.
- Avoid hardcoded crop names, datasets, and model assumptions.
- Keep every AI capability independently trainable, evaluable, deployable, and replaceable.
- Make centralized training production-ready before adding XAI, federated learning, edge deployment, fusion, and web features.
- Preserve training progress across Google Colab disconnects, GPU limits, and runtime restarts.

## Folder Structure

```text
AgroFedVision/
  agrofedvision/
    api/                    FastAPI routes, auth, prediction endpoints
    datasets/               Dataset registries, manifests, tf.data builders
    deployment/             Cloud packaging and model release utilities
    edge/                   TFLite, ONNX, quantization, device profiles
    evaluation/             Metrics, reports, plots, validation utilities
    federated/              Flower clients, servers, FL strategies
    fusion/                 Decision fusion contracts and engines
    models/                 Model builders and modality-specific encoders
    training/               Training loops, callbacks, resume orchestration
    utils/                  Reproducibility, paths, JSON, environment helpers
    xai/                    Grad-CAM, SHAP, Integrated Gradients
  frontend/                 React app, dashboards, model management UI
  mobile/                   Future mobile client
  train.py                  Phase 1 centralized training launcher
  ARCHITECTURE.md           Platform architecture
```

Only Phase 1 is implemented now. Future-phase directories are package boundaries so new work can be added without rewriting centralized training.

## Module Responsibilities

`datasets/` owns dataset discovery, manifest loading, label encoding, fold creation, and `tf.data` pipelines. It exposes generic records such as image path and label instead of crop-specific classes.

`models/` owns model construction. Phase 1 provides an EfficientNetB0 image classifier. Later modules can add sensor encoders, UAV encoders, satellite encoders, and multimodal heads behind the same builder pattern.

`training/` owns staged training, Keras compilation, callbacks, checkpoint recovery, mixed precision, and resume state. It does not know crop names.

`evaluation/` owns metrics, classification reports, confusion matrices, and training plots.

`fusion/` will combine independent predictions from image, sensor, UAV, weather, and satellite modules into one agricultural decision report.

`xai/` will generate model explanations such as Grad-CAM, Grad-CAM++, SHAP, and Integrated Gradients.

`federated/` will adapt trained modules into Flower clients and servers with FedAvg, FedProx, checkpoint recovery, client resume, and failure handling.

`edge/` will export trained models to TensorFlow Lite and ONNX, then apply quantization profiles for Raspberry Pi, Jetson Nano, and ESP32-class devices.

`api/` and `frontend/` will become the web platform: authentication, dashboards, model management, prediction history, and reporting.

## Data Flow

1. A user declares available data types, or the backend detects them from uploaded payloads.
2. The input router creates a modality manifest: image, sensor, UAV, weather, multispectral, or future satellite data.
3. Each available modality runs through its independent model pipeline.
4. Each module emits structured predictions with confidence, quality signals, and optional explanations.
5. The Decision Fusion Engine merges outputs into a final report.
6. The report is stored for dashboards, auditing, future training, and model monitoring.

Phase 1 implements the image training path:

```text
CSV manifest
  -> stratified folds
  -> tf.data decode/resize/cache/shuffle/batch/prefetch
  -> augmentation
  -> EfficientNet preprocess_input once
  -> EfficientNetB0
  -> classifier head
  -> fold metrics and artifacts
```

## Phase 1 Training Architecture

Phase 1 uses TensorFlow 2.20+, Keras 3, mixed precision, EfficientNetB0 ImageNet weights, and 5-fold stratified cross-validation.

The model graph is:

```text
Input 260x260x3
  -> RandomFlip
  -> RandomRotation
  -> RandomZoom
  -> RandomContrast
  -> RandomBrightness
  -> EfficientNet preprocess_input
  -> EfficientNetB0 backbone
  -> GlobalAveragePooling2D
  -> BatchNormalization
  -> Dropout
  -> Dense(256)
  -> BatchNormalization
  -> Dropout
  -> Dense(128)
  -> BatchNormalization
  -> Dropout
  -> Dense(10, softmax)
```

Stage 1 freezes the EfficientNet backbone and trains the classifier for 25 epochs with Adam at `3e-4`.

Stage 2 unfreezes the backbone, keeps every BatchNormalization layer frozen, and fine-tunes only the final meaningful EfficientNet blocks using Adam at `1e-5`.

## Resume Strategy

Training writes all artifacts under:

```text
results/
  models/
  checkpoints/
  backup/
  history/
  logs/
  metrics/
  plots/
  summary/
  resume_state.json
```

Every fold and stage has separate files. Keras `.keras` files store model architecture, weights, and optimizer state. `BackupAndRestore` protects in-progress `fit()` calls, and `resume_state.json` records current fold, current stage, and completed epochs. Running `python train.py` again resumes at the correct fold, stage, and epoch.

## Scalability

New crops are added by registering datasets or manifests, not by changing model code. New model families are added under `models/` and selected by configuration. New input combinations are handled by the future router and fusion contracts rather than a growing set of hardcoded conditional scripts.

For large training workloads, the same package can be extended with distributed strategies, sharded datasets, cloud object storage, and experiment tracking. For deployment, trained Keras models become the source artifacts for API serving, edge conversion, federated initialization, and XAI.

## Bottlenecks And Improvements

- `cache()` can exceed RAM for very large datasets. The pipeline allows disabling cache with `--no-cache`; a future improvement is file-backed cache per fold.
- EfficientNetB0 is a strong baseline but may underfit difficult multispectral or UAV tasks. Future model registry entries should include EfficientNetV2, ConvNeXt, ViT, and task-specific UAV encoders.
- Cross-validation multiplies training cost by five. For rapid experiments, add a development profile with fewer folds and epochs.
- CSV manifests should eventually become versioned dataset registry entries with checksums and schema validation.
- Model metrics should later be persisted to MongoDB or an experiment tracker for dashboard comparison.
- XAI generation should run asynchronously because Grad-CAM and SHAP can add significant latency.

## Recommended Roadmap

1. Complete Phase 1 centralized image training with fault-tolerant checkpoints.
2. Add XAI reports for trained image models.
3. Convert the training package into Flower clients and server workflows.
4. Add TFLite and ONNX export with quantization tests.
5. Implement structured decision fusion across image, sensor, and UAV outputs.
6. Add FastAPI, MongoDB, JWT, React dashboards, and prediction history.
