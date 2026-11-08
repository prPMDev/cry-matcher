# Third-party notices

The MIT License in `LICENSE` covers only the original code of this project
(`index.html`, `engine.js`, `sw.js`, `manifest.webmanifest` and the icons).
The parts below belong to their owners and stay under their own license.

## YAMNet model weights

- File: `yamnet-weights.js`
- Source: YAMNet, from the TensorFlow Models repository (`research/audioset/yamnet`).
- Copyright: Google LLC.
- License: Apache License 2.0 (full text in `LICENSE-Apache-2.0.txt`).
- Changes: the weights were converted from the original Keras file to a
  JavaScript file for the browser. Batch normalization was folded into the
  convolution weights, and the weights are stored as 16-bit floats.
  The network output was checked against the original model.

## TensorFlow.js

- File: `tf.min.js` (version 4.22.0, unchanged)
- Copyright: Google LLC.
- License: Apache License 2.0 (full text in `LICENSE-Apache-2.0.txt`).
