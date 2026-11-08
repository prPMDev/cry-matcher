# Cry Matcher

A private web app. It keeps your own reference cry clips for five sounds (neh, owh, heh, eairh, eh)
and tells you which sound a new recording is closest to.

- Your clips stay on your phone (browser storage). They are not in this repository and are not sent anywhere.
- Use Export in the app to make a backup file.
- After the first visit the app works offline.
- This is a private experiment. It is not medical advice.

Third-party parts, both under the Apache License 2.0 (see LICENSE-Apache-2.0.txt):
- TensorFlow.js 4.22.0 (tf.min.js), Google LLC.
- YAMNet sound model weights (yamnet-weights.js), Google LLC, converted for the browser.
