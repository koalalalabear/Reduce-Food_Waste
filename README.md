# Reduce-Food-Waste
This project uses a custom trained EfficientNetB0 Model to detect fresh produce and displays the duration that the produce is fresh for. It is an android application of the [Fresh-to-Death-Data-Science-Project](https://github.com/koalalalabear/Fresh-to-Death-Data-Science-Project/tree/main) project as I wanted to challenge myself to gain some android development experience since I had none. 

## Builing and Running the App:

1. Download the code for the starter app from [google codelabs]([https://developers.google.com/codelabs/tflite-object-detection-android#0](https://developers.google.com/codelabs/tflite-object-detection-android#2) which was designed for experienced mobile developers who want to gain experience with Machine Learning.
2. Import starter folder into Android Studio and sync the project with Gradle Files.
3. Add custom model.tflite to the starter app under assets folder of the starter app and updated dependencies in gradle.
4. Set up and run on-device object detection on an image by modifying the function runObjectDetection in MainActivity.kt, including using TFLite's API to create a TensorImage from Bitmap, implementing and tuning the object detector instance and displaing the results.

## Sources:

[Google codelabs]([https://developers.google.com/codelabs/tflite-object-detection-android#0](https://developers.google.com/codelabs/tflite-object-detection-android#2)
[Setting up and configuring the emulator](https://developer.android.com/studio/run/emulator)
[Realtime Object Detection Android App youtube tutorial](https://developer.android.com/studio/run/emulator)
[How to Enable USB Debugging on an Android Device](https://www.youtube.com/watch?v=0usgePpr8_Y&ab_channel=TheUnlockr)


