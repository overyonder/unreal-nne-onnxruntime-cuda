# Unreal NNE ONNX Runtime CUDA

This Unreal Engine 5.8 plugin registers `NNERuntimeORTCuda`, a synchronous NNE runtime that accepts CPU tensor bindings and executes ONNX graphs with ONNX Runtime's CUDA execution provider. This interface matches the MetaHuman face tracker without changing engine source.

The current target is Linux x86_64 with ONNX Runtime 1.24.3, CUDA 12.8, and cuDNN 9.8. The pinned cuDNN release retains Pascal support for the GTX 1070; newer cuDNN releases have dropped it.

## Install third-party files

Run:

```sh
./Scripts/install-onnxruntime-cuda.sh
```

The script downloads pinned official ONNX Runtime and NVIDIA packages and verifies their hashes. It gives the matched Microsoft core a plugin-specific SONAME so it can coexist with Epic's customized CPU core without mixing their private provider ABIs.

## Unreal project integration

Place or symlink this repository at `Plugins/NNERuntimeORTCuda` and enable the
plugin. Add both backend selections to the project's
`Config/DefaultEngine.ini` file:

```ini
[ConsoleVariables]
mh.RealtimeVideo.Backend=NNERuntimeORTCuda
mh.FaceTracker.Backend=NNERuntimeORTCuda
```

`mh.RealtimeVideo.Backend` selects the runtime for MetaHuman Video Live Link.
`mh.FaceTracker.Backend` selects it for the separate offline contour tracker.
These settings do not take effect from `Config/DefaultGame.ini`. Restart the
editor after changing them, then recreate any existing MetaHuman Live Link
source because it retains the backend selected when the source was created.

MetaHuman reports this plugin as `using CPU runtime (NNERuntimeORTCuda)` because
it exposes Unreal's synchronous CPU-binding interface. The tensor buffers are
owned by the caller on the CPU, but ONNX Runtime transfers them and executes the
network through the CUDA execution provider. The backend name in parentheses is
the useful confirmation that the plugin was selected.

UE 5.8's bundled MetaHuman model assets list only Epic's CPU and DirectML runtimes as cook targets. In the editor this plugin reads their retained ONNX source bytes directly, so those read-only engine assets do not need to be duplicated or modified. A packaged game must retarget or duplicate the model assets so `NNERuntimeORTCuda` model data is included during cooking.

The initial model-data format accepts self-contained ONNX files. Models imported with ONNX external-data weight files are rejected instead of silently dropping their weights.

## Related plugin

[Linux Video Capture Media](https://github.com/overyonder/unreal-linux-v4l2-media)
provides the V4L2 Media Framework source that supplies Linux webcam frames to
MetaHuman Video Live Link. Use that capture plugin for camera input and this NNE
plugin to run the neural-network solver on a supported NVIDIA GPU.
