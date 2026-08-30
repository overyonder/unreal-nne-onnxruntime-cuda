using System.IO;
using UnrealBuildTool;

public class NNERuntimeORTCuda : ModuleRules
{
	public NNERuntimeORTCuda(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		bEnableExceptions = true;
		bDisableAutoRTFMInstrumentation = true;

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"NNE",
			"Projects"
		});

		if (Target.Platform != UnrealTargetPlatform.Linux || Target.Architecture != UnrealArch.X64)
		{
			throw new BuildException("NNERuntimeORTCuda currently supports Linux x86_64 only.");
		}

		string OnnxRuntimeCudaRoot = Path.Combine(PluginDirectory, "Source", "ThirdParty", "OnnxRuntimeCuda");
		string OnnxRuntimeCudaLibraries = Path.Combine(PluginDirectory, "Binaries", "ThirdParty", "OnnxRuntimeCuda", "Linux", "x86_64");
		string ThirdPartyLicenses = Path.Combine(PluginDirectory, "ThirdPartyLicenses");

		PrivateIncludePaths.Add(Path.Combine(OnnxRuntimeCudaRoot, "include"));
		PrivateDefinitions.Add("ORT_API_MANUAL_INIT");

		string[] RequiredOnnxRuntimeCudaFiles = new string[]
		{
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_c_api.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_cxx_api.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_cxx_inline.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_float16.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_ep_c_api.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_ep_device_ep_metadata_keys.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_env_config_keys.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_lite_custom_op.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_run_options_config_keys.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "onnxruntime_session_options_config_keys.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "provider_options.h"),
			Path.Combine(OnnxRuntimeCudaRoot, "include", "cpu_provider_factory.h"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libonnxruntime_cuda_for_unreal.so.1"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libonnxruntime_providers_shared.so"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libonnxruntime_providers_cuda.so"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcublasLt.so.12"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcublas.so.12"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcurand.so.10"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcufft.so.11"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudart.so.12"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_adv.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_cnn.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_engines_precompiled.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_engines_runtime_compiled.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_graph.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_heuristic.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libcudnn_ops.so.9"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libnvrtc.so.12"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libnvrtc-builtins.so.12.8"),
			Path.Combine(OnnxRuntimeCudaLibraries, "libnvJitLink.so.12"),
			Path.Combine(ThirdPartyLicenses, "ONNXRuntime-LICENSE.txt"),
			Path.Combine(ThirdPartyLicenses, "ONNXRuntime-ThirdPartyNotices.txt"),
			Path.Combine(ThirdPartyLicenses, "NVIDIA-CUDA-License.txt"),
			Path.Combine(ThirdPartyLicenses, "NVIDIA-cuDNN-License.txt")
		};

		foreach (string RequiredOnnxRuntimeCudaFile in RequiredOnnxRuntimeCudaFiles)
		{
			if (!File.Exists(RequiredOnnxRuntimeCudaFile))
			{
				throw new BuildException("NNERuntimeORTCuda third-party installation is incomplete. Run Scripts/install-onnxruntime-cuda.sh. Missing: " + RequiredOnnxRuntimeCudaFile);
			}
		}

		foreach (string SharedLibraryPath in Directory.GetFiles(OnnxRuntimeCudaLibraries, "*.so*"))
		{
			RuntimeDependencies.Add(SharedLibraryPath);
		}
		foreach (string ThirdPartyLicensePath in Directory.GetFiles(ThirdPartyLicenses, "*.txt"))
		{
			RuntimeDependencies.Add(ThirdPartyLicensePath);
		}
	}
}
