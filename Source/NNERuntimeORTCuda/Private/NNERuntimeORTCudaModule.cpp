#include "NNERuntimeORTCudaModule.h"

#include "HAL/PlatformProcess.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "NNE.h"
#include "NNERuntimeORTCuda.h"
#include "OnnxRuntimeCudaEnvironment.h"
#include "onnxruntime_c_api.h"

DEFINE_LOG_CATEGORY_STATIC(LogNNERuntimeORTCudaModule, Log, All);

namespace
{
	void* LoadRequiredSharedLibrary(const FString& SharedLibraryPath, TArray<void*>& LoadedSharedLibraryHandles)
	{
		void* SharedLibraryHandle = FPlatformProcess::GetDllHandle(*SharedLibraryPath);
		if (SharedLibraryHandle != nullptr)
		{
			LoadedSharedLibraryHandles.Add(SharedLibraryHandle);
		}
		else
		{
			UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("Could not load %s."), *SharedLibraryPath);
		}
		return SharedLibraryHandle;
	}
}

void FNNERuntimeORTCudaModule::StartupModule()
{
	TArray<void*> LoadedSharedLibraryHandles;
	const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("NNERuntimeORTCuda"));
	if (!Plugin.IsValid())
	{
		UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("Could not find the NNERuntimeORTCuda plugin descriptor."));
		return;
	}

	const FString SharedLibraryDirectory = FPaths::Combine(
		Plugin->GetBaseDir(),
		TEXT("Binaries/ThirdParty/OnnxRuntimeCuda/Linux/x86_64"));
	const FString OnnxRuntimeSharedLibraryPath = FPaths::Combine(SharedLibraryDirectory, TEXT("libonnxruntime_cuda_for_unreal.so.1"));
	const FString ProviderSharedLibraryPath = FPaths::Combine(SharedLibraryDirectory, TEXT("libonnxruntime_providers_shared.so"));
	void* OnnxRuntimeSharedLibraryHandle = LoadRequiredSharedLibrary(OnnxRuntimeSharedLibraryPath, LoadedSharedLibraryHandles);
	if (OnnxRuntimeSharedLibraryHandle == nullptr
		|| LoadRequiredSharedLibrary(ProviderSharedLibraryPath, LoadedSharedLibraryHandles) == nullptr)
	{
		for (int32 SharedLibraryIndex = LoadedSharedLibraryHandles.Num() - 1; SharedLibraryIndex >= 0; --SharedLibraryIndex)
		{
			FPlatformProcess::FreeDllHandle(LoadedSharedLibraryHandles[SharedLibraryIndex]);
		}
		return;
	}
	Environment = MakeShared<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaEnvironment>(MoveTemp(LoadedSharedLibraryHandles));

	using FOrtGetApiBase = const OrtApiBase* (*)();
	FOrtGetApiBase OrtGetApiBase = reinterpret_cast<FOrtGetApiBase>(
		FPlatformProcess::GetDllExport(OnnxRuntimeSharedLibraryHandle, TEXT("OrtGetApiBase")));
	if (OrtGetApiBase == nullptr)
	{
		UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("The CUDA ONNX Runtime library does not export OrtGetApiBase."));
		Environment.Reset();
		return;
	}

	const OrtApi* OnnxRuntimeApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (OnnxRuntimeApi == nullptr)
	{
		UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("The CUDA ONNX Runtime API version does not match the plugin headers."));
		Environment.Reset();
		return;
	}

	try
	{
		Environment->Initialize(OnnxRuntimeApi);
		Runtime = NewObject<UNNERuntimeORTCuda>();
		Runtime->AddToRoot();
		Runtime->Initialize(Environment.ToSharedRef());
		if (UE::NNE::RegisterRuntime(TWeakInterfacePtr<INNERuntime>(Runtime.Get())) != UE::NNE::ERegisterRuntimeStatus::Ok)
		{
			UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("NNE rejected registration of NNERuntimeORTCuda."));
			Runtime->RemoveFromRoot();
			Runtime.Reset();
			Environment.Reset();
		}
	}
	catch (const Ort::Exception& Exception)
	{
		UE_LOG(LogNNERuntimeORTCudaModule, Error, TEXT("Could not initialize CUDA ONNX Runtime: %s"), UTF8_TO_TCHAR(Exception.what()));
		Environment.Reset();
	}
}

void FNNERuntimeORTCudaModule::ShutdownModule()
{
	if (Runtime.IsValid())
	{
		UE::NNE::UnregisterRuntime(TWeakInterfacePtr<INNERuntime>(Runtime.Get()));
		Runtime->RemoveFromRoot();
		Runtime.Reset();
	}
	Environment.Reset();
}

IMPLEMENT_MODULE(FNNERuntimeORTCudaModule, NNERuntimeORTCuda)
