#include "OnnxRuntimeCudaEnvironment.h"

#include "HAL/PlatformProcess.h"

namespace UE::NNERuntimeORTCuda::Private
{
	FOnnxRuntimeCudaEnvironment::FOnnxRuntimeCudaEnvironment(TArray<void*>&& InLoadedSharedLibraryHandles)
		: LoadedSharedLibraryHandles(MoveTemp(InLoadedSharedLibraryHandles))
	{
	}

	void FOnnxRuntimeCudaEnvironment::Initialize(const OrtApi* InOnnxRuntimeApi)
	{
		Ort::InitApi(InOnnxRuntimeApi);
		Environment = MakeUnique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "NNERuntimeORTCuda");
	}

	FOnnxRuntimeCudaEnvironment::~FOnnxRuntimeCudaEnvironment()
	{
		Environment.Reset();
		for (int32 SharedLibraryIndex = LoadedSharedLibraryHandles.Num() - 1; SharedLibraryIndex >= 0; --SharedLibraryIndex)
		{
			FPlatformProcess::FreeDllHandle(LoadedSharedLibraryHandles[SharedLibraryIndex]);
		}
	}

	Ort::Env& FOnnxRuntimeCudaEnvironment::GetEnvironment()
	{
		return *Environment;
	}
}
