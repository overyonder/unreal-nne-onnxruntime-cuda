#pragma once

#include "onnxruntime_cxx_api.h"

namespace UE::NNERuntimeORTCuda::Private
{
	class FOnnxRuntimeCudaEnvironment final
	{
	public:
		explicit FOnnxRuntimeCudaEnvironment(TArray<void*>&& InLoadedSharedLibraryHandles);
		~FOnnxRuntimeCudaEnvironment();

		void Initialize(const OrtApi* InOnnxRuntimeApi);
		Ort::Env& GetEnvironment();

	private:
		TUniquePtr<Ort::Env> Environment;
		TArray<void*> LoadedSharedLibraryHandles;
	};
}
