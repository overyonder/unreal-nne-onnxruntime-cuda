#pragma once

#include "NNERuntimeCPU.h"
#include "NNETypes.h"
#include "onnxruntime_cxx_api.h"

namespace UE::NNE
{
	class FSharedModelData;
}

namespace UE::NNERuntimeORTCuda::Private
{
	class FOnnxRuntimeCudaEnvironment;

	class FOnnxRuntimeCudaModel final : public UE::NNE::IModelCPU
	{
	public:
		FOnnxRuntimeCudaModel(
			TSharedRef<FOnnxRuntimeCudaEnvironment> InEnvironment,
			TSharedRef<UE::NNE::FSharedModelData> InModelData);

		virtual TSharedPtr<UE::NNE::IModelInstanceCPU> CreateModelInstanceCPU() override;

	private:
		TSharedRef<FOnnxRuntimeCudaEnvironment> Environment;
		TSharedRef<UE::NNE::FSharedModelData> ModelData;
	};

	class FOnnxRuntimeCudaModelInstance final : public UE::NNE::IModelInstanceCPU
	{
	public:
		explicit FOnnxRuntimeCudaModelInstance(TSharedRef<FOnnxRuntimeCudaEnvironment> InEnvironment);

		bool Initialize(TConstArrayView64<uint8> SerializedModelData);

		virtual TConstArrayView<UE::NNE::FTensorDesc> GetInputTensorDescs() const override;
		virtual TConstArrayView<UE::NNE::FTensorDesc> GetOutputTensorDescs() const override;
		virtual TConstArrayView<UE::NNE::FTensorShape> GetInputTensorShapes() const override;
		virtual TConstArrayView<UE::NNE::FTensorShape> GetOutputTensorShapes() const override;
		virtual ESetInputTensorShapesStatus SetInputTensorShapes(TConstArrayView<UE::NNE::FTensorShape> InInputShapes) override;
		virtual ERunSyncStatus RunSync(
			TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors,
			TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors) override;

	private:
		bool ReadInputAndOutputTensorDescriptions();
		bool ReadTensorDescriptions(
			bool bReadInputTensors,
			TArray<UE::NNE::FTensorDesc>& OutTensorDescriptions,
			TArray<ONNXTensorElementDataType>& OutOnnxTensorElementTypes,
			TArray<Ort::AllocatedStringPtr>& OutAllocatedTensorNames,
			TArray<const char*>& OutTensorNames);

		TSharedRef<FOnnxRuntimeCudaEnvironment> Environment;
		TUniquePtr<Ort::Session> Session;
		Ort::AllocatorWithDefaultOptions Allocator;
		Ort::MemoryInfo CpuMemoryInformation;
		TArray<UE::NNE::FTensorDesc> InputTensorDescriptions;
		TArray<UE::NNE::FTensorDesc> OutputTensorDescriptions;
		TArray<UE::NNE::FTensorShape> InputTensorShapes;
		TArray<UE::NNE::FTensorShape> OutputTensorShapes;
		TArray<ONNXTensorElementDataType> InputOnnxTensorElementTypes;
		TArray<ONNXTensorElementDataType> OutputOnnxTensorElementTypes;
		TArray<Ort::AllocatedStringPtr> AllocatedInputTensorNames;
		TArray<Ort::AllocatedStringPtr> AllocatedOutputTensorNames;
		TArray<const char*> InputTensorNames;
		TArray<const char*> OutputTensorNames;
	};
}
