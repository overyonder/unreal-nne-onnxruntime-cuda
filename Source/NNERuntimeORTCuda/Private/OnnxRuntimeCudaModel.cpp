#include "OnnxRuntimeCudaModel.h"

#include "Algo/Transform.h"
#include "NNEModelData.h"
#include "OnnxRuntimeCudaEnvironment.h"
#include "OnnxRuntimeCudaModelDataFormat.h"

DEFINE_LOG_CATEGORY_STATIC(LogNNERuntimeORTCudaModel, Log, All);

namespace UE::NNERuntimeORTCuda::Private
{
	namespace
	{
		TOptional<ENNETensorDataType> TranslateOnnxTensorElementTypeToUnrealTensorDataType(ONNXTensorElementDataType OnnxTensorElementType)
		{
			switch (OnnxTensorElementType)
			{
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      return ENNETensorDataType::Float;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      return ENNETensorDataType::UInt8;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       return ENNETensorDataType::Int8;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:     return ENNETensorDataType::UInt16;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:      return ENNETensorDataType::Int16;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      return ENNETensorDataType::Int32;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      return ENNETensorDataType::Int64;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       return ENNETensorDataType::Boolean;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    return ENNETensorDataType::Half;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     return ENNETensorDataType::Double;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     return ENNETensorDataType::UInt32;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     return ENNETensorDataType::UInt64;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:  return ENNETensorDataType::Complex64;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return ENNETensorDataType::Complex128;
			case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   return ENNETensorDataType::BFloat16;
			default:                                       return {};
			}
		}

		TArray<int64_t> MakeOnnxTensorDimensions(const UE::NNE::FTensorShape& TensorShape)
		{
			TArray<int64_t> Dimensions;
			Dimensions.Reserve(TensorShape.Rank());
			Algo::Transform(TensorShape.GetData(), Dimensions, [](uint32 Dimension) { return static_cast<int64_t>(Dimension); });
			return Dimensions;
		}

		uint64 GetRequiredTensorBufferSize(const UE::NNE::FTensorShape& TensorShape, ENNETensorDataType TensorDataType)
		{
			return TensorShape.Volume() * UE::NNE::GetTensorDataTypeSizeInBytes(TensorDataType);
		}
	}

	FOnnxRuntimeCudaModel::FOnnxRuntimeCudaModel(
		TSharedRef<FOnnxRuntimeCudaEnvironment> InEnvironment,
		TSharedRef<UE::NNE::FSharedModelData> InModelData)
		: Environment(MoveTemp(InEnvironment)), ModelData(MoveTemp(InModelData))
	{
	}

	TSharedPtr<UE::NNE::IModelInstanceCPU> FOnnxRuntimeCudaModel::CreateModelInstanceCPU()
	{
		TSharedPtr<FOnnxRuntimeCudaModelInstance> ModelInstance = MakeShared<FOnnxRuntimeCudaModelInstance>(Environment);
		return ModelInstance->Initialize(ModelData->GetView()) ? ModelInstance : nullptr;
	}

	FOnnxRuntimeCudaModelInstance::FOnnxRuntimeCudaModelInstance(TSharedRef<FOnnxRuntimeCudaEnvironment> InEnvironment)
		: Environment(MoveTemp(InEnvironment)),
		  CpuMemoryInformation(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
	{
	}

	bool FOnnxRuntimeCudaModelInstance::Initialize(TConstArrayView64<uint8> SerializedModelData)
	{
		if (SerializedModelData.Num() <= SerializedOnnxModelHeaderSize)
		{
			UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("The serialized ONNX model is empty or truncated."));
			return false;
		}

		uint32 Magic = 0;
		uint32 Version = 0;
		FMemory::Memcpy(&Magic, SerializedModelData.GetData(), sizeof(Magic));
		FMemory::Memcpy(&Version, SerializedModelData.GetData() + sizeof(Magic), sizeof(Version));
		if (Magic != SerializedOnnxModelMagic || Version != SerializedOnnxModelVersion)
		{
			UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("The serialized ONNX model has an unsupported header."));
			return false;
		}

		try
		{
			Ort::SessionOptions SessionOptions;
			SessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
			SessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

			Ort::CUDAProviderOptions CudaProviderOptions;
			CudaProviderOptions.Update({
				{ "device_id", "0" },
				{ "cudnn_conv_algo_search", "HEURISTIC" },
				{ "do_copy_in_default_stream", "1" }
			});
			SessionOptions.AppendExecutionProvider_CUDA_V2(*CudaProviderOptions);

			const uint8* OnnxModelData = SerializedModelData.GetData() + SerializedOnnxModelHeaderSize;
			const size_t OnnxModelDataSize = static_cast<size_t>(SerializedModelData.Num() - SerializedOnnxModelHeaderSize);
			Session = MakeUnique<Ort::Session>(Environment->GetEnvironment(), OnnxModelData, OnnxModelDataSize, SessionOptions);
			return ReadInputAndOutputTensorDescriptions();
		}
		catch (const Ort::Exception& Exception)
		{
			UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("Could not create the CUDA ONNX Runtime session: %s"), UTF8_TO_TCHAR(Exception.what()));
			return false;
		}
	}

	TConstArrayView<UE::NNE::FTensorDesc> FOnnxRuntimeCudaModelInstance::GetInputTensorDescs() const
	{
		return InputTensorDescriptions;
	}

	TConstArrayView<UE::NNE::FTensorDesc> FOnnxRuntimeCudaModelInstance::GetOutputTensorDescs() const
	{
		return OutputTensorDescriptions;
	}

	TConstArrayView<UE::NNE::FTensorShape> FOnnxRuntimeCudaModelInstance::GetInputTensorShapes() const
	{
		return InputTensorShapes;
	}

	TConstArrayView<UE::NNE::FTensorShape> FOnnxRuntimeCudaModelInstance::GetOutputTensorShapes() const
	{
		return OutputTensorShapes;
	}

	FOnnxRuntimeCudaModelInstance::ESetInputTensorShapesStatus FOnnxRuntimeCudaModelInstance::SetInputTensorShapes(
		TConstArrayView<UE::NNE::FTensorShape> InInputShapes)
	{
		InputTensorShapes.Reset();
		OutputTensorShapes.Reset();

		if (InInputShapes.Num() != InputTensorDescriptions.Num())
		{
			UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("Received %d input shapes for a model with %d inputs."), InInputShapes.Num(), InputTensorDescriptions.Num());
			return ESetInputTensorShapesStatus::Fail;
		}

		for (int32 TensorIndex = 0; TensorIndex < InInputShapes.Num(); ++TensorIndex)
		{
			if (!InInputShapes[TensorIndex].IsCompatibleWith(InputTensorDescriptions[TensorIndex].GetShape()))
			{
				UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("Input shape %d is incompatible with tensor %s."), TensorIndex, *InputTensorDescriptions[TensorIndex].GetName());
				return ESetInputTensorShapesStatus::Fail;
			}
		}

		TArray<UE::NNE::FTensorShape> ResolvedOutputTensorShapes;
		ResolvedOutputTensorShapes.Reserve(OutputTensorDescriptions.Num());
		for (const UE::NNE::FTensorDesc& OutputTensorDescription : OutputTensorDescriptions)
		{
			if (!OutputTensorDescription.GetShape().IsConcrete())
			{
				ResolvedOutputTensorShapes.Reset();
				break;
			}
			ResolvedOutputTensorShapes.Add(UE::NNE::FTensorShape::MakeFromSymbolic(OutputTensorDescription.GetShape()));
		}

		InputTensorShapes = InInputShapes;
		OutputTensorShapes = MoveTemp(ResolvedOutputTensorShapes);
		return ESetInputTensorShapesStatus::Ok;
	}

	FOnnxRuntimeCudaModelInstance::ERunSyncStatus FOnnxRuntimeCudaModelInstance::RunSync(
		TConstArrayView<UE::NNE::FTensorBindingCPU> InInputTensors,
		TConstArrayView<UE::NNE::FTensorBindingCPU> InOutputTensors)
	{
		if (!Session.IsValid() || InputTensorShapes.Num() != InputTensorDescriptions.Num())
		{
			return ERunSyncStatus::Fail;
		}
		if (InInputTensors.Num() != InputTensorDescriptions.Num()
			|| (!InOutputTensors.IsEmpty() && InOutputTensors.Num() != OutputTensorDescriptions.Num()))
		{
			return ERunSyncStatus::Fail;
		}

		try
		{
			TArray<Ort::Value> InputValues;
			InputValues.Reserve(InInputTensors.Num());
			for (int32 TensorIndex = 0; TensorIndex < InInputTensors.Num(); ++TensorIndex)
			{
				const uint64 RequiredSize = GetRequiredTensorBufferSize(InputTensorShapes[TensorIndex], InputTensorDescriptions[TensorIndex].GetDataType());
				if (InInputTensors[TensorIndex].Data == nullptr || InInputTensors[TensorIndex].SizeInBytes != RequiredSize)
				{
					return ERunSyncStatus::Fail;
				}
				TArray<int64_t> Dimensions = MakeOnnxTensorDimensions(InputTensorShapes[TensorIndex]);
				InputValues.Add(Ort::Value::CreateTensor(
					CpuMemoryInformation,
					InInputTensors[TensorIndex].Data,
					static_cast<size_t>(InInputTensors[TensorIndex].SizeInBytes),
					Dimensions.GetData(),
					Dimensions.Num(),
					InputOnnxTensorElementTypes[TensorIndex]));
			}

			TArray<Ort::Value> OutputValues;
			OutputValues.Reserve(OutputTensorDescriptions.Num());
			for (int32 TensorIndex = 0; TensorIndex < OutputTensorDescriptions.Num(); ++TensorIndex)
			{
				const bool bOutputShapeIsKnown = OutputTensorShapes.Num() == OutputTensorDescriptions.Num();
				const bool bOutputBindingIsUsable = bOutputShapeIsKnown
					&& !InOutputTensors.IsEmpty()
					&& InOutputTensors[TensorIndex].Data != nullptr
					&& InOutputTensors[TensorIndex].SizeInBytes >= GetRequiredTensorBufferSize(
						OutputTensorShapes[TensorIndex],
						OutputTensorDescriptions[TensorIndex].GetDataType());
				if (!bOutputBindingIsUsable)
				{
					OutputValues.Emplace(nullptr);
					continue;
				}
				TArray<int64_t> Dimensions = MakeOnnxTensorDimensions(OutputTensorShapes[TensorIndex]);
				OutputValues.Add(Ort::Value::CreateTensor(
					CpuMemoryInformation,
					InOutputTensors[TensorIndex].Data,
					static_cast<size_t>(InOutputTensors[TensorIndex].SizeInBytes),
					Dimensions.GetData(),
					Dimensions.Num(),
					OutputOnnxTensorElementTypes[TensorIndex]));
			}

			Session->Run(
				Ort::RunOptions { nullptr },
				InputTensorNames.GetData(),
				InputValues.GetData(),
				InputValues.Num(),
				OutputTensorNames.GetData(),
				OutputValues.GetData(),
				OutputValues.Num());

			if (OutputTensorShapes.IsEmpty())
			{
				OutputTensorShapes.Reserve(OutputValues.Num());
				for (int32 TensorIndex = 0; TensorIndex < OutputValues.Num(); ++TensorIndex)
				{
					const std::vector<int64_t> OnnxDimensions = OutputValues[TensorIndex].GetTensorTypeAndShapeInfo().GetShape();
					TArray<uint32, TInlineAllocator<UE::NNE::FTensorShape::MaxRank>> UnrealDimensions;
					UnrealDimensions.Reserve(OnnxDimensions.size());
					Algo::Transform(OnnxDimensions, UnrealDimensions, [](int64 Dimension) { return static_cast<uint32>(Dimension); });
					OutputTensorShapes.Add(UE::NNE::FTensorShape::Make(UnrealDimensions));

					if (!InOutputTensors.IsEmpty() && InOutputTensors[TensorIndex].Data != nullptr)
					{
						const uint64 RequiredSize = GetRequiredTensorBufferSize(OutputTensorShapes[TensorIndex], OutputTensorDescriptions[TensorIndex].GetDataType());
						if (InOutputTensors[TensorIndex].SizeInBytes >= RequiredSize)
						{
							FMemory::Memcpy(InOutputTensors[TensorIndex].Data, OutputValues[TensorIndex].GetTensorRawData(), RequiredSize);
						}
					}
				}
			}
			return ERunSyncStatus::Ok;
		}
		catch (const Ort::Exception& Exception)
		{
			UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("CUDA ONNX Runtime inference failed: %s"), UTF8_TO_TCHAR(Exception.what()));
			return ERunSyncStatus::Fail;
		}
	}

	bool FOnnxRuntimeCudaModelInstance::ReadInputAndOutputTensorDescriptions()
	{
		return ReadTensorDescriptions(true, InputTensorDescriptions, InputOnnxTensorElementTypes, AllocatedInputTensorNames, InputTensorNames)
			&& ReadTensorDescriptions(false, OutputTensorDescriptions, OutputOnnxTensorElementTypes, AllocatedOutputTensorNames, OutputTensorNames);
	}

	bool FOnnxRuntimeCudaModelInstance::ReadTensorDescriptions(
		bool bReadInputTensors,
		TArray<UE::NNE::FTensorDesc>& OutTensorDescriptions,
		TArray<ONNXTensorElementDataType>& OutOnnxTensorElementTypes,
		TArray<Ort::AllocatedStringPtr>& OutAllocatedTensorNames,
		TArray<const char*>& OutTensorNames)
	{
		const size_t TensorCount = bReadInputTensors ? Session->GetInputCount() : Session->GetOutputCount();
		OutTensorDescriptions.Reserve(TensorCount);
		OutOnnxTensorElementTypes.Reserve(TensorCount);
		OutAllocatedTensorNames.Reserve(TensorCount);
		OutTensorNames.Reserve(TensorCount);

		for (size_t TensorIndex = 0; TensorIndex < TensorCount; ++TensorIndex)
		{
			Ort::AllocatedStringPtr TensorName = bReadInputTensors
				? Session->GetInputNameAllocated(TensorIndex, Allocator)
				: Session->GetOutputNameAllocated(TensorIndex, Allocator);
			Ort::TypeInfo TensorTypeInformation = bReadInputTensors
				? Session->GetInputTypeInfo(TensorIndex)
				: Session->GetOutputTypeInfo(TensorIndex);
			Ort::ConstTensorTypeAndShapeInfo TensorShapeInformation = TensorTypeInformation.GetTensorTypeAndShapeInfo();
			const ONNXTensorElementDataType OnnxTensorElementType = TensorShapeInformation.GetElementType();
			const TOptional<ENNETensorDataType> UnrealTensorDataType = TranslateOnnxTensorElementTypeToUnrealTensorDataType(OnnxTensorElementType);
			if (!UnrealTensorDataType.IsSet())
			{
				UE_LOG(LogNNERuntimeORTCudaModel, Error, TEXT("Tensor %s uses unsupported ONNX element type %d."), UTF8_TO_TCHAR(TensorName.get()), static_cast<int32>(OnnxTensorElementType));
				return false;
			}

			const std::vector<int64_t> OnnxDimensions = TensorShapeInformation.GetShape();
			TArray<int32, TInlineAllocator<UE::NNE::FTensorShape::MaxRank>> UnrealDimensions;
			UnrealDimensions.Reserve(OnnxDimensions.size());
			Algo::Transform(OnnxDimensions, UnrealDimensions, [](int64 Dimension) { return static_cast<int32>(Dimension); });
			if (UnrealDimensions.Num() > UE::NNE::FTensorShape::MaxRank)
			{
				return false;
			}

			const UE::NNE::FSymbolicTensorShape SymbolicShape = UE::NNE::FSymbolicTensorShape::Make(UnrealDimensions);
			OutTensorDescriptions.Add(UE::NNE::FTensorDesc::Make(UTF8_TO_TCHAR(TensorName.get()), SymbolicShape, UnrealTensorDataType.GetValue()));
			OutOnnxTensorElementTypes.Add(OnnxTensorElementType);
			OutAllocatedTensorNames.Add(MoveTemp(TensorName));
			OutTensorNames.Add(OutAllocatedTensorNames.Last().get());
		}

		return true;
	}
}
