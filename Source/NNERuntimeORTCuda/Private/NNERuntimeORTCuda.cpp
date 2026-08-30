#include "NNERuntimeORTCuda.h"

#include "NNEModelData.h"
#include "OnnxRuntimeCudaEnvironment.h"
#include "OnnxRuntimeCudaModel.h"
#include "OnnxRuntimeCudaModelDataFormat.h"

#include UE_INLINE_GENERATED_CPP_BY_NAME(NNERuntimeORTCuda)

DEFINE_LOG_CATEGORY_STATIC(LogNNERuntimeORTCuda, Log, All);

namespace UE::NNERuntimeORTCuda::Private
{
	const FGuid RuntimeModelDataGuid(0x3f26068b, 0x1d7f4cce, 0x89472d89, 0x67ee59b4);
	const FString OnnxExternalDataDescriptorKey(TEXT("OnnxExternalDataDescriptor"));
	const FString OnnxExternalDataBytesKey(TEXT("OnnxExternalDataBytes"));

	bool DoesModelUseExternalOnnxData(const UNNEModelData& ModelData)
	{
		return !ModelData.GetAdditionalFileData(OnnxExternalDataDescriptorKey).IsEmpty()
			|| !ModelData.GetAdditionalFileData(OnnxExternalDataBytesKey).IsEmpty();
	}

	TSharedPtr<UE::NNE::FSharedModelData> SerializeOnnxModelForCudaRuntime(TConstArrayView64<uint8> FileData)
	{
		TArray64<uint8> SerializedModelData;
		SerializedModelData.SetNumUninitialized(SerializedOnnxModelHeaderSize + FileData.Num());
		FMemory::Memcpy(SerializedModelData.GetData(), &SerializedOnnxModelMagic, sizeof(SerializedOnnxModelMagic));
		FMemory::Memcpy(SerializedModelData.GetData() + sizeof(SerializedOnnxModelMagic), &SerializedOnnxModelVersion, sizeof(SerializedOnnxModelVersion));
		FMemory::Memcpy(SerializedModelData.GetData() + SerializedOnnxModelHeaderSize, FileData.GetData(), FileData.Num());
		return MakeShared<UE::NNE::FSharedModelData>(MakeSharedBufferFromArray(MoveTemp(SerializedModelData)), 0);
	}
}

void UNNERuntimeORTCuda::Initialize(TSharedRef<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaEnvironment> InEnvironment)
{
	Environment = MoveTemp(InEnvironment);
}

FString UNNERuntimeORTCuda::GetRuntimeName() const
{
	return TEXT("NNERuntimeORTCuda");
}

UNNERuntimeORTCuda::ECanCreateModelDataStatus UNNERuntimeORTCuda::CanCreateModelData(
	const FString& FileType,
	TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
	const FGuid& FileId,
	const ITargetPlatform* TargetPlatform) const
{
	return FileType.Equals(TEXT("onnx"), ESearchCase::IgnoreCase) && !FileData.IsEmpty() && AdditionalFileData.IsEmpty()
		? ECanCreateModelDataStatus::Ok
		: ECanCreateModelDataStatus::FailFileIdNotSupported;
}

TSharedPtr<UE::NNE::FSharedModelData> UNNERuntimeORTCuda::CreateModelData(
	const FString& FileType,
	TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
	const FGuid& FileId,
	const ITargetPlatform* TargetPlatform)
{
	using namespace UE::NNERuntimeORTCuda::Private;

	if (CanCreateModelData(FileType, FileData, AdditionalFileData, FileId, TargetPlatform) != ECanCreateModelDataStatus::Ok)
	{
		UE_LOG(LogNNERuntimeORTCuda, Error, TEXT("Cannot create CUDA model data for %s."), *FileId.ToString(EGuidFormats::Digits));
		return nullptr;
	}

	return SerializeOnnxModelForCudaRuntime(FileData);
}

FString UNNERuntimeORTCuda::GetModelDataIdentifier(
	const FString& FileType,
	TConstArrayView64<uint8> FileData,
	const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
	const FGuid& FileId,
	const ITargetPlatform* TargetPlatform) const
{
	using namespace UE::NNERuntimeORTCuda::Private;
	return FString::Printf(TEXT("%s-%s-%u"), *FileId.ToString(EGuidFormats::Digits), *RuntimeModelDataGuid.ToString(EGuidFormats::Digits), SerializedOnnxModelVersion);
}

UNNERuntimeORTCuda::ECanCreateModelCPUStatus UNNERuntimeORTCuda::CanCreateModelCPU(TObjectPtr<UNNEModelData> ModelData) const
{
	using namespace UE::NNERuntimeORTCuda::Private;
	if (ModelData == nullptr)
	{
		return ECanCreateModelCPUStatus::Fail;
	}

#if WITH_EDITOR
	if (ModelData->GetFileType().Equals(TEXT("onnx"), ESearchCase::IgnoreCase)
		&& !ModelData->GetFileData().IsEmpty()
		&& !DoesModelUseExternalOnnxData(*ModelData))
	{
		return ECanCreateModelCPUStatus::Ok;
	}
#endif

	const TSharedPtr<UE::NNE::FSharedModelData> SharedModelData = ModelData->GetModelData(GetRuntimeName());
	if (!SharedModelData.IsValid() || SharedModelData->GetView().Num() <= SerializedOnnxModelHeaderSize)
	{
		return ECanCreateModelCPUStatus::Fail;
	}

	uint32 Magic = 0;
	uint32 Version = 0;
	FMemory::Memcpy(&Magic, SharedModelData->GetView().GetData(), sizeof(Magic));
	FMemory::Memcpy(&Version, SharedModelData->GetView().GetData() + sizeof(Magic), sizeof(Version));
	return Magic == SerializedOnnxModelMagic && Version == SerializedOnnxModelVersion
		? ECanCreateModelCPUStatus::Ok
		: ECanCreateModelCPUStatus::Fail;
}

TSharedPtr<UE::NNE::IModelCPU> UNNERuntimeORTCuda::CreateModelCPU(TObjectPtr<UNNEModelData> ModelData)
{
	using namespace UE::NNERuntimeORTCuda::Private;

	if (!Environment.IsValid() || CanCreateModelCPU(ModelData) != ECanCreateModelCPUStatus::Ok)
	{
		return nullptr;
	}

	TSharedPtr<UE::NNE::FSharedModelData> SharedModelData;
#if WITH_EDITOR
	if (ModelData->GetFileType().Equals(TEXT("onnx"), ESearchCase::IgnoreCase)
		&& !ModelData->GetFileData().IsEmpty()
		&& !DoesModelUseExternalOnnxData(*ModelData))
	{
		SharedModelData = SerializeOnnxModelForCudaRuntime(ModelData->GetFileData());
	}
#endif
	if (!SharedModelData.IsValid())
	{
		SharedModelData = ModelData->GetModelData(GetRuntimeName());
	}
	if (!SharedModelData.IsValid())
	{
		return nullptr;
	}

	return MakeShared<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaModel>(
		Environment.ToSharedRef(),
		SharedModelData.ToSharedRef());
}
