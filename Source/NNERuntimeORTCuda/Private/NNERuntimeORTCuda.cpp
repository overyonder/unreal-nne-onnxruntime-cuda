#include "NNERuntimeORTCuda.h"

#include "NNEModelData.h"
#include "Serialization/MemoryReader.h"
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

	TSharedPtr<UE::NNE::FSharedModelData> SerializeOnnxModelForCudaRuntime(
        TConstArrayView64<uint8> FileData,
        TConstArrayView64<uint8> Descriptor = {},
        TConstArrayView64<uint8> Weights = {})
    {
        if (FileData.IsEmpty() || Descriptor.IsEmpty() != Weights.IsEmpty()) return nullptr;
        if (!Descriptor.IsEmpty())
        {
            FMemoryReaderView Reader(MakeMemoryView(Descriptor), true);
            TMap<FString, int64> Sizes;
            Reader << Sizes;
            if (Reader.IsError() || Sizes.IsEmpty()) return nullptr;
            int64 Total = 0;
            for (const auto& Entry : Sizes)
            {
                if (Entry.Value < 0 || Entry.Value > Weights.Num() - Total) return nullptr;
                Total += Entry.Value;
            }
            if (Total != Weights.Num()) return nullptr;
        }
        const int64 OnnxSize = FileData.Num(), DescriptorSize = Descriptor.Num(), WeightsSize = Weights.Num();
        TArray64<uint8> Bytes;
        Bytes.SetNumUninitialized(SerializedOnnxModelHeaderSize + OnnxSize + DescriptorSize + WeightsSize);
        uint8* Cursor = Bytes.GetData();
        FMemory::Memcpy(Cursor, &SerializedOnnxModelMagic, sizeof(uint32)); Cursor += sizeof(uint32);
        FMemory::Memcpy(Cursor, &SerializedOnnxModelVersion, sizeof(uint32)); Cursor += sizeof(uint32);
        FMemory::Memcpy(Cursor, &OnnxSize, sizeof(int64)); Cursor += sizeof(int64);
        FMemory::Memcpy(Cursor, &DescriptorSize, sizeof(int64)); Cursor += sizeof(int64);
        FMemory::Memcpy(Cursor, &WeightsSize, sizeof(int64)); Cursor += sizeof(int64);
        FMemory::Memcpy(Cursor, FileData.GetData(), OnnxSize); Cursor += OnnxSize;
        if (DescriptorSize) FMemory::Memcpy(Cursor, Descriptor.GetData(), DescriptorSize);
        Cursor += DescriptorSize;
        if (WeightsSize) FMemory::Memcpy(Cursor, Weights.GetData(), WeightsSize);
        return MakeShared<UE::NNE::FSharedModelData>(MakeSharedBufferFromArray(MoveTemp(Bytes)), 0);
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
	return FileType.Equals(TEXT("onnx"), ESearchCase::IgnoreCase) && !FileData.IsEmpty()
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

	const auto* Descriptor = AdditionalFileData.Find(OnnxExternalDataDescriptorKey);
    const auto* Weights = AdditionalFileData.Find(OnnxExternalDataBytesKey);
    return SerializeOnnxModelForCudaRuntime(FileData,
        Descriptor ? *Descriptor : TConstArrayView64<uint8>(),
        Weights ? *Weights : TConstArrayView64<uint8>());
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
		&& !ModelData->GetFileData().IsEmpty())
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
		&& !ModelData->GetFileData().IsEmpty())
	{
		// GetAdditionalFileData incorrectly narrows large buffers to a 32-bit view in UE 5.8.1.
        // GetModelData passes its complete additional-data map to CreateModelData as 64-bit views.
        if (!ModelData->GetAdditionalFileData(OnnxExternalDataDescriptorKey).IsEmpty())
            SharedModelData = ModelData->GetModelData(GetRuntimeName());
        else
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
