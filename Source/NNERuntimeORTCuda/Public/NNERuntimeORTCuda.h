#pragma once

#include "NNERuntime.h"
#include "NNERuntimeCPU.h"
#include "UObject/Object.h"

#include "NNERuntimeORTCuda.generated.h"

namespace UE::NNERuntimeORTCuda::Private
{
	class FOnnxRuntimeCudaEnvironment;
}

UCLASS()
class NNERUNTIMEORTCUDA_API UNNERuntimeORTCuda final : public UObject, public INNERuntime, public INNERuntimeCPU
{
	GENERATED_BODY()

public:
	void Initialize(TSharedRef<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaEnvironment> InEnvironment);

	virtual FString GetRuntimeName() const override;
	virtual ECanCreateModelDataStatus CanCreateModelData(
		const FString& FileType,
		TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
		const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) const override;
	virtual TSharedPtr<UE::NNE::FSharedModelData> CreateModelData(
		const FString& FileType,
		TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
		const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) override;
	virtual FString GetModelDataIdentifier(
		const FString& FileType,
		TConstArrayView64<uint8> FileData,
		const TMap<FString, TConstArrayView64<uint8>>& AdditionalFileData,
		const FGuid& FileId,
		const ITargetPlatform* TargetPlatform) const override;

	virtual ECanCreateModelCPUStatus CanCreateModelCPU(TObjectPtr<UNNEModelData> ModelData) const override;
	virtual TSharedPtr<UE::NNE::IModelCPU> CreateModelCPU(TObjectPtr<UNNEModelData> ModelData) override;

private:
	TSharedPtr<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaEnvironment> Environment;
};
