#pragma once

#include "Modules/ModuleManager.h"
#include "UObject/WeakObjectPtr.h"

class UNNERuntimeORTCuda;

namespace UE::NNERuntimeORTCuda::Private
{
	class FOnnxRuntimeCudaEnvironment;
}

class FNNERuntimeORTCudaModule final : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
	virtual bool SupportsDynamicReloading() override { return false; }

private:
	TSharedPtr<UE::NNERuntimeORTCuda::Private::FOnnxRuntimeCudaEnvironment> Environment;
	TWeakObjectPtr<UNNERuntimeORTCuda> Runtime;
};
