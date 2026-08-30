#pragma once

namespace UE::NNERuntimeORTCuda::Private
{
	inline constexpr uint32 SerializedOnnxModelMagic = 0x5544434f;
	inline constexpr uint32 SerializedOnnxModelVersion = 1;
	inline constexpr int64 SerializedOnnxModelHeaderSize = sizeof(SerializedOnnxModelMagic) + sizeof(SerializedOnnxModelVersion);
}
