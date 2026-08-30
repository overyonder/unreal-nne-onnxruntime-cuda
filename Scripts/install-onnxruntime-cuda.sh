#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plugin_directory="$(cd -- "${script_directory}/.." && pwd)"
download_directory="${plugin_directory}/.third-party-downloads"
header_directory="${plugin_directory}/Source/ThirdParty/OnnxRuntimeCuda/include"
library_directory="${plugin_directory}/Binaries/ThirdParty/OnnxRuntimeCuda/Linux/x86_64"
license_directory="${plugin_directory}/ThirdPartyLicenses"

download_file_and_verify_sha256() {
	local download_url="$1"
	local destination_path="$2"
	local expected_sha256="$3"
	local partial_destination_path="${destination_path}.partial"

	if [[ -f "${destination_path}" ]] && printf '%s  %s\n' "${expected_sha256}" "${destination_path}" | sha256sum --check --status; then
		printf '%s: OK\n' "${destination_path}"
		return
	fi

	rm -f -- "${partial_destination_path}"
	curl --fail --location --output "${partial_destination_path}" "${download_url}"
	printf '%s  %s\n' "${expected_sha256}" "${partial_destination_path}" | sha256sum --check
	mv -- "${partial_destination_path}" "${destination_path}"
}

for required_command in curl sha256sum python3 install find patchelf; do
	command -v "${required_command}" >/dev/null || {
		printf 'Required command is unavailable: %s\n' "${required_command}" >&2
		exit 1
	}
done

mkdir -p "${download_directory}" "${header_directory}" "${library_directory}" "${license_directory}"

onnxruntime_archive="${download_directory}/microsoft.ml.onnxruntime.gpu.linux.1.24.3.nupkg"
download_file_and_verify_sha256 \
	"https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.linux/1.24.3/microsoft.ml.onnxruntime.gpu.linux.1.24.3.nupkg" \
	"${onnxruntime_archive}" \
	"c8689f44907f8ff700a27195b9cf8f9e13a1e23e1acea2d6094ebbe2b5dfe877"

nvidia_cuda_runtime_archive="${download_directory}/nvidia_cuda_runtime_cu12-12.8.90.whl"
nvidia_cuda_nvrtc_archive="${download_directory}/nvidia_cuda_nvrtc_cu12-12.8.93.whl"
nvidia_cublas_archive="${download_directory}/nvidia_cublas_cu12-12.8.4.1.whl"
nvidia_cufft_archive="${download_directory}/nvidia_cufft_cu12-11.3.3.83.whl"
nvidia_curand_archive="${download_directory}/nvidia_curand_cu12-10.3.9.90.whl"
nvidia_nvjitlink_archive="${download_directory}/nvidia_nvjitlink_cu12-12.8.93.whl"
nvidia_cudnn_archive="${download_directory}/nvidia_cudnn_cu12-9.8.0.87.whl"

download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/0d/9b/a997b638fcd068ad6e4d53b8551a7d30fe8b404d6f1804abf1df69838932/nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
	"${nvidia_cuda_runtime_archive}" \
	"adade8dcbd0edf427b7204d480d6066d33902cab2a4707dcfc48a2d0fd44ab90"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/05/6b/32f747947df2da6994e999492ab306a903659555dddc0fbdeb9d71f75e52/nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl" \
	"${nvidia_cuda_nvrtc_archive}" \
	"a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/dc/61/e24b560ab2e2eaeb3c839129175fb330dfcfc29e5203196e5541a4c44682/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl" \
	"${nvidia_cublas_archive}" \
	"8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/1f/13/ee4e00f30e676b66ae65b4f08cb5bcbb8392c03f54f2d5413ea99a5d1c80/nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" \
	"${nvidia_cufft_archive}" \
	"4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/fb/aa/6584b56dc84ebe9cf93226a5cde4d99080c8e90ab40f0c27bda7a0f29aa1/nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl" \
	"${nvidia_curand_archive}" \
	"b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/f6/74/86a07f1d0f42998ca31312f998bd3b9a7eff7f52378f4f270c8679c77fb9/nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl" \
	"${nvidia_nvjitlink_archive}" \
	"81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88"
download_file_and_verify_sha256 \
	"https://files.pythonhosted.org/packages/77/f0/8236c886a061d203e51247aec2b8e3a8f5350178251ab57237daf2140680/nvidia_cudnn_cu12-9.8.0.87-py3-none-manylinux_2_27_x86_64.whl" \
	"${nvidia_cudnn_archive}" \
	"d6b02cd0e3e24aa31d0193a8c39fec239354360d7d81055edddb69f35d53a4c8"

temporary_directory="$(mktemp -d)"
trap 'rm -rf -- "${temporary_directory}"' EXIT

python3 -m zipfile -e "${onnxruntime_archive}" "${temporary_directory}/onnxruntime-package"

onnxruntime_header_directory="${temporary_directory}/onnxruntime-package/buildTransitive/native/include"
for onnxruntime_public_header_path in "${onnxruntime_header_directory}"/*.h; do
	install -m 0644 "${onnxruntime_public_header_path}" "${header_directory}/"
done
install -m 0644 "${temporary_directory}/onnxruntime-package/LICENSE" "${license_directory}/ONNXRuntime-LICENSE.txt"
install -m 0644 "${temporary_directory}/onnxruntime-package/ThirdPartyNotices.txt" "${license_directory}/ONNXRuntime-ThirdPartyNotices.txt"

onnxruntime_native_directory="${temporary_directory}/onnxruntime-package/runtimes/linux-x64/native"
install -m 0755 "${onnxruntime_native_directory}/libonnxruntime.so" "${library_directory}/libonnxruntime_cuda_for_unreal.so.1"
install -m 0755 "${onnxruntime_native_directory}/libonnxruntime_providers_shared.so" "${library_directory}/"
install -m 0755 "${onnxruntime_native_directory}/libonnxruntime_providers_cuda.so" "${library_directory}/"

for nvidia_archive in \
	"${nvidia_cuda_runtime_archive}" \
	"${nvidia_cuda_nvrtc_archive}" \
	"${nvidia_cublas_archive}" \
	"${nvidia_cufft_archive}" \
	"${nvidia_curand_archive}" \
	"${nvidia_nvjitlink_archive}" \
	"${nvidia_cudnn_archive}"; do
	archive_name="$(basename -- "${nvidia_archive}" .whl)"
	python3 -m zipfile -e "${nvidia_archive}" "${temporary_directory}/${archive_name}"
	while IFS= read -r -d '' shared_library_path; do
		install -m 0755 "${shared_library_path}" "${library_directory}/$(basename -- "${shared_library_path}")"
	done < <(find "${temporary_directory}/${archive_name}/nvidia" -type f -name '*.so*' -print0)
done

install -m 0644 \
	"${temporary_directory}/nvidia_cuda_runtime_cu12-12.8.90/nvidia_cuda_runtime_cu12-12.8.90.dist-info/License.txt" \
	"${license_directory}/NVIDIA-CUDA-License.txt"
install -m 0644 \
	"${temporary_directory}/nvidia_cudnn_cu12-9.8.0.87/nvidia_cudnn_cu12-9.8.0.87.dist-info/License.txt" \
	"${license_directory}/NVIDIA-cuDNN-License.txt"

patchelf --set-soname libonnxruntime_cuda_for_unreal.so.1 "${library_directory}/libonnxruntime_cuda_for_unreal.so.1"
for shared_library_path in "${library_directory}"/*.so*; do
	patchelf --set-rpath '$ORIGIN' "${shared_library_path}"
done

printf 'Installed ONNX Runtime CUDA 1.24.3, CUDA 12.8, and Pascal-compatible cuDNN 9.8 into %s\n' "${library_directory}"
