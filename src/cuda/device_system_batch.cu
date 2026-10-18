// SystemBatch im Gerätespeicher: ein Puffer, Upload und Download pro Feld.

#include "mikroptim/cuda/device_info.hpp"
#include "mikroptim/cuda/device_system_batch.hpp"

#include "cuda_check.hpp"
#include "device_guard.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mko::cuda {

namespace {

// Kopiert region.bytes Byte von `src` (Host) an base + region.offset (Gerät).
void to_device(char* base, const BufferRegion& region, const void* src) {
    if (region.bytes == 0) {
        return;
    }
    MKO_CUDA_CHECK(cudaMemcpy(base + region.offset, src, region.bytes, cudaMemcpyHostToDevice));
}

// Kopiert region.bytes Byte von base + region.offset (Gerät) nach `dst` (Host).
void to_host(const char* base, const BufferRegion& region, void* dst) {
    if (region.bytes == 0) {
        return;
    }
    MKO_CUDA_CHECK(cudaMemcpy(dst, base + region.offset, region.bytes, cudaMemcpyDeviceToHost));
}

template <class T>
const T* region_ptr(const void* buffer, const BufferRegion& region) {
    return reinterpret_cast<const T*>(static_cast<const char*>(buffer) + region.offset);
}

} // namespace

DeviceSystemBatch::DeviceSystemBatch(const SystemBatchHost& host, int device) : device_(device) {
    upload(host);
}

DeviceSystemBatch::~DeviceSystemBatch() {
    release();
}

DeviceSystemBatch::DeviceSystemBatch(DeviceSystemBatch&& other) noexcept
    : device_(other.device_), buffer_(other.buffer_), capacity_(other.capacity_),
      loaded_(other.loaded_), topology_(std::move(other.topology_)), variants_(other.variants_),
      layout_(other.layout_), reference_wavelength_nm_(other.reference_wavelength_nm_),
      source_point_(other.source_point_), source_na_(other.source_na_) {
    other.buffer_ = nullptr;
    other.capacity_ = 0;
    other.loaded_ = false;
    other.variants_ = 0;
}

DeviceSystemBatch& DeviceSystemBatch::operator=(DeviceSystemBatch&& other) noexcept {
    if (this != &other) {
        release();
        device_ = other.device_;
        buffer_ = other.buffer_;
        capacity_ = other.capacity_;
        loaded_ = other.loaded_;
        topology_ = std::move(other.topology_);
        variants_ = other.variants_;
        layout_ = other.layout_;
        reference_wavelength_nm_ = other.reference_wavelength_nm_;
        source_point_ = other.source_point_;
        source_na_ = other.source_na_;
        other.buffer_ = nullptr;
        other.capacity_ = 0;
        other.loaded_ = false;
        other.variants_ = 0;
    }
    return *this;
}

void DeviceSystemBatch::release() noexcept {
    loaded_ = false;
    variants_ = 0;
    if (buffer_ == nullptr) {
        return;
    }
    try {
        const detail::DeviceGuard guard(device_);
        cudaFree(buffer_); // Fehler beim Freigeben sind nicht behebbar
    } catch (...) {
        cudaFree(buffer_);
    }
    buffer_ = nullptr;
    capacity_ = 0;
}

void DeviceSystemBatch::upload(const SystemBatchHost& host, int first, int count) {
    const int total = host.variants();
    if (first < 0 || first > total) {
        throw std::out_of_range("DeviceSystemBatch::upload: erste Variante " +
                                std::to_string(first) + " außerhalb von [0, " +
                                std::to_string(total) + "]");
    }
    if (count < 0) {
        count = total - first;
    }
    if (count > total - first) {
        throw std::out_of_range("DeviceSystemBatch::upload: Varianten [" + std::to_string(first) +
                                ", " + std::to_string(first) + " + " + std::to_string(count) +
                                ") außerhalb des Batches (" + std::to_string(total) + ")");
    }
    const SystemBatchLayout layout = make_batch_layout(host.topology(), count);

    loaded_ = false;
    variants_ = 0;
    if (device_ < 0) {
        device_ = detail::current_device();
    }
    const detail::DeviceGuard guard(device_);
    if (layout.total_bytes > capacity_) {
        if (buffer_ != nullptr) {
            MKO_CUDA_CHECK(cudaFree(buffer_));
            buffer_ = nullptr;
            capacity_ = 0;
        }
        MKO_CUDA_CHECK(cudaMalloc(&buffer_, layout.total_bytes));
        capacity_ = layout.total_bytes;
    }

    char* base = static_cast<char*>(buffer_);
    const BatchShape sh = host.shape();
    const auto v0 = static_cast<std::size_t>(first);
    to_device(base, layout.type, host.type_data().data());
    to_device(base, layout.ghost_surface, host.ghost_surface_data().data());
    to_device(base, layout.ghost_norm_radius, host.ghost_norm_radius_data().data());
    to_device(base, layout.wavelength_nm, host.wavelength_nm_data().data());
    to_device(base, layout.wavelength_weight, host.wavelength_weight_data().data());
    to_device(base, layout.field_weight, host.field_weight_data().data());
    to_device(base, layout.curvature, host.curvature_data().data() + v0 * sh.surface_stride());
    to_device(base, layout.thickness, host.thickness_data().data() + v0 * sh.surface_stride());
    to_device(base, layout.index, host.index_data().data() + v0 * sh.index_stride());
    to_device(base, layout.pupil_radius, host.pupil_radius_data().data() + v0);
    to_device(base, layout.field_angle, host.field_angle_data().data() + v0 * sh.field_stride());
    to_device(base, layout.field_target, host.field_target_data().data() + v0 * sh.field_stride());
    to_device(base, layout.ghost_coeff,
              host.ghost_coeff_data().data() + v0 * sh.ghost_coeff_stride());
    to_device(base, layout.ghost_index,
              host.ghost_index_data().data() + v0 * sh.ghost_index_stride());

    topology_ = host.topology();
    variants_ = count;
    layout_ = layout;
    reference_wavelength_nm_ = host.reference_wavelength_nm;
    source_point_ = host.source_point;
    source_na_ = host.source_na;
    loaded_ = true;
}

void DeviceSystemBatch::download(SystemBatchHost& host, int first) const {
    if (!loaded_) {
        throw std::logic_error("DeviceSystemBatch::download: Container ist leer");
    }
    if (!(host.topology() == topology_)) {
        throw std::invalid_argument("DeviceSystemBatch::download: andere Topologie im Ziel");
    }
    if (first < 0 || first > host.variants() - variants_) {
        throw std::out_of_range("DeviceSystemBatch::download: Ziel hat " +
                                std::to_string(host.variants()) + " Varianten, gebraucht " +
                                std::to_string(first) + " + " + std::to_string(variants_));
    }
    const detail::DeviceGuard guard(device_);
    const char* base = static_cast<const char*>(buffer_);

    // Typen und Geisterplatten-Flächen folgen aus der Topologie: zurücklesen und vergleichen.
    std::vector<SurfaceKind> type(static_cast<std::size_t>(topology_.surfaces));
    std::vector<int> ghost_surface(static_cast<std::size_t>(topology_.ghost_plates));
    to_host(base, layout_.type, type.data());
    to_host(base, layout_.ghost_surface, ghost_surface.data());
    if (type != topology_.type ||
        !std::equal(ghost_surface.begin(), ghost_surface.end(), host.ghost_surface_data().begin(),
                    host.ghost_surface_data().end())) {
        throw CudaError("DeviceSystemBatch::download: Topologie im Gerätespeicher verändert");
    }

    const BatchShape sh = host.shape();
    const auto v0 = static_cast<std::size_t>(first);
    to_host(base, layout_.ghost_norm_radius, host.ghost_norm_radius_data().data());
    to_host(base, layout_.wavelength_nm, host.wavelength_nm_data().data());
    to_host(base, layout_.wavelength_weight, host.wavelength_weight_data().data());
    to_host(base, layout_.field_weight, host.field_weight_data().data());
    to_host(base, layout_.curvature, host.curvature_data().data() + v0 * sh.surface_stride());
    to_host(base, layout_.thickness, host.thickness_data().data() + v0 * sh.surface_stride());
    to_host(base, layout_.index, host.index_data().data() + v0 * sh.index_stride());
    to_host(base, layout_.pupil_radius, host.pupil_radius_data().data() + v0);
    to_host(base, layout_.field_angle, host.field_angle_data().data() + v0 * sh.field_stride());
    to_host(base, layout_.field_target, host.field_target_data().data() + v0 * sh.field_stride());
    to_host(base, layout_.ghost_coeff,
            host.ghost_coeff_data().data() + v0 * sh.ghost_coeff_stride());
    to_host(base, layout_.ghost_index,
            host.ghost_index_data().data() + v0 * sh.ghost_index_stride());
    host.reference_wavelength_nm = reference_wavelength_nm_;
    host.source_point = source_point_;
    host.source_na = source_na_;
}

SystemBatchHost DeviceSystemBatch::download() const {
    if (!loaded_) {
        throw std::logic_error("DeviceSystemBatch::download: Container ist leer");
    }
    SystemBatchHost host(topology_, variants_);
    download(host, 0);
    return host;
}

SystemBatchView<double> DeviceSystemBatch::view() const {
    if (!loaded_) {
        throw std::logic_error("DeviceSystemBatch::view: Container ist leer");
    }
    SystemBatchView<double> b{};
    b.shape = BatchShape{variants_,        topology_.surfaces,     topology_.wavelengths,
                         topology_.fields, topology_.ghost_plates, topology_.ghost_terms};
    b.source = topology_.source;
    b.source_point[0] = source_point_[0];
    b.source_point[1] = source_point_[1];
    b.source_point[2] = source_point_[2];
    b.source_na = source_na_;
    b.reference_wavelength_nm = reference_wavelength_nm_;
    b.type = region_ptr<SurfaceKind>(buffer_, layout_.type);
    b.ghost_surface = region_ptr<int>(buffer_, layout_.ghost_surface);
    b.ghost_norm_radius = region_ptr<double>(buffer_, layout_.ghost_norm_radius);
    b.wavelength_nm = region_ptr<double>(buffer_, layout_.wavelength_nm);
    b.wavelength_weight = region_ptr<double>(buffer_, layout_.wavelength_weight);
    b.field_weight = region_ptr<double>(buffer_, layout_.field_weight);
    b.curvature = region_ptr<double>(buffer_, layout_.curvature);
    b.thickness = region_ptr<double>(buffer_, layout_.thickness);
    b.index = region_ptr<double>(buffer_, layout_.index);
    b.pupil_radius = region_ptr<double>(buffer_, layout_.pupil_radius);
    b.field_angle = region_ptr<double>(buffer_, layout_.field_angle);
    b.field_target = region_ptr<double>(buffer_, layout_.field_target);
    b.ghost_coeff = region_ptr<double>(buffer_, layout_.ghost_coeff);
    b.ghost_index = region_ptr<double>(buffer_, layout_.ghost_index);
    return b;
}

int max_variants_on_device(const BatchTopology& topology, const OutputRequest& outputs,
                           const MemoryReserve& reserve, std::size_t extra_bytes_per_variant,
                           int device) {
    return max_variants(device_memory_info(device).free_bytes, topology, outputs, reserve,
                        extra_bytes_per_variant);
}

} // namespace mko::cuda
