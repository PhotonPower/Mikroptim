#include "mikroptim/version.hpp"

#ifdef MIKROPTIM_HAS_CUDA
#include "mikroptim/cuda/device_info.hpp"
#endif

#include <cstdio>
#include <string_view>

namespace {

void print_usage() {
    std::puts("Aufruf: mikroptim [--version] [--devices]\n"
              "  --version   Version ausgeben\n"
              "  --devices   CUDA-Geräte auflisten");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }
    const std::string_view arg{argv[1]};
    if (arg == "--version") {
        std::printf("mikroptim %s\n", mko::kVersion);
        return 0;
    }
    if (arg == "--devices") {
#ifdef MIKROPTIM_HAS_CUDA
        return mko::cuda::print_devices() > 0 ? 0 : 1;
#else
        std::puts("Ohne CUDA gebaut (MIKROPTIM_ENABLE_CUDA=OFF).");
        return 1;
#endif
    }
    print_usage();
    return 1;
}
