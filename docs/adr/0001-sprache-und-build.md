# ADR-0001: C++20, CMake, Linux und Windows

- **Status:** angenommen
- **Datum:** 2026-10-18

## Kontext
Hochleistungs-Raytracing mit CUDA, Nutzung unter Linux und Windows, Implementierung durch Programmieragenten.

## Entscheidung
C++20 für Host und Device, CMake ≥ 3.25 mit Presets, GoogleTest für Tests. Unterstützte Compiler: GCC ≥ 11, Clang ≥ 14, MSVC 2022. Abhängigkeiten werden per FetchContent mit fester Version eingebunden. Erlaubte externe Bibliotheken: GoogleTest, nlohmann/json, nanobind, CUDA-Bibliotheken (cuBLAS, cuSOLVER, cuFFT).

## Konsequenzen
Keine C++23-Features (z. B. `std::expected`). CI baut auf Linux und Windows.

## Verworfene Alternativen
Python mit Numba/CuPy (zu wenig Kontrolle über Datenlayout und Numerik), Rust (CUDA-Ökosystem weniger ausgereift).
