#pragma once

// Verteilung unabhängiger Arbeit pro Variante auf Host-Threads (intern, Mikroptim::optim).
// Feste, zusammenhängende Aufteilung; jede Variante wird von genau einem Thread bearbeitet,
// das Ergebnis hängt also nicht von der Threadzahl ab, solange `body` nur in die Ausgabe der
// eigenen Variante schreibt. Ohne std::execution::par, das unter GCC TBB bräuchte.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

namespace mko::detail {

// Zahl der Threads: `threads` > 0 wie angegeben, sonst hardware_concurrency (mindestens 1);
// höchstens `count`.
inline int thread_count(int threads, int count) {
    int t = threads;
    if (t <= 0) {
        t = static_cast<int>(std::thread::hardware_concurrency());
    }
    return std::max(1, std::min(t, count));
}

// Ruft body(v) für alle v in [first, first + count) auf. Thread c bearbeitet
// [first + ⌊count·c/T⌋, first + ⌊count·(c+1)/T⌋) aufsteigend und bricht beim ersten Fehler ab.
// Danach wird die Ausnahme des ersten Threads mit Fehler weitergeworfen; das ist die der
// kleinsten Variante, die eine Ausnahme wirft (unabhängig von T).
template <class Body>
void parallel_for_variants(int first, int count, int threads, const Body& body) {
    if (count <= 0) {
        return;
    }
    const int T = thread_count(threads, count);
    const auto bound = [&](int c) {
        return first + static_cast<int>(static_cast<std::int64_t>(count) * c / T);
    };
    if (T == 1) {
        for (int v = first; v < first + count; ++v) {
            body(v);
        }
        return;
    }
    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(T));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(T));
    const auto work = [&](int c) {
        try {
            for (int v = bound(c); v < bound(c + 1); ++v) {
                body(v);
            }
        } catch (...) {
            errors[static_cast<std::size_t>(c)] = std::current_exception();
        }
    };
    try {
        for (int c = 1; c < T; ++c) {
            pool.emplace_back(work, c);
        }
    } catch (...) {
        for (std::thread& th : pool) {
            th.join();
        }
        throw; // Thread konnte nicht gestartet werden
    }
    work(0); // Abschnitt 0 im aufrufenden Thread
    for (std::thread& th : pool) {
        th.join();
    }
    for (const std::exception_ptr& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

} // namespace mko::detail
