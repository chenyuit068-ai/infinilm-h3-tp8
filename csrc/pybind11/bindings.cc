#include <pybind11/pybind11.h>

#include "cache/cache.hpp"
#include "engine/engine.hpp"
#include "../models/minimax_h3/h3_nine_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_infinilm, m) {
    m.doc() = "InfiniLM Python bindings";

    infinilm::cache::bind_cache(m);
    infinilm::engine::bind_hook_registry(m);
    infinilm::engine::distributed::bind_dist_config(m);
    infinilm::engine::bind_infer_engine(m);
    infinilm::h3_nine::bind(m);
    m.def("_h3_streaming_storage_available", [] { return true; });
    m.def("_h3_trim_memory", [] {
        pybind11::gil_scoped_release release;
        infinicore::context::syncDevice();
        infinicore::context::trimMemory();
    });
}
