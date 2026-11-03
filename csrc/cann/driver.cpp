/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "asnumpy/cann/driver.hpp"
#include <cstdlib>
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include "asnumpy/utils/device_memory_pool.hpp"
#include "asnumpy/utils/status_handler.hpp"
#include "fmt/format.h"

namespace {
spdlog::logger* g_logger = nullptr;
}

void asnumpy::cann::init_logging() {
    if (g_logger)
        return;

    auto logger = spdlog::stdout_color_mt("asnumpy");
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->flush_on(spdlog::level::warn);

    // Check ASNUMPY_DEBUG env var (coordinated with Python loguru)
    const char* debug_env = std::getenv("ASNUMPY_DEBUG");
    if (debug_env && std::string(debug_env) == "1") {
        logger->set_level(spdlog::level::debug);
    } else {
        logger->set_level(spdlog::level::warn);
    }

    // Check ASNUMPY_LOG_DIR for file sink (default: current working directory)
    const char* log_dir = std::getenv("ASNUMPY_LOG_DIR");
    std::string log_path =
        (log_dir && log_dir[0] != '\0') ? std::string(log_dir) + "/asnumpy_cpp.log" : "asnumpy_cpp.log";
    try {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path, false);
        file_sink->set_level(logger->level());
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
        logger->sinks().push_back(file_sink);
    } catch (const std::exception& e) {
        logger->warn("Failed to create log file sink: {}", e.what());
    }

    g_logger = logger.get();
    spdlog::set_default_logger(logger);
}

void asnumpy::cann::shutdown_logging() {
    spdlog::default_logger()->flush();
    spdlog::shutdown();
    g_logger = nullptr;
}

void asnumpy::cann::init() {
    init_logging();

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        auto message = aclGetRecentErrMsg();
        std::string detail = message ? std::string(" - ") + message : "";
        spdlog::error("[driver.cpp](init) aclInit error = {}{}", ret, detail);
        throw std::runtime_error(fmt::format("[driver.cpp](init) aclInit error = {}{}", ret, detail));
    }
    LOG_INFO("CANN backend initialized successfully");
}

void asnumpy::cann::finalize() {
    // Drain the pool BEFORE aclFinalize. Cached blocks are owned by the pool, not
    // by any live NPUArray, so nothing else would return them; leaving them
    // cached means aclFinalize runs with device memory still held.
    asnumpy::DeviceMemoryPool::Clear();

    auto ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        auto message = aclGetRecentErrMsg();
        std::string detail = message ? std::string(" - ") + message : "";
        spdlog::error("[driver.cpp](finalize) aclFinalize error = {}{}", ret, detail);
        throw std::runtime_error(fmt::format("[driver.cpp](finalize) aclFinalize error = {}{}", ret, detail));
    }
    LOG_INFO("CANN backend finalized");
    shutdown_logging();
}
