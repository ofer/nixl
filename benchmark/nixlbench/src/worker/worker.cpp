/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 */

#include "worker.h"
#include "benchmark_config.h"
#include "runtime/asio_runtime.h"
#include "runtime/etcd/etcd_rt.h"
#include "runtime/null_rt.h"

#include <sstream>
#include <unistd.h>

namespace nixlbench {

bool
usesNullRuntime(const benchmarkConfig &config) {
    return isStorageBackend(config.backend) && config.runtime.etcd_endpoints.empty();
}

int
runtimeWorldSize(const benchmarkConfig &config) {
    if (isStorageBackend(config.backend)) {
        return 1;
    }
    if (config.transfer.mode == XFERBENCH_MODE_SG) {
        return config.worker.num_initiator_dev + config.worker.num_target_dev;
    }
    return 2;
}

std::string
rankRoleName(const benchmarkConfig &config, int rank) {
    if (usesNullRuntime(config)) {
        return "initiator";
    }
    if (config.transfer.mode == XFERBENCH_MODE_SG) {
        if (rank >= 0 && rank < config.worker.num_initiator_dev) {
            return "initiator";
        }
        return "target";
    }
    if (config.transfer.mode == XFERBENCH_MODE_MG) {
        if (0 == rank) {
            return "initiator";
        }
        return "target";
    }
    return "";
}

std::vector<std::string>
parseWorkerDeviceList(const benchmarkConfig &config) {
    std::vector<std::string> devices;
    std::string dev;
    std::stringstream ss(config.worker.device_list);

    // TODO: Add support for other schemes
    if (config.transfer.scheme == XFERBENCH_SCHEME_PAIRWISE && config.worker.device_list != "all") {
        while (std::getline(ss, dev, ',')) {
            devices.push_back(dev);
        }

        if ((int)devices.size() != config.worker.num_initiator_dev ||
            (int)devices.size() != config.worker.num_target_dev) {
            std::cerr << "Incorrect device list " << config.worker.device_list
                      << " provided for pairwise scheme " << devices.size() << "# devices"
                      << std::endl;
            return {};
        }
    } else {
        devices.push_back("all");
    }

    return devices;
}

} // namespace nixlbench

namespace {

static xferBenchRT *
createRT(const nixlbench::benchmarkConfig &config, std::atomic<int> *terminate) {
    // For storage backends without ETCD endpoints, use null runtime
    if (nixlbench::usesNullRuntime(config)) {
        std::cout << "Using null runtime for storage backend without ETCD" << std::endl;
        return new xferBenchNullRT();
    }

    int total = nixlbench::runtimeWorldSize(config);

#if HAVE_ETCD
    if (XFERBENCH_RT_ETCD == config.runtime.type) {
        xferBenchEtcdRT *etcd_rt = new xferBenchEtcdRT(
            config.common.benchmark_group, config.runtime.etcd_endpoints, total, terminate);
        if (etcd_rt->setup() != 0) {
            std::cerr << "Failed to setup ETCD runtime" << std::endl;
            delete etcd_rt;
            exit (EXIT_FAILURE);
        }
        return etcd_rt;
    }
#endif

    if (config.runtime.type == XFERBENCH_RT_ASIO) {
        if (total != 2) {
            std::cerr << "Invalid total " << total << " for ASIO runtime -- supports only 2"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        return new xferBenchAsioRT(config.asio_address, config.asio_port);
    }

    std::cerr << "Invalid runtime: " << config.runtime.type << std::endl;
    exit(EXIT_FAILURE);
}

} // namespace

int xferBenchWorker::synchronize() {
    // For storage backends without ETCD, no synchronization needed
    if (nixlbench::usesNullRuntime(benchmark_config)) {
        return 0;
    }

    if (rt->barrier("sync") != 0) {
        std::cerr << "Failed to synchronize" << std::endl;
        // Best-effort cleanup of non-leased etcd keys (e.g. the "size" key)
        // so they don't poison subsequent runs in the same benchmark_group.
        rt->cleanupForExit();
        // Use _Exit() instead of exit() to bypass atexit handlers (e.g. gRPC shutdown).
        // exit() would deadlock with the etcd KeepAlive background thread: gRPC shutdown
        // waits for open streams to close, but the KeepAlive thread keeps renewing the
        // lease stream indefinitely. _Exit() kills all threads immediately, which closes
        // the gRPC stream and lets the lease expire on the etcd server side.
        std::_Exit(EXIT_FAILURE);
    }

    return 0;
}

xferBenchWorker::xferBenchWorker(const nixlbench::benchmarkConfig &config)
    : benchmark_config(config), config(nixlbench::makeLegacyConfigFromBenchmarkConfig(config)) {
    terminate = 0;

    rt = createRT(config, &terminate);
    if (!rt) {
        std::cerr << "Failed to create runtime object" << std::endl;
        exit(EXIT_FAILURE);
    }

    int rank = rt->getRank();

    name = nixlbench::rankRoleName(config, rank);

    // Set the RT for utils
    xferBenchUtils::setRT(rt);
}

xferBenchWorker::~xferBenchWorker() {
    delete rt;
}

std::string xferBenchWorker::getName() const {
    return name;
}

bool xferBenchWorker::isMasterRank() {
    return (0 == rt->getRank());
}

bool xferBenchWorker::isInitiator() {
    return ("initiator" == name);
}

bool xferBenchWorker::isTarget() {
    return ("target" == name);
}

static_assert(std::atomic<int>::is_always_lock_free,
              "xferBenchWorker::terminate must be lock-free for safe use in signal handlers");

std::atomic<int> xferBenchWorker::terminate = 0;

void xferBenchWorker::signalHandler(int signal) {
    static const char msg[] = "Ctrl-C received, exiting...\n";
    constexpr int stdout_fd = 1;
    constexpr int max_count = 1;
    auto size = write(stdout_fd, msg, sizeof(msg) - 1);
    (void)size;

    if (terminate.fetch_add(1) >= max_count) {
        std::_Exit(EXIT_FAILURE);
    }
}
