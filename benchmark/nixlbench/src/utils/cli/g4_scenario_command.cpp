/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/g4_scenario_command.h"

#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runtime_sync.h"
#include "benchmark/nixl_storage_allocator.h"
#include "benchmark/transfer_descriptor_strategy.h"
#include "worker/nixl/nixl_backend_params.h"
#include "nixl.h"
#include "runtime/null_rt.h"
#include "utils/utils.h"
#include <algorithm>
#include <iostream>
#include <atomic>
#include <omp.h>

namespace nixlbench {
namespace {

    //     std::atomic<int> g3_terminate{0};

    //     void
    //     g3SignalHandler(int signal) {
    //         (void)signal;
    //         static const char msg[] = "Ctrl-C received, exiting...\n";
    //         constexpr int stdout_fd = 1;
    //         constexpr int max_count = 1;
    //         auto size = write(stdout_fd, msg, sizeof(msg) - 1);
    //         (void)size;

    //         if (++g3_terminate > max_count) {
    //             std::_Exit(EXIT_FAILURE);
    //         }
    //     }

    //     bool
    //     g3Signaled() {
    //         return g3_terminate.load() != 0;
    //     }

    nixl_status_t
    executeSingleTransfer(nixlAgent &agent,
                          nixlXferReqH *req,
                          xferBenchTimer &timer,
                          xferBenchStats &thread_stats) {
        nixl_status_t rc = agent.postXferReq(req);
        thread_stats.post_duration.add(timer.lap());
        // while (!g3Signaled() && NIXL_IN_PROG == rc) {
        rc = agent.getXferStatus(req);
        // }
        return rc;
        // return g3Signaled() ? NIXL_ERR_UNKNOWN : rc;
    }

    int
    executeTransferIterations(nixlAgent &agent,
                              const nixl_xfer_op_t op,
                              nixl_xfer_dlist_t &local_desc,
                              nixl_xfer_dlist_t &remote_desc,
                              const std::string &target,
                              nixl_opt_args_t &params,
                              int num_iter,
                              xferBenchTimer &timer,
                              xferBenchStats &thread_stats) {
        if (num_iter <= 0) {
            return 0;
        }

        nixlXferReqH *req = nullptr;
        nixl_status_t create_rc =
            agent.createXferReq(op, local_desc, remote_desc, target, req, &params);
        if (NIXL_SUCCESS != create_rc) {
            std::cerr << "createXferReq failed: " << nixlEnumStrings::statusStr(create_rc)
                      << std::endl;
            return EXIT_FAILURE;
        }
        thread_stats.prepare_duration.add(timer.lap());

        for (int i = 0; i < num_iter; ++i) {
            // if (g3Signaled()) {
            //     agent.releaseXferReq(req);
            //     return EXIT_FAILURE;
            // }

            nixl_status_t rc = executeSingleTransfer(agent, req, timer, thread_stats);
            if (rc != NIXL_SUCCESS) {
                std::cout << "NIXL Xfer failed with status: " << nixlEnumStrings::statusStr(rc)
                          << std::endl;
                agent.releaseXferReq(req);
                return EXIT_FAILURE;
            }
            thread_stats.transfer_duration.add(timer.lap());
        }

        if (agent.releaseXferReq(req) != NIXL_SUCCESS) {
            std::cout << "NIXL releaseXferReq failed" << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    int
    executeTransfer(nixlAgent &agent,
                    nixl_mem_t local_segment_type,
                    nixl_mem_t remote_segment_type,
                    const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                    const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
                    nixl_xfer_op_t op,
                    int num_iter,
                    int num_threads,
                    xferBenchStats &stats) {
        std::atomic<int> ret{EXIT_SUCCESS};
        stats.clear();

        xferBenchTimer total_timer;
#pragma omp parallel num_threads(num_threads)
        {
            xferBenchStats thread_stats;
            thread_stats.reserve(static_cast<size_t>(std::max(num_iter, 0)));
            xferBenchTimer timer;
            const int tid = omp_get_thread_num();
            const auto &local_iov = local_iovs[tid];
            const auto &remote_iov = remote_iovs[tid];

            nixl_xfer_dlist_t local_desc(local_segment_type);
            nixl_xfer_dlist_t remote_desc(remote_segment_type);
            iovListToNixlXferDlist(local_iov, local_desc);
            iovListToNixlXferDlist(remote_iov, remote_desc);

            nixl_opt_args_t params;
            const int result = executeTransferIterations(agent,
                                                         op,
                                                         local_desc,
                                                         remote_desc,
                                                         "initiator",
                                                         params,
                                                         num_iter,
                                                         timer,
                                                         thread_stats);
            if (result != EXIT_SUCCESS) {
                ret.store(result);
            }

#pragma omp critical
            { stats.add(thread_stats); }
        }

        stats.total_duration.add(total_timer.lap());
        return ret.load();
    }

    class g4NixlTransferStrategy : public benchmarkTransferStrategy {
    public:
        g4NixlTransferStrategy(nixlAgent &agent,
                               const benchmarkConfig &config,
                               remoteIovStrategy &remote_strategy)
            : agent_(agent),
              config_(config),
              remote_strategy_(remote_strategy) {}

        std::variant<xferBenchStats, int>
        execute(const std::vector<std::vector<xferBenchIOV>> &local_descriptors) override {
            auto remote_result = remote_strategy_.createTransferIovs(
                local_descriptors, config_.transfer.start_block_size);
            if (std::holds_alternative<int>(remote_result)) {
                return std::get<int>(remote_result);
            }
            auto remote_descriptors =
                std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(remote_result));

            int num_iter = config_.common.num_iter / config_.transfer.num_threads;
            int warmup_iter = config_.common.warmup_iter / config_.transfer.num_threads;
            if (config_.transfer.start_block_size > LARGE_BLOCK_SIZE) {
                num_iter /= config_.common.large_blk_iter_ftr;
                warmup_iter /= config_.common.large_blk_iter_ftr;
            }

            xferBenchStats stats;
            const nixl_xfer_op_t xfer_op =
                config_.transfer.op_type == XFERBENCH_OP_READ ? NIXL_READ : NIXL_WRITE;
            int ret = executeTransfer(agent_,
                                      DRAM_SEG,
                                      FILE_SEG,
                                      local_descriptors,
                                      remote_descriptors,
                                      xfer_op,
                                      warmup_iter,
                                      config_.transfer.num_threads,
                                      stats);
            if (ret != EXIT_SUCCESS) {
                return ret;
            }

            stats.clear();
            ret = executeTransfer(agent_,
                                  DRAM_SEG,
                                  FILE_SEG,
                                  local_descriptors,
                                  remote_descriptors,
                                  xfer_op,
                                  num_iter,
                                  config_.transfer.num_threads,
                                  stats);
            if (ret != EXIT_SUCCESS) {
                return ret;
            }

            auto local_validation_descriptors = local_descriptors;
            if (!xferBenchUtils::validateTransfer(
                    config_, true, local_validation_descriptors, remote_descriptors)) {
                return EXIT_FAILURE;
            }

            return stats;
        }

    private:
        nixlAgent &agent_;
        benchmarkConfig config_;
        remoteIovStrategy &remote_strategy_;
    };

    class g4StatsResultSink : public benchmarkResultSink {
    public:
        explicit g4StatsResultSink(benchmarkConfig config) : config_(std::move(config)) {}

        void
        record(const xferBenchStats &stats) override {
            xferBenchUtils::printStats(config_,
                                       false,
                                       config_.transfer.start_block_size,
                                       config_.transfer.start_batch_size,
                                       stats);
        }

    private:
        benchmarkConfig config_;
    };

    benchmarkConfig
    makeG4BenchmarkConfig(const g4ScenarioRequest &request,
                          southboundPluginBenchmarkCommand &plugin) {
        const auto &metadata = plugin.metadataOptions();
        benchmarkConfig config;
        config.backend.name = std::string(plugin.name());
        config.backend.capabilities = plugin.capabilities();
        config.backend.memory_types = plugin.supportedMemoryTypes();
        config.backend.options = metadata;
        config.transfer.num_threads = request.parallel_threads;
        config.transfer.start_block_size = request.block_size_bytes;
        config.transfer.max_block_size = request.block_size_bytes;
        config.transfer.start_batch_size = request.batch_size;
        config.transfer.max_batch_size = request.batch_size;
        config.transfer.op_type = request.action_mode == "read" || request.action_mode == "READ" ?
            XFERBENCH_OP_READ :
            XFERBENCH_OP_WRITE;
        config.transfer.total_buffer_size = parseFileSize(request.file_size);
        config.storage.filepath = metadata.stringOption("filepath");
        config.storage.filenames = metadata.stringOption("filenames");
        config.storage.num_files = request.parallel_threads * metadata.intOption("num_files", 1);
        config.storage.enable_direct = metadata.boolOption("enable_direct");
        if (config.backend.capabilities.requiresDirectStorage) {
            config.storage.enable_direct = true;
        }
        return config;
    }

} // namespace

g4ScenarioCommand::g4ScenarioCommand()
    : options_{
          cliOption::option("file_size", "File size", &request_.file_size, true),
          cliOption::option("num_files", "Number of files", &request_.num_files, true),
          cliOption::option("block_size_bytes", "Block size", &request_.block_size_bytes, true),
          cliOption::option("parallel_threads",
                            "Parallel threads",
                            &request_.parallel_threads,
                            true),
          cliOption::option("action_mode", "Action mode", &request_.action_mode, true),
          cliOption::option("randomized_read_location",
                            "Randomized read location",
                            &request_.randomized_read_location,
                            true),
          cliOption::option("batch_size", "Batch size", &request_.batch_size, true),
          cliOption::option("open_behavior", "Open behavior", &request_.open_behavior, true)} {}

std::string_view
g4ScenarioCommand::name() const {
    return "g4";
}

std::string_view
g4ScenarioCommand::description() const {
    return "Run G4 key-value scenario";
}

const std::vector<cliOption> &
g4ScenarioCommand::getOptions() const {
    return options_;
}

scenario_type_t
g4ScenarioCommand::scenarioType() const {
    return scenario_type_t::G4;
}

bool
g4ScenarioCommand::supportsPlugin(nixl_mem_list_t supportedMemoryTypes,
                                  nixlBackendPluginCapabilities pluginCapabilities) const {
    // the g4 scenario only runs on files and objects
    if (std::find(supportedMemoryTypes.begin(), supportedMemoryTypes.end(), FILE_SEG) ==
            supportedMemoryTypes.end() &&
        std::find(supportedMemoryTypes.begin(), supportedMemoryTypes.end(), OBJ_SEG) ==
            supportedMemoryTypes.end()) {
        return false;
    }
    return true;
}

request_key_value_pairs_t
g4ScenarioCommand::requestKeyValues() const {
    return request_.toKeyValuePairs();
}

bool
g4ScenarioCommand::isRequestValid(const g4ScenarioRequest &request) const {
    // validate file size
    const size_t file_size = parseFileSize(request.file_size);
    if (file_size == 0) {
        std::cerr << "Invalid file size: " << request.file_size << " must be greater than zero;"
                  << std::endl;
        return false;
    }

    if (request.batch_size < 1) {
        std::cerr << "Invalid batch size: " << request.batch_size << " must be greater than 1"
                  << std::endl;
        return false;
    }

    if (request.block_size_bytes < 0) {
        std::cerr << "Invalid block size: " << request.block_size_bytes
                  << " must be greater than zero" << std::endl;
        return false;
    }

    if (request.parallel_threads <= 0) {
        std::cerr << "Invalid number of parallel threads: " << request.parallel_threads
                  << " must be greater than zero" << std::endl;
        return false;
    }
    if (request.num_files < 1) {
        std::cerr << "Invalid number of files: " << request.num_files
                  << " must be greater than zero" << std::endl;
        return false;
    }

    if (request.open_behavior != "open-close" && request.open_behavior != "keep-open") {
        std::cerr << "Invalid open behavior: " << request.open_behavior
                  << " must be open-close or keep-open" << std::endl;
        return false;
    }

    return true;
}

int
g4ScenarioCommand::run(southboundPluginBenchmarkCommand &plugin) {
    // this should  never occur as the CLI should only present things that have the proper
    // capabilities, but this is here just in case...
    if (!supportsPlugin(plugin.supportedMemoryTypes(), plugin.capabilities())) {
        std::cerr << "G4 requires a plugin that can read and write files or objects" << std::endl;
        return 1;
    }

    if (!isRequestValid(request_)) {
        return 1;
    }

    // g3_terminate.store(0);
    // auto previous_signal_handler = std::signal(SIGINT, g3SignalHandler);
    // auto signal_guard = make_scope_guard(
    //     [previous_signal_handler] { std::signal(SIGINT, previous_signal_handler); });

    benchmarkConfig benchmark_config = makeG4BenchmarkConfig(request_, plugin);
    xferBenchNullRT runtime;
    xferBenchUtils::setRT(&runtime);
    std::cout << "Single instance storage backend - no synchronization needed" << std::endl;

    nixlAgentConfig agent_config;
    agent_config.syncMode = benchmark_config.transfer.num_threads > 1 ?
        nixl_thread_sync_t::NIXL_THREAD_SYNC_RW :
        nixl_thread_sync_t::NIXL_THREAD_SYNC_DEFAULT;
    nixlAgent agent("initiator", agent_config);

    nixl_mem_list_t mems;
    nixl_b_params_t backend_params;
    nixl_status_t status =
        agent.getPluginParams(benchmark_config.backend.name, mems, backend_params);
    if (status != NIXL_SUCCESS) {
        std::cerr << "getPluginParams failed: " << nixlEnumStrings::statusStr(status) << std::endl;
        return EXIT_FAILURE;
    }

    backend_params = applyPluginOptions(benchmark_config.backend.options, backend_params);
    nixlBackendH *backend = nullptr;
    status = agent.createBackend(benchmark_config.backend.name, backend_params, backend);
    if (status != NIXL_SUCCESS) {
        std::cerr << "createBackend failed: " << nixlEnumStrings::statusStr(status) << std::endl;
        return EXIT_FAILURE;
    }

    nullBenchmarkRuntimeSync sync;
    dramLocalIovStrategy local_iovs;
    fileRemoteIovStrategy remote_iovs(
        benchmark_config.storage, benchmark_config.backend.name, benchmark_config.transfer.op_type);
    nixlStorageAllocator allocator(agent,
                                   backend,
                                   benchmark_config.transfer.num_threads,
                                   benchmark_config.transfer.total_buffer_size,
                                   benchmark_config.storage.enable_direct,
                                   local_iovs,
                                   remote_iovs);

    auto descriptors =
        makeTransferDescriptorStrategy(benchmark_config, request_.randomized_read_location);
    g4NixlTransferStrategy transfer(agent, benchmark_config, remote_iovs);
    fixedIterationPolicy iterations(1, benchmarkAllocationLifecycle::AllocateOnce);
    g4StatsResultSink results(benchmark_config);

    xferBenchUtils::printStatsHeader(benchmark_config);
    benchmarkRunComponents components{sync, allocator, *descriptors, transfer, iterations, results};
    benchmarkExecutor executor;
    int ret = executor.run(components);
    // if (g3Signaled()) {
    //     return EXIT_FAILURE;
    // }

    return ret;
}

const g4ScenarioRequest &
g4ScenarioCommand::request() const {
    return request_;
}

} // namespace nixlbench
