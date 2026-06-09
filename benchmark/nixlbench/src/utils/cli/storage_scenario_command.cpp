/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/storage_scenario_command.h"

#include "benchmark/benchmark_executor.h"
#include "benchmark/benchmark_runtime_sync.h"
#include "benchmark/nixl_storage_allocator.h"
#include "benchmark_config.h"
#include "runtime/null_rt.h"
#include "utils/scope_guard.h"
#include "utils/utils.h"
#include "worker/nixl/nixl_backend_params.h"

#include <nixl.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <string>
#include <string_view>
#include <utility>
#include <unistd.h>
#include <vector>

namespace nixlbench {
namespace {

    std::atomic<int> storage_terminate{0};

    void
    storageSignalHandler(int signal) {
        (void)signal;
        static const char msg[] = "Ctrl-C received, exiting...\n";
        constexpr int stdout_fd = 1;
        constexpr int max_count = 1;
        auto size = write(stdout_fd, msg, sizeof(msg) - 1);
        (void)size;

        if (++storage_terminate > max_count) {
            std::_Exit(EXIT_FAILURE);
        }
    }

    bool
    storageSignaled() {
        return storage_terminate.load() != 0;
    }

    nixl_status_t
    executeSingleTransfer(nixlAgent &agent,
                          nixlXferReqH *req,
                          xferBenchTimer &timer,
                          xferBenchStats &thread_stats) {
        nixl_status_t rc = agent.postXferReq(req);
        thread_stats.post_duration.add(timer.lap());
        while (!storageSignaled() && NIXL_IN_PROG == rc) {
            rc = agent.getXferStatus(req);
        }
        return storageSignaled() ? NIXL_ERR_UNKNOWN : rc;
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
            return EXIT_SUCCESS;
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
            if (storageSignaled()) {
                agent.releaseXferReq(req);
                return EXIT_FAILURE;
            }

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

    void
    printDebugIovs(const std::vector<std::vector<xferBenchIOV>> &iovs, const std::string &header) {
        std::cout << header << ":\n";
        for (size_t i = 0; i < iovs.size(); ++i) {
            std::cout << "  Thread " << i << " IOVs:\n";

            size_t max_offset = 0;
            for (const auto &iov : iovs[i]) {
                std::cout << "    addr: " << std::dec << iov.addr << std::dec
                          << ", len: " << iov.len << ", devId: " << iov.devId
                          << ", metaInfo: " << iov.metaInfo << "\n";

                if (iov.addr + iov.len > max_offset) {
                    max_offset = iov.addr + iov.len;
                }
            }

            std::cout << "    Total size covered by IOVs: " << max_offset << "\n";
        }
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

        if (false) {
            printDebugIovs(local_iovs, "Local IOVs");
            printDebugIovs(remote_iovs, "Remote IOVs");
        }

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

    class storageNixlTransferStrategy : public benchmarkTransferStrategy {
    public:
        storageNixlTransferStrategy(nixlAgent &agent,
                                    const benchmarkConfig &config,
                                    remoteIovStrategy &remote_strategy,
                                    int transfers_per_execute)
            : agent_(agent),
              config_(config),
              remote_strategy_(remote_strategy),
              transfers_per_execute_(transfers_per_execute) {}

        std::variant<xferBenchStats, int>
        execute(const std::vector<std::vector<xferBenchIOV>> &local_descriptors) override {
            if (storageSignaled()) {
                return EXIT_FAILURE;
            }

            auto remote_result = remote_strategy_.createTransferIovs(
                local_descriptors, config_.transfer.start_block_size);
            if (std::holds_alternative<int>(remote_result)) {
                return std::get<int>(remote_result);
            }
            auto remote_descriptors =
                std::get<std::vector<std::vector<xferBenchIOV>>>(std::move(remote_result));

            xferBenchStats stats;
            const nixl_xfer_op_t xfer_op =
                config_.transfer.op_type == XFERBENCH_OP_READ ? NIXL_READ : NIXL_WRITE;

            int ret = executeTransfer(agent_,
                                      DRAM_SEG,
                                      remote_strategy_.segmentType(),
                                      local_descriptors,
                                      remote_descriptors,
                                      xfer_op,
                                      transfers_per_execute_,
                                      config_.transfer.num_threads,
                                      stats);

            if (ret != EXIT_SUCCESS) {
                return ret;
            }

            if (storageSignaled()) {
                return EXIT_FAILURE;
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
        int transfers_per_execute_;
    };

    class storageStatsResultSink : public benchmarkResultSink {
    public:
        explicit storageStatsResultSink(benchmarkConfig config) : config_(std::move(config)) {
            stats_.clear();
        }

        void
        record(const xferBenchStats &stats) override {
            total_duration_us_ += stats.total_duration.avg();
            stats_.add(stats);
            ++record_count_;
        }

        void
        print() {
            if (record_count_ == 0) {
                return;
            }

            stats_.total_duration.clear();
            stats_.total_duration.add(total_duration_us_);
            xferBenchUtils::printStats(config_,
                                       false,
                                       config_.transfer.start_block_size,
                                       config_.transfer.start_batch_size,
                                       stats_);
        }

    private:
        benchmarkConfig config_;
        xferBenchStats stats_;
        double total_duration_us_ = 0.0;
        int record_count_ = 0;
    };

    benchmarkConfig
    makeStorageBenchmarkConfig(const storageScenarioRequest &request,
                               southboundPluginBenchmarkCommand &plugin) {
        const auto &metadata = plugin.metadataOptions();
        benchmarkConfig config;
        config.common.num_iter = request.num_iter;
        config.backend.name = std::string(plugin.name());
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
        config.storage.num_files = metadata.intOption("num_files", 1);
        config.storage.enable_direct = metadata.boolOption("enable_direct");
        return config;
    }

} // namespace

storageScenarioCommand::storageScenarioCommand(std::string name,
                                               std::string description,
                                               scenario_type_t scenario_type,
                                               benchmarkAllocationLifecycle allocation_lifecycle)
    : name_(std::move(name)),
      description_(std::move(description)),
      scenario_type_(scenario_type),
      allocation_lifecycle_(allocation_lifecycle),
      options_{
          cliOption::option("file-size",
                            "File size, can be shorthand (5M, 10G ...)",
                            &request_.file_size,
                            false),
          cliOption::option("parallel-threads",
                            "Parallel threads",
                            &request_.parallel_threads,
                            false),
          cliOption::option("block-size",
                            "Block size - amount of data to transfer in each transfer in bytes",
                            &request_.block_size_bytes,
                            false),
          cliOption::option("batch-size",
                            "Batch size - number of data transfers to perform in each batch",
                            &request_.batch_size,
                            false),
          cliOption::option("num_iter,num-iterations",
                            "Number of times to run the execution loop",
                            &request_.num_iter),
          cliOption::option(
              "action-mode",
              "Sets whether the benchmark will read, write, or interleave reading and writing",
              &request_.action_mode,
              false),
          cliOption::option("randomized-read-location",
                            "Whether to read / write in random locations or sequentially",
                            &request_.randomized_read_location,
                            false)} {}

std::string_view
storageScenarioCommand::name() const {
    return name_;
}

std::string_view
storageScenarioCommand::description() const {
    return description_;
}

const std::vector<cliOption> &
storageScenarioCommand::getOptions() const {
    return options_;
}

scenario_type_t
storageScenarioCommand::scenarioType() const {
    return scenario_type_;
}

bool
storageScenarioCommand::supportsPlugin(nixl_mem_list_t supportedMemoryTypes) const {
    return std::find(supportedMemoryTypes.begin(), supportedMemoryTypes.end(), FILE_SEG) !=
        supportedMemoryTypes.end();
}

request_key_value_pairs_t
storageScenarioCommand::requestKeyValues() const {
    return request_.toKeyValuePairs();
}

bool
storageScenarioCommand::isRequestValid(const storageScenarioRequest &request) const {
    const size_t file_size = parseFileSize(request.file_size);
    if (file_size == 0) {
        return false;
    }

    if (request.batch_size == 0 || request.batch_size > file_size) {
        return false;
    }

    if (request.block_size_bytes == 0) {
        return false;
    }

    if (request.parallel_threads <= 0) {
        return false;
    }

    if (request.num_iter <= 0) {
        return false;
    }

    return true;
}

int
storageScenarioCommand::run(southboundPluginBenchmarkCommand &plugin) {
    if (!supportsPlugin(plugin.supportedMemoryTypes())) {
        std::cerr << "Storage scenarios require a plugin that can read and write files"
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (!isRequestValid(request_)) {
        return EXIT_FAILURE;
    }

    storage_terminate.store(0);
    auto previous_signal_handler = std::signal(SIGINT, storageSignalHandler);
    auto signal_guard = make_scope_guard(
        [previous_signal_handler] { std::signal(SIGINT, previous_signal_handler); });

    benchmarkConfig benchmark_config = makeStorageBenchmarkConfig(request_, plugin);
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

    const int transfer_iterations = request_.num_iter;
    const bool allocate_once = allocation_lifecycle_ == benchmarkAllocationLifecycle::AllocateOnce;
    const int executor_iterations = allocate_once ? 1 : transfer_iterations;
    const int transfers_per_execute = allocate_once ? transfer_iterations : 1;

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
    storageNixlTransferStrategy transfer(
        agent, benchmark_config, remote_iovs, transfers_per_execute);
    fixedIterationPolicy iterations(executor_iterations, allocation_lifecycle_);
    storageStatsResultSink results(benchmark_config);

    xferBenchUtils::printStatsHeader(benchmark_config);
    benchmarkRunComponents components{sync, allocator, *descriptors, transfer, iterations, results};
    benchmarkExecutor executor;
    int ret = executor.run(components);
    if (storageSignaled()) {
        return EXIT_FAILURE;
    }
    if (ret == EXIT_SUCCESS) {
        results.print();
    }

    return ret;
}

const storageScenarioRequest &
storageScenarioCommand::request() const {
    return request_;
}

allocateOnceScenarioCommand::allocateOnceScenarioCommand()
    : storageScenarioCommand(
          "allocate-once",
          "Run allocate-once storage scenario. This simulates the G3 scenario by opening and "
          "registering storage once, reusing it for all transfers, and closing it at the end.",
          scenario_type_t::ALLOCATE_ONCE,
          benchmarkAllocationLifecycle::AllocateOnce) {}

allocatePerIterationScenarioCommand::allocatePerIterationScenarioCommand()
    : storageScenarioCommand(
          "allocate-per-iteration",
          "Run allocate-per-iteration storage scenario. This simulates the G4 scenario by opening, "
          "registering, transferring, deregistering, and closing storage each iteration.",
          scenario_type_t::ALLOCATE_PER_ITERATION,
          benchmarkAllocationLifecycle::AllocatePerIteration) {}

} // namespace nixlbench
