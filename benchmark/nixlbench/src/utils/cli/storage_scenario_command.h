/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_STORAGE_SCENARIO_COMMAND_H
#define NIXLBENCH_STORAGE_SCENARIO_COMMAND_H

#include "benchmark/transfer_descriptor_strategy.h"
#include "nixl_types.h"
#include "utils/cli/benchmark_command.h"

#include <string>
#include <string_view>
#include <vector>

namespace nixlbench {

class storageScenarioCommand : public benchmarkScenario {
public:
    storageScenarioCommand(std::string name,
                           std::string description,
                           scenario_type_t scenario_type,
                           benchmarkAllocationLifecycle allocation_lifecycle);

    std::string_view
    name() const override;
    std::string_view
    description() const override;
    const std::vector<cliOption> &
    getOptions() const override;
    scenario_type_t
    scenarioType() const override;
    bool
    supportsPlugin(nixl_mem_list_t supportedMemoryTypes) const override;
    request_key_value_pairs_t
    requestKeyValues() const override;
    int
    run(southboundPluginBenchmarkCommand &plugin) override;
    const storageScenarioRequest &
    request() const;

private:
    bool
    isRequestValid(const storageScenarioRequest &request) const;

    std::string name_;
    std::string description_;
    scenario_type_t scenario_type_;
    benchmarkAllocationLifecycle allocation_lifecycle_;
    storageScenarioRequest request_;
    std::vector<cliOption> options_;
};

class allocateOnceScenarioCommand final : public storageScenarioCommand {
public:
    allocateOnceScenarioCommand();
};

class allocatePerIterationScenarioCommand final : public storageScenarioCommand {
public:
    allocatePerIterationScenarioCommand();
};

} // namespace nixlbench

#endif // NIXLBENCH_STORAGE_SCENARIO_COMMAND_H
