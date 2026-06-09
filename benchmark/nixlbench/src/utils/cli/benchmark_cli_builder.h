/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_BENCHMARK_CLI_BUILDER_H
#define NIXLBENCH_BENCHMARK_CLI_BUILDER_H

#include "utils/cli/benchmark_command.h"
#include "utils/cli/storage_scenario_command.h"
#include "utils/cli/raw_command.h"

#include <memory>
#include <vector>

namespace nixlbench {

class benchmarkCliBuilder {
public:
    benchmarkCliBuilder();

    int
    parse(int argc, char **argv);

    int
    run();

    bool
    helpRequested() const;

    const benchmarkScenario *
    selectedScenario() const;

    const southboundPluginBenchmarkCommand *
    selectedPlugin() const;

private:
    allocateOnceScenarioCommand allocateOnce_;
    allocatePerIterationScenarioCommand allocatePerIteration_;
    rawCommand raw_;

    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> allocateOncePlugins_;
    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> allocatePerIterationPlugins_;
    std::vector<std::unique_ptr<southboundPluginBenchmarkCommand>> rawPlugins_;

    benchmarkScenario *selectedScenario_ = nullptr;
    southboundPluginBenchmarkCommand *selectedPlugin_ = nullptr;
    bool help_ = false;
};

} // namespace nixlbench

#endif // NIXLBENCH_BENCHMARK_CLI_BUILDER_H
