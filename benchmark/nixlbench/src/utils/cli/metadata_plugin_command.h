/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXLBENCH_METADATA_PLUGIN_COMMAND_H
#define NIXLBENCH_METADATA_PLUGIN_COMMAND_H

#include "utils/cli/benchmark_command.h"

namespace nixlbench {

class metadataPluginCommand : public southboundPluginBenchmarkCommand {
public:
    metadataPluginCommand(std::string backend_name,
                          nixl_b_params_t option_specs,
                          nixl_mem_list_t supported_memory_types);

    std::string_view
    name() const override;

    std::string_view
    description() const override;

    const std::vector<cliOption> &
    getOptions() const override;

    const metadataPluginOptionMap &
    metadataOptions() const override;

    const nixl_mem_list_t &
    supportedMemoryTypes() const override;

    request_key_value_pairs_t
    requestKeyValues() const override;

private:
    std::string name_;
    std::string description_;
    nixl_b_params_t optionSpecs_;
    nixl_mem_list_t supportedMemoryTypes_;
    metadataPluginOptionMap optionValues_;
    std::vector<cliOption> options_;
};

} // namespace nixlbench

#endif // NIXLBENCH_METADATA_PLUGIN_COMMAND_H
