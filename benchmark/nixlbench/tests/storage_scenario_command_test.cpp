/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/cli/storage_scenario_command.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace nixlbench {
namespace {

    std::vector<std::string>
    optionNames(const benchmarkScenario &command) {
        std::vector<std::string> names;
        for (const auto &option : command.getOptions()) {
            names.push_back(option.name);
        }
        return names;
    }

    TEST(StorageScenarioCommandTest, AllocateOnceReplacesG3Naming) {
        allocateOnceScenarioCommand command;

        EXPECT_EQ(std::string(command.name()), "allocate-once");
        EXPECT_NE(std::string(command.description()).find("G3"), std::string::npos);
        EXPECT_EQ(command.scenarioType(), scenario_type_t::ALLOCATE_ONCE);
        EXPECT_TRUE(command.supportsPlugin(nixl_mem_list_t{FILE_SEG}));
        EXPECT_FALSE(command.supportsPlugin(nixl_mem_list_t{DRAM_SEG}));
    }

    TEST(StorageScenarioCommandTest, AllocatePerIterationReplacesG4Naming) {
        allocatePerIterationScenarioCommand command;

        EXPECT_EQ(std::string(command.name()), "allocate-per-iteration");
        EXPECT_NE(std::string(command.description()).find("G4"), std::string::npos);
        EXPECT_EQ(command.scenarioType(), scenario_type_t::ALLOCATE_PER_ITERATION);
        EXPECT_TRUE(command.supportsPlugin(nixl_mem_list_t{FILE_SEG}));
        EXPECT_FALSE(command.supportsPlugin(nixl_mem_list_t{OBJ_SEG}));
    }

    TEST(StorageScenarioCommandTest, StorageScenariosShareG3StyleOptions) {
        allocateOnceScenarioCommand allocate_once;
        allocatePerIterationScenarioCommand allocate_per_iteration;

        const std::vector<std::string> expected_options{"file-size",
                                                        "parallel-threads",
                                                        "block-size",
                                                        "batch-size",
                                                        "action-mode",
                                                        "randomized-read-location"};
        EXPECT_EQ(optionNames(allocate_once), expected_options);
        EXPECT_EQ(optionNames(allocate_per_iteration), expected_options);
    }

} // namespace
} // namespace nixlbench
