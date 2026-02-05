/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "delay_inject_config.h"
#include "env_config.h"
#include "log.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Global delay injection configuration
DelayInjectConfig g_delay_inject_config;

// Helper function to get current timestamp in ISO 8601 format
static std::string get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm tm_now;
  localtime_r(&time_t_now, &tm_now);

  std::ostringstream oss;
  oss << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
}

bool DelayInjectConfig::save_to_file(const std::string& filepath) const {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    return false;
  }

  json config_json;
  config_json["version"] = version;
  config_json["delay_ns"] = delay_ns;

  json kernels_json = json::object();
  for (const auto& [kernel_name, kdc] : kernels) {
    json kernel_json;
    kernel_json["kernel_name"] = kdc.kernel_name;
    kernel_json["timestamp"] = kdc.timestamp;

    nlohmann::ordered_json points_json = nlohmann::ordered_json::object();
    for (const auto& [pc_offset, ip] : kdc.instrumentation_points) {
      nlohmann::ordered_json point_json;
      point_json["pc"] = ip.pc_offset;
      point_json["sass"] = ip.sass;
      point_json["delay"] = ip.delay_ns;
      point_json["on"] = ip.enabled;
      points_json[std::to_string(pc_offset)] = point_json;
    }
    kernel_json["instrumentation_points"] = points_json;
    kernels_json[kernel_name] = kernel_json;
  }
  config_json["kernels"] = kernels_json;

  file << config_json.dump(2);
  return true;
}

void init_delay_json_config() {
  g_delay_inject_config.version = "1.0";
  g_delay_inject_config.delay_ns = delay_ns;
  g_delay_inject_config.kernels.clear();
}

KernelDelayInjectConfig* create_kernel_delay_config(const std::string& kernel_name) {
  // Create new entry with timestamp-based key
  std::string timestamp = get_current_timestamp();
  // WORKAROUND: Use kernel_name + timestamp as key to handle multiple compilations
  // of the same kernel (e.g., during autotuning), each compilation gets a unique entry.
  // TODO: Use checksum-based key (e.g., hash of kernel binary) for more robust identification.
  std::string key = kernel_name + "_" + timestamp;

  KernelDelayInjectConfig kdc;
  kdc.kernel_name = kernel_name;
  kdc.timestamp = timestamp;
  g_delay_inject_config.kernels[key] = kdc;

  loprintf_v("Created delay config for kernel: %s (key: %s)\n", kernel_name.c_str(), key.c_str());
  return &g_delay_inject_config.kernels[key];
}

void register_delay_instrumentation_point(KernelDelayInjectConfig* kdc, Instr* instr, uint32_t delay_ns, bool enabled) {
  if (!kdc || !instr) {
    return;
  }

  uint64_t pc_offset = instr->getOffset();
  auto ret = kdc->instrumentation_points.emplace(pc_offset, DelayInstrumentationPoint());
  if (!ret.second) {
    return;  // Already registered
  }

  auto& ip = ret.first->second;
  ip.pc_offset = pc_offset;
  ip.sass = std::string(instr->getSass());
  ip.delay_ns = delay_ns;
  ip.enabled = enabled;
}

void finalize_delay_config() {
  if (delay_dump_path.empty()) {
    return;
  }

  loprintf_v("Saving delay config to %s (%zu kernels)\n", delay_dump_path.c_str(),
             g_delay_inject_config.kernels.size());

  if (!g_delay_inject_config.save_to_file(delay_dump_path)) {
    fprintf(stderr, "ERROR: Failed to save delay config to %s\n", delay_dump_path.c_str());
  }
}
