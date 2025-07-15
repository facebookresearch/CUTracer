/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Meta's license terms.
 * See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;

enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};
volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

/* lock */
pthread_mutex_t cuda_event_mutex;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

std::map<int, std::string> id_to_sass_map;

// Original NVIDIA implementation
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
  pthread_mutex_lock(&cuda_event_mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&cuda_event_mutex);
    return;
  }
  skip_callback_flag = true;

  /* Identify all the possible CUDA launch events */
  if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchGrid ||
      cbid == API_CUDA_cuLaunchGridAsync || cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
      cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
    /* cast params to launch parameter based on cbid since if we are here
     * we know these are the right parameters types */
    CUfunction func;
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      func = p->f;
    } else {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      func = p->f;
    }

    if (!is_exit) {
      /* Make sure GPU is idle */
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      int nregs = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

      nvbit_enable_instrumented(ctx, func, true);

      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
        printf(
            "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
            "%d - shmem %d - cuda stream id %ld\n",
            nvbit_get_func_name(ctx, func), p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ,
            p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ, nregs,
            shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream);
      } else {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        printf(
            "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
            "%d - shmem %d - cuda stream id %ld\n",
            nvbit_get_func_name(ctx, func), p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
            p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
      }
    } else {
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      cudaError_t kernelError = cudaGetLastError();
      if (kernelError != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
        assert(0);
      }

      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);
    }
  }
  skip_callback_flag = false;
  pthread_mutex_unlock(&cuda_event_mutex);
}

// Original NVIDIA implementation
void nvbit_tool_init(CUcontext ctx) {
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&cuda_event_mutex, &attr);

  recv_thread_done = RecvThreadState::WORKING;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, recv_thread_fun, NULL);
  nvbit_set_tool_pthread(channel_host.get_thread());
}

// Original NVIDIA implementation
void nvbit_at_ctx_term(CUcontext ctx) {
  skip_callback_flag = true;
  /* Notify receiver thread and wait for receiver thread to
   * notify back */
  recv_thread_done = RecvThreadState::STOP;
  while (recv_thread_done != RecvThreadState::FINISHED);
  channel_host.destroy(false);
  skip_callback_flag = false;
}
