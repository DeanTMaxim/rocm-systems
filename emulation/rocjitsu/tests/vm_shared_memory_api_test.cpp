// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "uapi/linux/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
template <typename Function> Function load_function(void *library, const char *name) {
  auto function = reinterpret_cast<Function>(dlsym(library, name));
  EXPECT_NE(function, nullptr) << name << ": " << dlerror();
  return function;
}

TEST(VmSharedMemoryApiTest, MapsBorrowedHandlesWithoutPreloading) {
  ASSERT_EQ(std::fflush(stdout), 0);
  const bool zero_limit = std::getenv("RJ_TEST_HARD_ZERO") != nullptr;
  const bool explicit_process = std::getenv("RJ_TEST_EXPLICIT_PROCESS") != nullptr;
  if (zero_limit) {
    const rlimit limit{};
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limit), 0);
  }
  ASSERT_EQ(dlsym(RTLD_DEFAULT, "rj_vm_create"), nullptr);
  // Keep the interposer's process-lifetime state loaded until process exit.
  void *library = dlopen(RJ_VM_API_LIBRARY, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  ASSERT_NE(library, nullptr) << dlerror();
  ASSERT_EQ(dlsym(RTLD_DEFAULT, "rj_vm_create"), nullptr);
  const auto create = load_function<decltype(&rj_vm_create)>(library, "rj_vm_create");
  const auto destroy = load_function<decltype(&rj_vm_destroy)>(library, "rj_vm_destroy");
  const auto gpu_id = load_function<decltype(&rj_vm_gpu_id)>(library, "rj_vm_gpu_id");
  const auto open_device =
      load_function<decltype(&rj_vm_device_open)>(library, "rj_vm_device_open");
  const auto close_device =
      load_function<decltype(&rj_vm_device_close)>(library, "rj_vm_device_close");
  const auto execute = load_function<decltype(&rj_vm_execute)>(library, "rj_vm_execute");
  const auto execute_as = load_function<decltype(&rj_vm_execute_as)>(library, "rj_vm_execute_as");
  const auto get_shared =
      load_function<decltype(&rj_vm_get_shared_mem)>(library, "rj_vm_get_shared_mem");
  const auto get_shared_as =
      load_function<decltype(&rj_vm_get_shared_mem_as)>(library, "rj_vm_get_shared_mem_as");
  const auto map_shared =
      load_function<decltype(&rj_vm_map_shared_mem)>(library, "rj_vm_map_shared_mem");
  ASSERT_FALSE(HasFailure());

  rj_vm_t *raw_vm = nullptr;
  ASSERT_EQ(create(RJ_VM_API_CONFIG, RJ_VM_MODE_DAEMON, &raw_vm), ROCJITSU_STATUS_SUCCESS);
  const std::unique_ptr<rj_vm_t, decltype(destroy)> vm(raw_vm, destroy);
  uint32_t process_id = 0;
  if (explicit_process) {
    ASSERT_EQ(open_device(vm.get(), getpid(), &process_id), ROCJITSU_STATUS_SUCCESS);
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  kfd_ioctl_alloc_memory_of_gpu_args allocation{};
  allocation.size = 2 * page;
  ASSERT_EQ(gpu_id(vm.get(), &allocation.gpu_id), ROCJITSU_STATUS_SUCCESS);
  allocation.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  rj_vm_cmd_t command{};
  command.cmd = AMDKFD_IOC_ALLOC_MEMORY_OF_GPU;
  command.buf = &allocation;
  command.buf_size = sizeof(allocation);
  command.shared_handle = command.in_handle = command.in_mem_handle = command.in_proc_handle = -1;
  ASSERT_EQ(explicit_process ? execute_as(vm.get(), process_id, &command)
                             : execute(vm.get(), &command),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(command.result, 0);
  ASSERT_GE(command.shared_handle, 0);
  rj_handle_t handle = -1;
  ASSERT_EQ(explicit_process ? get_shared_as(vm.get(), process_id, allocation.mmap_offset, &handle)
                             : get_shared(vm.get(), allocation.mmap_offset, &handle),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_GE(handle, 0);
  struct stat kind {};
  ASSERT_EQ(fstat(handle, &kind), 0);
  EXPECT_EQ(static_cast<bool>(S_ISSOCK(kind.st_mode)), zero_limit);
  if (zero_limit) {
    EXPECT_EQ(mmap(nullptr, page, PROT_READ, MAP_SHARED, handle, 0), MAP_FAILED);
    EXPECT_EQ(errno, ENODEV);
  }

  rj_vm_map_t mapping{};
  mapping.length = allocation.size;
  mapping.prot = PROT_READ | PROT_WRITE;
  mapping.flags = MAP_SHARED;
  ASSERT_EQ(map_shared(command.shared_handle, &mapping), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(mapping.map_errno, 0);
  auto *whole = reinterpret_cast<unsigned char *>(mapping.mapped_addr);
  whole[page] = 0x37;
  mapping.length = page;
  mapping.offset = page;
  mapping.addr = 0;
  ASSERT_EQ(map_shared(handle, &mapping), ROCJITSU_STATUS_SUCCESS);
  auto *tail = reinterpret_cast<unsigned char *>(mapping.mapped_addr);
  EXPECT_EQ(tail[0], 0x37);
  tail[0] = 0xA5;
  EXPECT_EQ(whole[page], 0xA5);
  EXPECT_GE(fcntl(handle, F_GETFD), 0);
  EXPECT_EQ(munmap(tail, page), 0);

  mapping.offset = 1;
  EXPECT_EQ(map_shared(handle, &mapping), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(reinterpret_cast<void *>(mapping.mapped_addr), MAP_FAILED);
  EXPECT_EQ(mapping.map_errno, EINVAL);
  mapping.offset = 0;
  EXPECT_EQ(map_shared(-1, &mapping), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(mapping.map_errno, EBADF);
  EXPECT_EQ(map_shared(handle, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = allocation.handle;
  command.cmd = AMDKFD_IOC_FREE_MEMORY_OF_GPU;
  command.buf = &free_args;
  command.buf_size = sizeof(free_args);
  ASSERT_EQ(explicit_process ? execute_as(vm.get(), process_id, &command)
                             : execute(vm.get(), &command),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(command.result, 0);
  EXPECT_EQ(whole[page], 0xA5);
  EXPECT_EQ(munmap(whole, allocation.size), 0);
  if (explicit_process) {
    EXPECT_EQ(close_device(vm.get(), process_id), ROCJITSU_STATUS_SUCCESS);
  }
}
} // namespace
