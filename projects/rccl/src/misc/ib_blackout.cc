/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Per-physical-NIC software blackout: blocks post_send, post_recv, and
 * poll_cq until a monotonic deadline. Inter-blackout intervals are
 * exponential; blackout length is fixed (RCCL_IB_BLACKOUT_DURATION_US).
 ************************************************************************/

#include "ib_blackout.h"
#include "core.h"
#include "param.h"

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <random>

RCCL_PARAM(IbBlackoutEnable, "IB_BLACKOUT_ENABLE", 0);
RCCL_PARAM(IbBlackoutMeanIntervalMs, "IB_BLACKOUT_MEAN_INTERVAL_MS", 450000);
RCCL_PARAM(IbBlackoutDurationUs, "IB_BLACKOUT_DURATION_US", 4);
RCCL_PARAM(IbBlackoutSeed, "IB_BLACKOUT_SEED", 1);
RCCL_PARAM(IbBlackoutDeterministic, "IB_BLACKOUT_DETERMINISTIC", 0);

struct BlackoutSlot {
  std::atomic<uint64_t> blackoutEndNs{0};
  std::atomic<int> shutdown{0};
  std::atomic<int> threadStarted{0};
  pthread_t thread{};
};

static BlackoutSlot g_slots[NCCL_IB_BLACKOUT_NLANES][NCCL_IB_BLACKOUT_MAX_DEVS];
static std::mutex g_initMutex[NCCL_IB_BLACKOUT_NLANES][NCCL_IB_BLACKOUT_MAX_DEVS];
static std::once_flag g_loggedOnce;

static uint64_t nowMonotonicNs() {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static uint64_t readDevUrandom64() {
  uint64_t x = 0;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f) {
    if (fread(&x, sizeof(x), 1, f) != 1) x = 0;
    fclose(f);
  }
  return x;
}

static uint64_t hashHostname() {
  char buf[256];
  if (gethostname(buf, sizeof(buf)) != 0) return 0;
  uint64_t h = 14695981039346656037ULL;
  for (const char* p = buf; *p; ++p) {
    h ^= (uint64_t)(unsigned char)*p;
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t makeBlackoutRngSeed(int lane, int devIndex) {
  uint64_t s = (uint64_t)rcclParamIbBlackoutSeed();
  s ^= (uint64_t)lane * 0x9e3779b97f4a7c15ULL;
  s ^= (uint64_t)devIndex * 0x85ebca6bUL;
  if (rcclParamIbBlackoutDeterministic()) {
    return s ? s : 1;
  }

  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
    ts.tv_sec = 0;
    ts.tv_nsec = 0;
  }
  s ^= readDevUrandom64();
  s ^= (uint64_t)getpid() * 0xc6a4a7935bd1e995ULL;
  s ^= (uint64_t)getppid() * 0x94d049bb133111ebULL;
  s ^= (uint64_t)(uintptr_t)pthread_self();
  s ^= (((uint64_t)(uint32_t)ts.tv_nsec) << 32) ^ (uint64_t)(uint32_t)ts.tv_sec;
  s ^= hashHostname();
  return s ? s : 1;
}

static void sleepNsInterruptible(uint64_t ns, std::atomic<int>* stop) {
  const uint64_t chunk = 100000000ULL; /* 100 ms */
  while (ns > 0 && stop->load(std::memory_order_relaxed) == 0) {
    uint64_t thisSleep = ns > chunk ? chunk : ns;
    struct timespec rq;
    rq.tv_sec = (time_t)(thisSleep / 1000000000ULL);
    rq.tv_nsec = (long)(thisSleep % 1000000000ULL);
    nanosleep(&rq, NULL);
    ns -= thisSleep;
  }
}

static void* blackoutThreadMain(void* arg) {
  uintptr_t packed = (uintptr_t)arg;
  int lane = (int)(packed >> 16);
  int devIndex = (int)(packed & 0xffff);

  const int64_t meanMs = rcclParamIbBlackoutMeanIntervalMs();
  const int64_t durUs = rcclParamIbBlackoutDurationUs();
  const double mean = (double)(meanMs < 1 ? 1 : meanMs);
  const uint64_t durNs = (uint64_t)(durUs < 0 ? 0 : durUs) * 1000ULL;

  std::mt19937_64 gen(makeBlackoutRngSeed(lane, devIndex));
  std::exponential_distribution<double> dist(1.0 / mean);

  BlackoutSlot* slot = &g_slots[lane][devIndex];

  while (slot->shutdown.load(std::memory_order_relaxed) == 0) {
    double idleMs = dist(gen);
    if (!(idleMs >= 0.0) || idleMs > 86400000.0) idleMs = 86400000.0;
    uint64_t idleNs = (uint64_t)std::llround(idleMs * 1000000.0);
    sleepNsInterruptible(idleNs, &slot->shutdown);
    if (slot->shutdown.load(std::memory_order_relaxed) != 0) break;

    const uint64_t t0 = nowMonotonicNs();
    const uint64_t end = t0 + durNs;
    slot->blackoutEndNs.store(end, std::memory_order_release);

    while (nowMonotonicNs() < end) {
      if (slot->shutdown.load(std::memory_order_relaxed) != 0) break;
      struct timespec rq = {0, 500000};
      nanosleep(&rq, NULL);
    }
    slot->blackoutEndNs.store(0ULL, std::memory_order_release);
  }
  return NULL;
}

void ncclIbBlackoutLaneDevInit(int lane, int ibDevIndex) {
  if (lane < 0 || lane >= NCCL_IB_BLACKOUT_NLANES) return;
  if (ibDevIndex < 0 || ibDevIndex >= NCCL_IB_BLACKOUT_MAX_DEVS) return;
  if (!rcclParamIbBlackoutEnable()) return;

  std::lock_guard<std::mutex> lock(g_initMutex[lane][ibDevIndex]);
  BlackoutSlot* slot = &g_slots[lane][ibDevIndex];
  if (slot->threadStarted.load(std::memory_order_acquire) != 0) return;

  std::call_once(g_loggedOnce, []() {
    INFO(NCCL_INIT | NCCL_NET,
         "NET/IB: RCCL_IB_BLACKOUT_ENABLE=1 (mean interval %lld ms, duration %lld us, seed %lld, deterministic=%lld)",
         (long long)rcclParamIbBlackoutMeanIntervalMs(),
         (long long)rcclParamIbBlackoutDurationUs(),
         (long long)rcclParamIbBlackoutSeed(),
         (long long)rcclParamIbBlackoutDeterministic());
  });

  slot->shutdown.store(0, std::memory_order_relaxed);
  slot->blackoutEndNs.store(0, std::memory_order_relaxed);
  uintptr_t arg = ((uintptr_t)(unsigned)lane << 16) | (uintptr_t)(unsigned)ibDevIndex;
  int pterr = pthread_create(&slot->thread, NULL, blackoutThreadMain, (void*)arg);
  if (pterr != 0) {
    WARN("NET/IB: RCCL blackout pthread_create failed for lane %d dev %d: %d", lane, ibDevIndex, pterr);
    return;
  }
  slot->threadStarted.store(1, std::memory_order_release);
}

void ncclIbBlackoutLaneStop(int lane) {
  if (lane < 0 || lane >= NCCL_IB_BLACKOUT_NLANES) return;
  for (int d = 0; d < NCCL_IB_BLACKOUT_MAX_DEVS; d++) {
    BlackoutSlot* slot = &g_slots[lane][d];
    if (slot->threadStarted.load(std::memory_order_acquire) == 0) continue;
    slot->shutdown.store(1, std::memory_order_release);
    slot->blackoutEndNs.store(0, std::memory_order_release);
    pthread_join(slot->thread, NULL);
    slot->shutdown.store(0, std::memory_order_relaxed);
    slot->blackoutEndNs.store(0, std::memory_order_relaxed);
    slot->threadStarted.store(0, std::memory_order_release);
  }
}

void ncclIbBlackoutLaneWait(int lane, int ibDevN) {
  if (!rcclParamIbBlackoutEnable()) return;
  if (lane < 0 || lane >= NCCL_IB_BLACKOUT_NLANES) return;
  if (ibDevN < 0 || ibDevN >= NCCL_IB_BLACKOUT_MAX_DEVS) return;

  BlackoutSlot* slot = &g_slots[lane][ibDevN];
  if (slot->threadStarted.load(std::memory_order_acquire) == 0) return;

  for (;;) {
    uint64_t end = slot->blackoutEndNs.load(std::memory_order_acquire);
    if (end == 0) return;

    uint64_t now = nowMonotonicNs();
    if (now >= end) return;

    uint64_t remain = end - now;
    if (remain > 1000000ULL) {
      struct timespec rq;
      rq.tv_sec = (time_t)(remain / 1000000000ULL);
      rq.tv_nsec = (long)(remain % 1000000000ULL);
      if (rq.tv_nsec >= 1000000000L) {
        rq.tv_sec++;
        rq.tv_nsec -= 1000000000L;
      }
      nanosleep(&rq, NULL);
    } else {
      sched_yield();
    }
  }
}
