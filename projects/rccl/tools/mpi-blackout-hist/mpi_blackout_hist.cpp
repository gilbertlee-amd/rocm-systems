/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * MPI + RCCL micro-benchmark for per-iteration latency with optional
 * RCCL_IB_BLACKOUT_* fault injection. Rank 0 prints summary stats and a
 * text histogram of all samples (all ranks, all timed iterations).
 ************************************************************************/

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <strings.h>
#include <vector>

#include "hip/hip_runtime.h"
#include "rccl/rccl.h"

#define NCCLCHK(cmd)                                                                                                   \
  do {                                                                                                                 \
    ncclResult_t r = (cmd);                                                                                            \
    if (r != ncclSuccess) {                                                                                            \
      fprintf(stderr, "[%d] NCCL failure %s:%d  %s\n", mpiRank, __FILE__, __LINE__, ncclGetErrorString(r));          \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                                  \
    }                                                                                                                  \
  } while (0)

#define HIPCHK(cmd)                                                                                                    \
  do {                                                                                                                 \
    hipError_t e = (cmd);                                                                                              \
    if (e != hipSuccess) {                                                                                             \
      fprintf(stderr, "[%d] HIP failure %s:%d  %s\n", mpiRank, __FILE__, __LINE__, hipGetErrorString(e));            \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                                                    \
    }                                                                                                                  \
  } while (0)

static int mpiRank;
static int mpiSize;

static int localRankFromEnv() {
  const char* keys[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID", "MV2_COMM_WORLD_LOCAL_RANK",
                        "PMI_LOCAL_RANK"};
  for (const char* k : keys) {
    const char* v = getenv(k);
    if (v && v[0]) return atoi(v);
  }
  return 0;
}

enum class CollOp { AllReduce, AllGather, AllToAll };

static void printUsage(const char* prog) {
  fprintf(stderr,
          "Usage: mpirun ... %s [options]\n"
          "  -o <op>        Collective: allreduce | allgather | alltoall (default alltoall)\n"
          "  -n <iters>     Timed iterations per rank when -T is not set (default 2000)\n"
          "  -T <sec>       Run timed iterations until max elapsed wall time (MPI_Wtime) across\n"
          "                 ranks is at least <sec> seconds (stops all ranks together). When set,\n"
          "                 -n is ignored for the timed phase.\n"
          "  --time <sec>   Same as -T\n"
          "  -w <iters>     Warmup iterations (default 50)\n"
          "  -e <count>     Element count (float) per op:\n"
          "                   AllReduce: count on each rank\n"
          "                   AllGather: sendcount per rank (recv = count * nranks)\n"
          "                   AllToAll:  count to each peer (buffers = count * nranks)\n"
          "                 (default 33554432: 128 MiB floats per AllReduce send or AllGather send;\n"
          "                  AllToAll send buffer is count*nranks floats)\n"
          "  -b <rows>      Max printed histogram rows when range is wide (default 20000);\n"
          "                 bins are fixed 0.01 ms on a global grid for cross-run comparison.\n"
          "  --csv          Print one sample per line (ms) on rank 0 after histogram\n"
          "  -h             Help\n"
          "\nExample with blackout:\n"
          "  RCCL_IB_BLACKOUT_ENABLE=1 RCCL_IB_BLACKOUT_MEAN_INTERVAL_MS=200 \\\n"
          "  RCCL_IB_BLACKOUT_DURATION_US=4 RCCL_IB_BLACKOUT_SEED=42 \\\n"
          "  mpirun -np 8 %s -T 120 -e 262144\n",
          prog, prog);
}

static CollOp parseOp(const char* s) {
  if (!strcasecmp(s, "allreduce")) return CollOp::AllReduce;
  if (!strcasecmp(s, "allgather")) return CollOp::AllGather;
  if (!strcasecmp(s, "alltoall")) return CollOp::AllToAll;
  fprintf(stderr, "Unknown op '%s'\n", s);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return CollOp::AllReduce;
}

static void runCollective(CollOp op, float* send, float* recv, size_t count, ncclComm_t comm, hipStream_t stream) {
  switch (op) {
  case CollOp::AllReduce:
    NCCLCHK(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
    break;
  case CollOp::AllGather:
    NCCLCHK(ncclAllGather(send, recv, count, ncclFloat, comm, stream));
    break;
  case CollOp::AllToAll:
    NCCLCHK(ncclAllToAll(send, recv, count, ncclFloat, comm, stream));
    break;
  }
}

static size_t bufferElements(CollOp op, size_t count) {
  if (op == CollOp::AllReduce) return count;
  return count * (size_t)mpiSize;
}

static void printHistogram(const std::vector<double>& ms, int maxHistRows, bool csvDump) {
  if (ms.empty()) {
    printf("No samples.\n");
    return;
  }
  constexpr double kBinWidthMs = 0.01;
  const size_t n = ms.size();
  double mn = *std::min_element(ms.begin(), ms.end());
  double mx = *std::max_element(ms.begin(), ms.end());
  std::vector<double> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double p) {
    if (n == 0) return 0.0;
    if (n == 1) return sorted[0];
    double x = (p / 100.0) * (double)(n - 1);
    size_t idx = (size_t)(x + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
  };
  double mean = 0;
  for (double t : ms) mean += t;
  mean /= (double)n;
  double var = 0;
  for (double t : ms) {
    double d = t - mean;
    var += d * d;
  }
  var /= (double)std::max<size_t>(n, 1);
  double sd = std::sqrt(var);

  printf("\n--- Timing (ms), N=%zu samples (all ranks) ---\n", n);
  printf("min=%.6g  max=%.6g  mean=%.6g  stddev=%.6g\n", mn, mx, mean, sd);
  printf("p50=%.6g  p90=%.6g  p95=%.6g  p99=%.6g  p99.9=%.6g\n", pct(50.0), pct(90.0), pct(95.0), pct(99.0),
         pct(99.9));

  if (maxHistRows < 8) maxHistRows = 8;

  /* Global 0.01 ms grid: bin index k = floor(t / 0.01) covers [k*0.01, (k+1)*0.01) ms. */
  const long long kMin = (long long)std::floor(mn / kBinWidthMs);
  const long long kMax = (long long)std::floor(mx / kBinWidthMs);
  const unsigned long long span = (unsigned long long)(kMax - kMin) + 1ULL;

  std::map<long long, int> sparse;
  for (double t : ms) {
    long long k = (long long)std::floor(t / kBinWidthMs);
    sparse[k]++;
  }

  printf("\n--- Histogram: fixed bin width %.2f ms (global grid) ---\n", kBinWidthMs);
  /* 12+1+12+1+12+1+7+2 chars before '*' column (pct = %% of all samples in bin). */
  printf("%12s %12s %12s %7s |\n", "bin_lo_ms", "bin_hi_ms", "count", "pct");

  const unsigned long long kDenseMax = 500000ULL;

  if (span <= kDenseMax) {
    std::vector<int> hist((size_t)span, 0);
    for (const auto& kv : sparse) {
      if (kv.first < kMin || kv.first > kMax) continue;
      hist[(size_t)(kv.first - kMin)] = kv.second;
    }
    int maxCount = 1;
    for (size_t i = 0; i < hist.size(); i++) maxCount = std::max(maxCount, hist[i]);

    const int barW = 50;
    if ((long long)span > maxHistRows) {
      fprintf(stderr,
              "[mpi_blackout_hist] %llu bins exceed -b=%d; printing contiguous bins with chunk markers (0.01 ms grid).\n",
              (unsigned long long)span, maxHistRows);
    }
    for (long long k = kMin; k <= kMax; k++) {
      int c = hist[(size_t)(k - kMin)];
      double lo = (double)k * kBinWidthMs;
      double hi = (double)(k + 1) * kBinWidthMs;
      double pct = 100.0 * (double)c / (double)n;
      int len = (int)std::llround((double)c / (double)maxCount * (double)barW);
      printf("%12.2f %12.2f %12d %7.2f |", lo, hi, c, pct);
      for (int j = 0; j < len; j++) putchar('*');
      putchar('\n');
      if ((long long)span > maxHistRows && (k - kMin + 1) % maxHistRows == 0 && k < kMax)
        printf("%-48s|\n", "--- chunk break ---");
    }
  } else {
    fprintf(stderr,
            "[mpi_blackout_hist] Span %llu bins (> %llu); printing **non-zero** bins only (same 0.01 ms grid).\n",
            (unsigned long long)span, kDenseMax);
    int maxCount = 1;
    for (const auto& kv : sparse) maxCount = std::max(maxCount, kv.second);
    const int barW = 50;
    int row = 0;
    for (const auto& kv : sparse) {
      long long k = kv.first;
      int c = kv.second;
      double lo = (double)k * kBinWidthMs;
      double hi = (double)(k + 1) * kBinWidthMs;
      double pct = 100.0 * (double)c / (double)n;
      int len = (int)std::llround((double)c / (double)maxCount * (double)barW);
      printf("%12.2f %12.2f %12d %7.2f |", lo, hi, c, pct);
      for (int j = 0; j < len; j++) putchar('*');
      putchar('\n');
      if (++row >= maxHistRows) {
        fprintf(stderr, "[mpi_blackout_hist] Truncated non-zero bin listing at -b=%d rows.\n", maxHistRows);
        break;
      }
    }
  }

  if (csvDump) {
    printf("\n--- CSV (ms) ---\n");
    for (double t : ms) printf("%.9g\n", t);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  CollOp coll = CollOp::AllToAll;
  int timedIters = 2000;
  double minTimedSec = 0.0;
  int warmup = 50;
  /* Default: 128 MiB as float32 (128*1024*1024 / 4) for AllReduce send; AllGather/AllToAll use same -e semantics. */
  size_t count = (size_t)128 * 1024 * 1024 / sizeof(float);
  int histBins = 20000;
  bool csvDump = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
      printUsage(argv[0]);
      MPI_Finalize();
      return 0;
    } else if (!strcmp(argv[i], "-o") && i + 1 < argc) {
      coll = parseOp(argv[++i]);
    } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
      timedIters = atoi(argv[++i]);
    } else if ((!strcmp(argv[i], "-T") || !strcmp(argv[i], "--time")) && i + 1 < argc) {
      minTimedSec = strtod(argv[++i], nullptr);
    } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
      count = (size_t)strtoull(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "-b") && i + 1 < argc) {
      histBins = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--csv")) {
      csvDump = true;
    } else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      printUsage(argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  if (warmup < 0 || count < 1) {
    fprintf(stderr, "Invalid -w or -e\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const bool useTimeLimit = (minTimedSec > 0.0);
  if (!useTimeLimit && timedIters < 1) {
    fprintf(stderr, "Invalid -n (need >= 1 when -T is not set)\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (useTimeLimit && !std::isfinite(minTimedSec)) {
    fprintf(stderr, "Invalid -T / --time (need finite positive seconds)\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int nDev = 0;
  HIPCHK(hipGetDeviceCount(&nDev));
  if (nDev <= 0) {
    fprintf(stderr, "No HIP devices.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int lr = localRankFromEnv();
  if (lr < 0 || lr >= nDev) lr = mpiRank % nDev;
  HIPCHK(hipSetDevice(lr));

  ncclUniqueId id;
  if (mpiRank == 0) NCCLCHK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHK(ncclCommInitRank(&comm, mpiSize, id, mpiRank));

  size_t nelem = bufferElements(coll, count);
  float *send = nullptr, *recv = nullptr;
  HIPCHK(hipMalloc(&send, nelem * sizeof(float)));
  HIPCHK(hipMalloc(&recv, nelem * sizeof(float)));
  HIPCHK(hipMemset(send, 1, nelem * sizeof(float)));
  HIPCHK(hipMemset(recv, 0, nelem * sizeof(float)));

  hipStream_t stream;
  HIPCHK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  using clock = std::chrono::steady_clock;
  for (int i = 0; i < warmup; i++) {
    runCollective(coll, send, recv, count, comm, stream);
  }
  HIPCHK(hipStreamSynchronize(stream));

  std::vector<double> localMs;
  localMs.reserve(useTimeLimit ? 65536 : (size_t)timedIters);

  if (useTimeLimit) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double tPhase0 = MPI_Wtime();
    while (true) {
      HIPCHK(hipStreamSynchronize(stream));
      auto t0 = clock::now();
      runCollective(coll, send, recv, count, comm, stream);
      HIPCHK(hipStreamSynchronize(stream));
      auto t1 = clock::now();
      std::chrono::duration<double, std::milli> dt = t1 - t0;
      localMs.push_back(dt.count());

      const double localElapsed = MPI_Wtime() - tPhase0;
      double maxElapsed = 0.0;
      MPI_Allreduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (maxElapsed >= minTimedSec) break;
    }
  } else {
    for (int i = 0; i < timedIters; i++) {
      HIPCHK(hipStreamSynchronize(stream));
      auto t0 = clock::now();
      runCollective(coll, send, recv, count, comm, stream);
      HIPCHK(hipStreamSynchronize(stream));
      auto t1 = clock::now();
      std::chrono::duration<double, std::milli> dt = t1 - t0;
      localMs.push_back(dt.count());
    }
  }

  HIPCHK(hipFree(send));
  HIPCHK(hipFree(recv));
  HIPCHK(hipStreamDestroy(stream));
  NCCLCHK(ncclCommDestroy(comm));

  int myN = (int)localMs.size();
  std::vector<int> recvCounts;
  if (mpiRank == 0) recvCounts.resize((size_t)mpiSize);
  MPI_Gather(&myN, 1, MPI_INT, mpiRank == 0 ? recvCounts.data() : nullptr, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpiRank == 0) {
    std::vector<int> displs((size_t)mpiSize);
    int total = 0;
    for (int r = 0; r < mpiSize; r++) {
      displs[(size_t)r] = total;
      total += recvCounts[(size_t)r];
    }
    std::vector<double> allMs((size_t)total);
    MPI_Gatherv(localMs.data(), myN, MPI_DOUBLE, allMs.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0,
                MPI_COMM_WORLD);

    const char* opStr = (coll == CollOp::AllReduce) ? "AllReduce" : (coll == CollOp::AllGather) ? "AllGather" : "AllToAll";
    if (useTimeLimit) {
      printf("mpi_blackout_hist: ranks=%d  op=%s  float_count=%zu  timed_phase>= %.6g s (iters/rank=%d)  warmup=%d  (rank0 local GPU %d)\n",
             mpiSize, opStr, count, minTimedSec, myN, warmup, lr);
    } else {
      printf("mpi_blackout_hist: ranks=%d  op=%s  float_count=%zu  timed_iters/rank=%d  warmup=%d  (rank0 local GPU %d)\n",
             mpiSize, opStr, count, timedIters, warmup, lr);
    }
    printHistogram(allMs, histBins, csvDump);
  } else {
    MPI_Gatherv(localMs.data(), myN, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
