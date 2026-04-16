/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Optional per-NIC fault injection for internal IB transports.
 ************************************************************************/

#ifndef RCCL_IB_BLACKOUT_H_
#define RCCL_IB_BLACKOUT_H_

/* Must match MAX_IB_DEVS in transport/net_ib*.cc */
#define NCCL_IB_BLACKOUT_MAX_DEVS 32

enum {
  NCCL_IB_BLACKOUT_LANE_IB = 0,
  NCCL_IB_BLACKOUT_LANE_CAST = 1,
  NCCL_IB_BLACKOUT_NLANES = 2
};

void ncclIbBlackoutLaneDevInit(int lane, int ibDevIndex);
void ncclIbBlackoutLaneStop(int lane);
void ncclIbBlackoutLaneWait(int lane, int ibDevN);

#endif
