/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.hpp - Krnl-level support functions -----------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

/// Get the AllocOp of the current GetRef.
memref::AllocOp getAllocOfGetRef(KrnlGetRefOp *getRef);

/// Return the top block.
Block *getTopBlock(Operation *op);

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(Operation *op);

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(Operation *op);

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(Operation *op);

/// Checks if this operation loads/stores from the result of a specific getRef.
/// A krnl.memcpy acts as both load and store.
bool isLoadStoreForGetRef(KrnlGetRefOp getRef, Operation *op);

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(Operation *op, Value operand);

/// Check if two GetRefs participate in the same krnl.memcpy.
bool usedBySameKrnlMemcpy(
    KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef);

/// Check if two GetRefs participate in the same operation.
bool usedBySameOp(KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef);

/// Get the number of GetRef ops associated with this AllocOp.
int64_t getAllocGetRefNum(memref::AllocOp *allocOp);

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(Operation *op);

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(Block *block, Operation *beforeOp, Operation *afterOp);

/// Check Alloc operation result is used by a krnl.getref.
bool checkOpResultIsUsedByGetRef(memref::AllocOp *allocOp);

//===----------------------------------------------------------------------===//
// Live range analysis support.
//===----------------------------------------------------------------------===//

/// Returns the first operation in the live range of a getRef.
Operation *getLiveRangeFirstOp(KrnlGetRefOp getRef);

/// Returns the last operation in the live range of a getRef.
Operation *getLiveRangeLastOp(KrnlGetRefOp getRef);

/// Check if an operation is in an existing live range.
bool operationInLiveRange(
    Operation *operation, std::vector<Operation *> liveRangeOpList);

/// Function that returns the live range of a GetRef operation. The live
/// range consists of all the operations in the in-order traversal of the
/// source code between the first load/store instruction from that GetRef
/// and the last load/store instruction from that GetRef.
std::vector<Operation *> getLiveRange(KrnlGetRefOp getRef);

/// The live range is contained between firstOp and lastOp.
bool liveRangeIsContained(Operation *firstOp, Operation *lastOp,
    std::vector<Operation *> liveRangeOpList);
