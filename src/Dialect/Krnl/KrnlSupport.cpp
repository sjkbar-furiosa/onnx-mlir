/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------- KrnlSupport.cpp - Krnl-level support functions -----------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains support code used at the level of the KRNL dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Krnl/KrnlSupport.hpp"
#include "src/Support/Common.hpp"

//===----------------------------------------------------------------------===//
// Return various operations.
//===----------------------------------------------------------------------===//

/// Get the AllocOp of the current GetRef.
memref::AllocOp getAllocOfGetRef(KrnlGetRefOp *getRef) {
  auto parentBlock = getRef->getOperation()->getBlock();

  memref::AllocOp alloc = nullptr;
  parentBlock->walk([&alloc, getRef](memref::AllocOp op) {
    auto getRefAlloc = getRef->getOperands()[0];
    if (op.getResult() == getRefAlloc)
      alloc = op;
  });

  return alloc;
}

/// Return the top block.
Block *getTopBlock(Operation *op) {
  // Get current block as the first top block candidate.
  Block *topBlock = op->getBlock();
  Operation *parentBlockOp = topBlock->getParentOp();

  while (!llvm::dyn_cast_or_null<FuncOp>(parentBlockOp)) {
    topBlock = parentBlockOp->getBlock();
    parentBlockOp = topBlock->getParentOp();
  }

  return topBlock;
}

//===----------------------------------------------------------------------===//
// Perform checks or get statistics about Krnl-level operations.
//===----------------------------------------------------------------------===//

/// Operation is a LoadOp or AffineLoadOp.
bool isLoad(Operation *op) { return llvm::dyn_cast_or_null<KrnlLoadOp>(op); }

/// Operation is a StoreOp or AffineStoreOp.
bool isStore(Operation *op) { return llvm::dyn_cast_or_null<KrnlStoreOp>(op); }

/// Operation is a KrnlMemcpyOp.
bool isKrnlMemcpy(Operation *op) {
  return llvm::dyn_cast_or_null<KrnlMemcpyOp>(op);
}

/// Checks if this operation loads/stores from the result of a specific getRef.
/// A krnl.memcpy acts as both load and store.
bool isLoadStoreForGetRef(KrnlGetRefOp getRef, Operation *op) {
  auto result = getRef.getResult();

  // Is used by load/store/krnl.memcpy.
  bool isUsedByLoadStore =
      (isLoad(op) && result == op->getOperands()[0]) ||
      (isStore(op) && result == op->getOperands()[1]) ||
      (isKrnlMemcpy(op) &&
          (result == op->getOperands()[0] || result == op->getOperands()[1]));

  // If not used by a load/store or krnl memcpy, then it can be used by
  // another operation. When this happens we assume that the lowering of the
  // operation will involve a load/store.
  if (!isUsedByLoadStore && !isLoad(op) && !isStore(op) && !isKrnlMemcpy(op))
    for (const auto &operand : op->getOperands())
      if (operand == result)
        return true;

  return isUsedByLoadStore;
}

/// Check if this value is an argument of one of the blocks nested around it.
bool isBlockArgument(Operation *op, Value operand) {
  // Parent operation of the current block.
  Operation *parentBlockOp;
  Block *currentBlock = op->getBlock();

  do {
    // Check the arguments of the current block.
    for (auto arg : currentBlock->getArguments())
      if (operand == arg)
        return true;

    parentBlockOp = currentBlock->getParentOp();
    currentBlock = parentBlockOp->getBlock();

  } while (!llvm::dyn_cast_or_null<FuncOp>(parentBlockOp));

  return false;
}

/// Check if two GetRefs participate in the same krnl.memcpy.
bool usedBySameKrnlMemcpy(
    KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef) {
  Block *topBlock = getTopBlock(firstGetRef->getOperation());

  bool sameKrnlMemcpy = false;
  topBlock->walk(
      [&sameKrnlMemcpy, firstGetRef, secondGetRef](KrnlMemcpyOp memcpyOp) {
        if ((memcpyOp.dest() == firstGetRef->getResult() &&
                memcpyOp.src() == secondGetRef->getResult()) ||
            (memcpyOp.dest() == secondGetRef->getResult() &&
                memcpyOp.src() == firstGetRef->getResult()))
          sameKrnlMemcpy = true;
      });

  return sameKrnlMemcpy;
}

/// Check if two GetRefs participate in the same operation.
bool usedBySameOp(KrnlGetRefOp *firstGetRef, KrnlGetRefOp *secondGetRef) {
  Block *topBlock = getTopBlock(firstGetRef->getOperation());

  bool sameOp = false;
  topBlock->walk([&sameOp, firstGetRef, secondGetRef](Operation *op) {
    bool firstUsed = false;
    for (const auto &operand : op->getOperands())
      if (operand == firstGetRef->getResult())
        firstUsed = true;

    if (firstUsed)
      for (const auto &operand : op->getOperands())
        if (operand == secondGetRef->getResult())
          sameOp = true;
  });
  return sameOp;
}

/// Get the number of GetRef ops associated with this AllocOp.
int64_t getAllocGetRefNum(memref::AllocOp *allocOp) {
  auto parentBlock = allocOp->getOperation()->getBlock();

  int64_t numGetRefs = 0;
  parentBlock->walk([&numGetRefs, allocOp](KrnlGetRefOp op) {
    auto result = allocOp->getResult();
    if (op.getOperands()[0] == result)
      numGetRefs++;
  });

  return numGetRefs;
}

/// Check if an operation is in the top-level block of the function.
bool opInTopLevelBlock(Operation *op) {
  Block *currentBlock = op->getBlock();

  // If the parent operation of the current block is a FuncOp then
  // this operation is in the top-level block.
  return llvm::dyn_cast_or_null<FuncOp>(currentBlock->getParentOp());
}

/// This function returns true if `beforeOp` is visited before `op` in a
/// traversal of the provided block.
bool opBeforeOp(Block *block, Operation *beforeOp, Operation *afterOp) {
  bool beforeOpIsBefore = true;
  bool beforeOpFound = false;
  block->walk(
      [&beforeOpIsBefore, &beforeOpFound, beforeOp, afterOp](Operation *op) {
        if (op == beforeOp)
          beforeOpFound = true;
        else if (op == afterOp && !beforeOpFound)
          beforeOpIsBefore = false;
      });
  return beforeOpIsBefore;
}

/// Check Alloc operation result is used by a krnl.getref.
bool checkOpResultIsUsedByGetRef(memref::AllocOp *allocOp) {
  FuncOp function = getContainingFunction(allocOp->getOperation());

  bool opIsUsedInGetRef = false;
  function.walk([&opIsUsedInGetRef, allocOp](KrnlGetRefOp op) {
    auto result = allocOp->getResult();
    for (const auto &operand : op.getOperands())
      if (operand == result)
        opIsUsedInGetRef = true;
  });

  return opIsUsedInGetRef;
}

//===----------------------------------------------------------------------===//
// Live range analysis support.
//===----------------------------------------------------------------------===//

/// Returns the first operation in the live range of a getRef.
Operation *getLiveRangeFirstOp(KrnlGetRefOp getRef) {
  Block *topBlock = getTopBlock(getRef.getOperation());

  Operation *firstLoadStore = nullptr;
  topBlock->walk([&firstLoadStore, getRef](Operation *op) {
    // If op is a Laod/Store, of any kind then assign it to lastLoadStore.
    if (!firstLoadStore && isLoadStoreForGetRef(getRef, op))
      firstLoadStore = op;
  });

  return firstLoadStore;
}

/// Returns the last operation in the live range of a getRef.
Operation *getLiveRangeLastOp(KrnlGetRefOp getRef) {
  Block *topBlock = getTopBlock(getRef.getOperation());

  Operation *lastLoadStore = nullptr;
  topBlock->walk([&lastLoadStore, getRef](Operation *op) {
    // If op is a Laod/Store, of any kind then assign it to lastLoadStore.
    if (isLoadStoreForGetRef(getRef, op))
      lastLoadStore = op;
  });

  return lastLoadStore;
}

/// Check if an operation is in an existing live range.
bool operationInLiveRange(
    Operation *operation, std::vector<Operation *> liveRangeOpList) {
  for (auto &op : liveRangeOpList) {
    if (op == operation)
      return true;
  }
  return false;
}

/// Function that returns the live range of a GetRef operation. The live
/// range consists of all the operations in the in-order traversal of the
/// source code between the first load/store instruction from that GetRef
/// and the last load/store instruction from that GetRef.
std::vector<Operation *> getLiveRange(KrnlGetRefOp getRef) {
  std::vector<Operation *> operations;

  auto topBlock = getTopBlock(getRef.getOperation());

  // Determine last load/store from getRef.
  Operation *lastLoadStore = getLiveRangeLastOp(getRef);

  bool operationInLiveRange = false;
  topBlock->walk([&operations, &operationInLiveRange, lastLoadStore, getRef](
                     Operation *op) {
    // If op is a Laod/Store, of any kind, then assign it to lastLoadStore.
    if (isLoadStoreForGetRef(getRef, op) && !operationInLiveRange)
      operationInLiveRange = true;

    if (operationInLiveRange)
      operations.emplace_back(op);

    if (op == lastLoadStore)
      operationInLiveRange = false;
  });

  return operations;
}

/// The live range is contained between firstOp and lastOp.
bool liveRangeIsContained(Operation *firstOp, Operation *lastOp,
    std::vector<Operation *> liveRangeOpList) {
  Operation *liveRangeFirstOp = liveRangeOpList[0];
  assert(liveRangeOpList.size() > 0 &&
         "Live range empty but must have at least one element.");
  Operation *liveRangeLastOp = liveRangeOpList[liveRangeOpList.size() - 1];

  Block *topLevelBlock = getTopBlock(firstOp);

  return opBeforeOp(topLevelBlock, firstOp, liveRangeFirstOp) &&
         opBeforeOp(topLevelBlock, liveRangeLastOp, lastOp);
}
