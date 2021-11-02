/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Common.hpp - Common Utilities -----------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common utilities and support code.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;

#if defined(__GNUC__) || defined(__clang__)
#define ATTRIBUTE(x) __attribute__((x))
#else
#define ATTRIBUTE(x)
#endif

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
Value emitConstantOp(
    OpBuilder &rewriter, Location loc, Type type, double value);

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(Operation *op);

/// This function returns the index in the list of alloc arguments of the
/// dynamic dimension corresponding to `index` in the MemRef shape.
/// As an example:
///
/// alloc(%d0, %d1, %d2) : memref<10x?x?x20x?x30xf32>
///
/// In the above alloc the list of alloc arguments is being represented by
/// %d0, %d1 and %d2. Their indices 0, 1, 2 correspond to `index` values
/// 1, 2 and 4 in the MemRef shape respectively
int64_t getAllocArgIndex(memref::AllocOp allocOp, int64_t index);

/// Get AllocOp alignment if it exists otherwise return zero.
int64_t getAllocAlignment(memref::AllocOp allocOp);

/// Check is all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType memRefType);

/// Check Alloc operation result is used by a krnl.getref.
bool checkOpResultIsUsedByGetRef(memref::AllocOp *allocOp);

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(MemRefType memRefType);

/// Get the size of a MemRef in bytes.
int64_t getMemRefSizeInBytes(Value value);

/// Get the size of a MemRef in bytes.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
Value getDynamicMemRefSizeInBytes(
    PatternRewriter &rewriter, Location loc, Value val);

/// Get the size of a dynamic MemRef in bytes.
Value getDynamicMemRefSizeInBytes(MemRefType type, Location loc,
    PatternRewriter &rewriter, memref::AllocOp allocOp);
