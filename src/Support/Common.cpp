/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Common.cpp - Common Utilities -----------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common utilities and support code.
//
//===----------------------------------------------------------------------===//

#include "src/Support/Common.hpp"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/TypeSwitch.h"

/// Emit constant operation.
Value emitConstantOp(
    OpBuilder &rewriter, Location loc, Type type, double value) {
  Attribute constantAttr;

  TypeSwitch<Type>(type)
      .Case<Float16Type>(
          [&](Type) { constantAttr = rewriter.getF16FloatAttr((float)value); })
      .Case<Float32Type>(
          [&](Type) { constantAttr = rewriter.getF32FloatAttr((float)value); })
      .Case<Float64Type>(
          [&](Type) { constantAttr = rewriter.getF64FloatAttr((float)value); })
      .Case<IntegerType>([&](Type) {
        auto width = type.cast<IntegerType>().getWidth();
        if (width == 1) {
          constantAttr = rewriter.getBoolAttr(value != 0);
        } else {
          constantAttr =
              rewriter.getIntegerAttr(type, APInt(width, (int64_t)value));
        }
      })
      .Case<IndexType>([&](Type) {
        constantAttr = rewriter.getIntegerAttr(type, (int64_t)value);
      })
      .Default([](Type) { llvm_unreachable("unsupported element type"); });
  return rewriter.create<arith::ConstantOp>(loc, constantAttr);
}

/// Retrieve function which contains the current operation.
FuncOp getContainingFunction(Operation *op) {
  Operation *parentFuncOp = op->getParentOp();

  // While parent is not a FuncOp and its cast to a FuncOp is null.
  while (!llvm::dyn_cast_or_null<FuncOp>(parentFuncOp))
    parentFuncOp = parentFuncOp->getParentOp();

  return cast<FuncOp>(parentFuncOp);
}

/// Get the order number of the dynamic index passed as input.
/// Example for the following shape:
///   <1x2x?x3x?x4xf32>
///
/// getAllocArgIndex(<1x2x?x3x?x4xf32>, 2) will return 0.
/// getAllocArgIndex(<1x2x?x3x?x4xf32>, 4) will return 1.
///
int64_t getAllocArgIndex(memref::AllocOp allocOp, int64_t index) {
  auto memRefShape =
      allocOp.getResult().getType().dyn_cast<MemRefType>().getShape();
  auto rank = memRefShape.size();

  int dynDimIdx = 0;
  for (unsigned int idx = 0; idx < rank; ++idx) {
    if (memRefShape[idx] < 0) {
      if (idx == index)
        return dynDimIdx;
      dynDimIdx++;
    }
  }

  return -1;
}

/// Get alignment of an AllocOp if it exists else return zero.
int64_t getAllocAlignment(memref::AllocOp allocOp) {
  if (IntegerAttr alignmentAttr = allocOp.alignmentAttr())
    return alignmentAttr.getInt();

  return 0;
}

/// Check if all dimensions are known at compile time.
bool hasAllConstantDimensions(MemRefType memRefType) {
  auto memRefShape = memRefType.getShape();
  for (unsigned int i = 0; i < memRefShape.size(); ++i)
    if (memRefShape[i] < 0)
      return false;
  return true;
}

/// Get the MemRef element size in bytes.
unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

/// Get the size of a static MemRef in bytes.
int64_t getMemRefSizeInBytes(Value value) {
  MemRefType memRefType = value.getType().dyn_cast<MemRefType>();
  auto memRefShape = memRefType.getShape();
  int64_t size = 1;
  for (unsigned int i = 0; i < memRefShape.size(); i++)
    size *= memRefShape[i];
  size *= getMemRefEltSizeInBytes(memRefType);
  return size;
}

/// Get the size of a MemRef in bytes.
/// If all the dimensions are static, emit a constant.
/// Otherwise, emit runtime computations.
Value getDynamicMemRefSizeInBytes(
    PatternRewriter &rewriter, Location loc, Value val) {
  MemRefType memRefType = val.getType().cast<MemRefType>();
  auto shape = memRefType.getShape();
  // Accumulate static dimensions first.
  int64_t staticSizeInBytes = getMemRefEltSizeInBytes(memRefType);
  bool allStaticDimensions = true;
  for (unsigned i = 0; i < shape.size(); i++) {
    if (shape[i] != -1)
      staticSizeInBytes *= shape[i];
    else
      allStaticDimensions = false;
  }
  // Accumulate the remaining dimensions that are unknown.
  Value sizeInBytes =
      emitConstantOp(rewriter, loc, rewriter.getI64Type(), staticSizeInBytes);
  if (!allStaticDimensions) {
    for (unsigned i = 0; i < shape.size(); i++) {
      if (shape[i] == -1) {
        Value index = rewriter.create<memref::DimOp>(loc, val, i);
        Value dim = rewriter.create<arith::IndexCastOp>(
            loc, index, rewriter.getI64Type());
        sizeInBytes = rewriter.create<arith::MulIOp>(loc, sizeInBytes, dim);
      }
    }
  }
  return sizeInBytes;
}

/// Get the size of a dynamic MemRef in bytes.
Value getDynamicMemRefSizeInBytes(MemRefType type, Location loc,
    PatternRewriter &rewriter, memref::AllocOp allocOp) {
  // Initialize the size variable with the size in bytes of the type.
  int64_t typeSize = getMemRefEltSizeInBytes(type);
  Value result =
      emitConstantOp(rewriter, loc, rewriter.getIndexType(), typeSize);

  // Multiply all dimensions (constant and dynamic).
  auto memRefShape = type.getShape();
  auto rank = memRefShape.size();
  int dynDimIdx = 0;
  for (unsigned int idx = 0; idx < rank; ++idx) {
    if (memRefShape[idx] < 0) {
      // Dyanmic size.
      auto dynamicDim = allocOp.getOperands()[dynDimIdx];
      dynDimIdx++;
      result = rewriter.create<arith::MulIOp>(loc, result, dynamicDim);
    } else {
      // Static size.
      auto staticDim = emitConstantOp(
          rewriter, loc, rewriter.getIndexType(), memRefShape[idx]);
      result = rewriter.create<arith::MulIOp>(loc, result, staticDim);
    }
  }

  return result;
}
