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
