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

#if defined(__GNUC__) || defined(__clang__)
#define ATTRIBUTE(x) __attribute__((x))
#else
#define ATTRIBUTE(x)
#endif

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;

// Emit a constant of a specific type.
// Use this function for small values only to avoid unexpected loss in type
// casting.
Value emitConstantOp(OpBuilder &rewriter, Location loc, Type type,
                     double value);
