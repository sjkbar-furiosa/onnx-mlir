# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSupport
  Common.cpp
  IndexExpr.cpp
  IndexExprDetail.cpp
  OMOptions.cpp


  LINK_LIBS PUBLIC
  MLIRMath
  MLIRAffine
  MLIRSCF
  )
