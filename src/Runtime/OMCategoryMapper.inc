/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- OMCategoryMapper.inc - OMCategoryMapper C/C++ Implementation ----===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of the OMCategoryMapper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus

#include <iostream>
#include <map>
#include <string>

using namespace std;

template <typename T1, typename T2>
static T2 lookup(
    T1 array1[], T2 array2[], int64_t nElems, T1 Datum, T2 Default) {
  //  assert(array1.size() == nElems && array2.size() == nElems);
  static map<T1, T2> Map;

  if (Map.empty()) {
    cout << "ETTORE" << endl;
    for (int64_t i = 0; i < nElems; ++i)
      Map[array1[i]] = array2[i];
  }

  auto it = Map.find(Datum);
  return (it != Map.end()) ? it->second : Default;
}

extern "C" void lookup_string(const char *array1[], int64_t array2[],
    const char *Datum, int64_t nElems, int64_t Default, int64_t &Result) {
  string *strings = new string[nElems];
  for (int i = 0; i < nElems; ++i)
    strings[i] = array1[i];

  Result = lookup<string, int64_t>(strings, array2, nElems, Datum, Default);
}

extern "C" void lookup_int(int64_t array1[], const char *array2[],
    int64_t Datum, int64_t nElems, const char *Default, char *&Result) {
  string *strings = new string[nElems];
  for (int i = 0; i < nElems; ++i)
    strings[i] = array2[i];

  string res = lookup<int64_t, string>(array1, strings, nElems, Datum, Default);
  strcpy(Result, res.c_str());
}

#endif // __cplusplus