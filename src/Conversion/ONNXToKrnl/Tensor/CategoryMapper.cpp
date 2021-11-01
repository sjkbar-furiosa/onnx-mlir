/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ CategoryMapper.cpp - Lowering CategoryMapper Op ---------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CategoryMapper Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

using namespace mlir;
using namespace std;

struct ONNXCategoryMapperOpLowering : public ConversionPattern {
  ONNXCategoryMapperOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            mlir::ONNXCategoryMapperOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    MemRefType outputMemRefType = convertToMemRefType(*op->result_type_begin());
    ArrayRef<int64_t> outputMemRefShape = outputMemRefType.getShape();
    uint64_t outputRank = outputMemRefShape.size();
    Type elementType = outputMemRefType.getElementType();

    // Insert alloc/dealloc pair for output tensor.
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc =
        insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);

    rewriter.create<KrnlCategoryMapperOp>(
        loc, alloc, numberOfRandomValues, meanValue, scaleValue, seedValue);

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

// Perform a 32 bit FNV hash on the given string
// (http://isthe.com/chongo/tech/comp/fnv).
uint32_t hash(uint32_t hval, const string &str) {
  constexpr uint32_t prime = 0x01000193;
  hval = (hval == 0) ? prime : hval;

  for (const char c : str) {
    hval *= prime;
    hval ^= c;
  }

  return hval;
}

// Extracts the keys of the given map.
template <typename KeyType, typename ValueType>
vector<KeyType> extractKeys(const map<KeyType, ValueType> &map) {
  vector<KeyType> keys;
  for (const auto &entry : map)
    keys.push_back(entry.first);

  return keys;
}

template <typename T>
void print(const vector<T> &V, const string &name) {
  cout << name << ": [ ";
  for (auto elem : V)
    cout << elem << ", ";
  cout << "]\n\n";
}

template <typename T>
void print(const vector<vector<T>> &V, const string &name) {
  cout << name << ": [";
  for (const vector<string> &v : V) {
    cout << "[ ";
    for (const string &str : v)
      cout << "'" << str << "' ";
    cout << "] ";
  }
  cout << "]\n\n";
}

template <typename K, typename V>
void print(const map<K, V> &M, const string &name) {
  cout << name << " : {";
  for (auto &entry : M)
    cout << "'" << entry.first << "': " << entry.second << ", ";
  cout << "}\n\n";
}

// Generate the integers in the range [0 .. max-1].
vector<uint32_t> range(uint32_t max) {
  //  assert(max > 0);
  vector<uint32_t> range(max);
  iota(range.begin(), range.end(), 0);
  return range;
}

// Generate the integers in the range [min .. max-1].
vector<uint32_t> range(uint32_t min, uint32_t max) {
  //  assert(min < max && max > 0);
  vector<uint32_t> range(max - min);
  iota(range.begin(), range.end(), min);
  return range;
}

vector<uint32_t> range(int32_t min, int32_t max, int32_t step) {
  //  assert(min < max && max > 0);
  vector<uint32_t> range;

  int32_t nElems = (max - min) / step;
  if (nElems < 1)
    return range;

  range.resize(nElems);
  int32_t num = min;
  generate_n(range.begin(), nElems, [&num, step]() {
    int32_t res = num;
    num += step;
    return res;
  });

  // print(range, "range");

  return range;
}

tuple<vector<int64_t>, vector<int64_t>> createMinimalPerfectHash(
    const map<string, int64_t> &dict) {
  size_t size = dict.size();

  // Step 1: place all of the keys into buckets.
  vector<int64_t> G(size, 0);
  vector<int64_t> V(size, -1);
  vector<string> keys = extractKeys<string, int64_t>(dict);

  vector<vector<string>> buckets(size);
  for (const string &key : keys)
    buckets[::hash(0, key) % size].push_back(key);

  // Step 2: Sort the buckets and process the ones with the most items
  // first.
  sort(buckets.begin(), buckets.end(),
      [](const vector<string> &v1, const vector<string> &v2) {
        return v1.size() > v2.size();
      });

  print(buckets, "buckets");

  uint32_t biMax = 0;
  for (uint32_t bi : range(size)) {
    biMax = bi;
    vector<string> &bucket = buckets[bi];
    if (bucket.size() <= 1)
      break;

    int32_t hval = 1;
    int32_t item = 0;
    vector<uint32_t> slots;

    // Repeatedly try different hash values until we find a hash function.
    // that places all items in the bucket into free slots.
    while (item < bucket.size()) {
      uint32_t slot = ::hash(hval, bucket[item]) % size;
      if (V[slot] != -1 ||
          find(slots.begin(), slots.end(), slot) != slots.end()) {
        hval++;
        item = 0;
        slots.clear();
      } else {
        slots.push_back(slot);
        item++;
      }
    }

    G[::hash(0, bucket[0]) % size] = hval;
    for (uint32_t i : range(bucket.size()))
      V[slots[i]] = dict.at(bucket[i]);
  }

  //  print(V, "V");

  // Place remaining buckets (containing a single entry) into a free slot.
  // Use a negative value of hval to indicate this.
  vector<uint32_t> freeList;
  for (uint32_t i : range(size))
    if (V[i] == -1)
      freeList.push_back(i);

  // print(freeList, "freeList");

  for (uint32_t i : range(biMax, size)) {
    vector<string> &bucket = buckets[i];
    if (bucket.size() == 0)
      break;

    uint32_t slot = freeList.back();
    freeList.pop_back();

    // Subtract one to ensure it's negative even if the zeroeth slot was
    // used.
    G[::hash(0, bucket[0]) % size] = -(int32_t)slot - 1;
    V[slot] = dict.at(bucket[0]);
  }

  return {G, V};
}

// Look up a value in the hash table, defined by G and V.
int64_t perfectHashLookup(
    const vector<int64_t> &G, const vector<int64_t> &V, const string &key) {
  int64_t d = G[::hash(0, key) % G.size()];
  if (d < 0)
    return V.at(-d - 1);
  return V.at(::hash(d, key) % V.size());
}

// This class represents a node in the directed acyclic word graph (DAWG). It
// has a list of edges to other nodes. It has functions for testing whether it
// is equivalent to another node. Nodes are equivalent if they have identical
// edges, and each identical edge leads to identical states. The __hash__ and
// __eq__ functions allow it to be used as a key in a python dictionary.
class DawgNode {
public:
  static uint32_t NextId;
  uint32_t id;
  map<char, DawgNode *> edges;
  bool final = false;

  DawgNode() : id(DawgNode::NextId++) {}

  string getName() const {
    vector<string> arr;
    if (final)
      arr.emplace_back("1");
    else
      arr.emplace_back("0");
    for (const auto &entry : edges) {
      arr.emplace_back(string(1, entry.first));
      arr.push_back(to_string(entry.second->id));
    }

    string str;
    for (const string &s : arr)
      str += s + "_";

    return str;
  }

  size_t getHash() const { return std::hash<string>{}(getName()); }

  bool operator==(const DawgNode &other) const {
    return getName() == other.getName();
  }
  bool operator<(const DawgNode &other) const {
    return getName() < other.getName();
  }
};

class Dawg {
public:
  string previousWord;
  DawgNode root;
  using EntryType = tuple<DawgNode *, char, DawgNode *>;
  vector<EntryType> uncheckedNodes;
  map<DawgNode *, DawgNode *> minimizedNodes;

  Dawg() : previousWord(""), root(DawgNode()) {}

  void insert(const string &word) {
    if (word.compare(previousWord) < 0)
      assert(false); // TODO emit error

    //    cout << "at line " << __LINE__ << endl;
    // find common prefix between word and previous word
    uint32_t commonPrefix = 0;
    for (uint32_t i : range(min(word.size(), previousWord.size()))) {
      if (word[i] != previousWord[i])
        break;
      commonPrefix++;
    }
    //  cout << "at line " << __LINE__ << endl;

    // Check the uncheckedNodes for redundant nodes, proceeding from last one
    // down to the common prefix size. Then truncate the list at that point.
    minimize(commonPrefix);

    // cout << "at line " << __LINE__ << endl;

    // add the suffix, starting from the correct node mid-way through the graph.
    DawgNode *node = nullptr;
    if (uncheckedNodes.size() == 0)
      node = &root;
    else {
      uint32_t last = uncheckedNodes.size() - 1;
      EntryType &entry = uncheckedNodes.at(last);
      node = get<2>(entry);
    }

    // cout << "at line " << __LINE__ << endl;
    for (char letter : word.substr(commonPrefix)) {
      auto *nextNode = new DawgNode();
      node->edges[letter] = nextNode;
      uncheckedNodes.emplace_back(make_tuple(node, letter, nextNode));
      node = nextNode;
    }
    // cout << "at line " << __LINE__ << endl;
    node->final = true;
    previousWord = word;
  }

  void finish() { minimize(0); }

  void minimize(int32_t downTo) {
    // proceed from the leaf up to a certain point.

    for (uint32_t i :
        range((int32_t)(uncheckedNodes.size() - 1), downTo - 1, -1)) {
      DawgNode *parent, *child;
      char letter;
      tie(parent, letter, child) = uncheckedNodes[i];

      if (minimizedNodes.find(child) != minimizedNodes.end())
        parent->edges[letter] = minimizedNodes[child];
      else
        minimizedNodes[child] = child;
      uncheckedNodes.pop_back();
    }
  }

  bool lookup(const string &word) const {
    const DawgNode *node = &root;
    for (char letter : word) {
      if (node->edges.find(letter) == node->edges.end())
        return false;
      node = node->edges.at(letter);
    }
    return node->final;
  }

  uint32_t nodeCount() const { return minimizedNodes.size(); }

  uint32_t edgeCount() const {
    uint32_t count = 0;
    for (const DawgNode *node : extractKeys(minimizedNodes))
      count += node->edges.size();
    return count;
  }
};

void populateLoweringONNXCategoryMapperOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {

  patterns.insert<ONNXCategoryMapperOpLowering>(ctx);
}