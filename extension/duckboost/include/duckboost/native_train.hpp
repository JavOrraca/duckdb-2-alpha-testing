//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/native_train.hpp
//
// Optional native XGBoost / LightGBM / CatBoost trainers (DUCKBOOST_WITH_*).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckboost/model.hpp"

namespace duckdb {
namespace duckboost {

//! True when this build compiled with DUCKBOOST_WITH_<BACKEND> (stub or linked).
bool NativeTrainerCompiled(BoostBackend backend);

//! Train using a vendor library when linked; stub builds throw a clear NotImplementedException.
BoostModel TrainNative(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options);

} // namespace duckboost
} // namespace duckdb
