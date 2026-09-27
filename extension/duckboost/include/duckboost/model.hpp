//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckboost/model.hpp
//
// Unified tree-ensemble model format shared by all duckboost backends.
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {
namespace duckboost {

enum class BoostBackend : uint8_t { REFERENCE = 0, XGBOOST = 1, LIGHTGBM = 2, CATBOOST = 3 };

enum class BoostTask : uint8_t { REGRESSION = 0, BINARY = 1 };

struct TreeNode {
	idx_t feature = 0;
	double threshold = 0;
	idx_t left = 0;
	idx_t right = 0;
	double value = 0;
	bool is_leaf = true;
};

struct BoostTree {
	vector<TreeNode> nodes;
};

struct BoostModel {
	idx_t duckboost_version = 1;
	BoostBackend backend = BoostBackend::REFERENCE;
	BoostTask task = BoostTask::REGRESSION;
	double base_score = 0;
	double learning_rate = 0.1;
	idx_t n_features = 0;
	vector<string> feature_names;
	vector<BoostTree> trees;

	string ToJSON() const;
	static BoostModel FromJSON(const string &json);

	double PredictRaw(const vector<double> &features) const;
	double Predict(const vector<double> &features) const;
};

struct TrainOptions {
	BoostBackend backend = BoostBackend::REFERENCE;
	BoostTask task = BoostTask::REGRESSION;
	idx_t n_estimators = 10;
	idx_t max_depth = 3;
	double learning_rate = 0.1;
	idx_t min_samples_leaf = 1;
	idx_t max_bins = 16;
	vector<string> feature_names;

	static TrainOptions FromMap(const unordered_map<string, string> &options);
};

struct EvalOptions {
	string metric = "auto"; // auto | rmse | mae | accuracy | logloss
	static EvalOptions FromMap(const unordered_map<string, string> &options);
};

struct SqlExportOptions {
	bool separate_trees = true;
	string prediction_alias = "prediction";
	static SqlExportOptions FromMap(const unordered_map<string, string> &options);
};

string BackendToString(BoostBackend backend);
BoostBackend BackendFromString(const string &name);
string TaskToString(BoostTask task);
BoostTask TaskFromString(const string &name);

bool BackendTrainingSupported(BoostBackend backend);
string BackendCapabilityNote(BoostBackend backend);

BoostModel TrainModel(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options);
double EvaluateModel(const BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const EvalOptions &options);
string ExportModelSQL(const BoostModel &model, const string &table_name, const vector<string> &feature_columns,
                      const SqlExportOptions &options);

} // namespace duckboost
} // namespace duckdb
