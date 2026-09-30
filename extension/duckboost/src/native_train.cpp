#include "duckboost/native_train.hpp"
#include "duckboost/import.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)
extern "C" {
typedef void *DMatrixHandle;
typedef void *BoosterHandle;
typedef uint64_t bst_ulong;

const char *XGBGetLastError();
int XGDMatrixCreateFromMat(const float *data, bst_ulong nrow, bst_ulong ncol, float missing, DMatrixHandle *out);
int XGDMatrixSetFloatInfo(DMatrixHandle handle, const char *field, const float *array, bst_ulong len);
int XGDMatrixFree(DMatrixHandle handle);
int XGBoosterCreate(const DMatrixHandle dmats[], bst_ulong len, BoosterHandle *out);
int XGBoosterFree(BoosterHandle handle);
int XGBoosterSetParam(BoosterHandle handle, const char *name, const char *value);
int XGBoosterUpdateOneIter(BoosterHandle handle, int iter, DMatrixHandle dtrain);
int XGBoosterPredict(BoosterHandle handle, DMatrixHandle dmat, int option_mask, unsigned ntree_limit, int training,
                     bst_ulong *out_len, const float **out_result);
int XGBoosterDumpModelEx(BoosterHandle handle, const char *fmap, int with_stats, const char *format, bst_ulong *out_len,
                         const char ***out_dump_array);
int XGBoosterDumpModelExWithFeatures(BoosterHandle handle, int fnum, const char **fname, const char **ftype,
                                     int with_stats, const char *format, bst_ulong *out_len, const char ***out_models);
}
#endif

#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)
extern "C" {
typedef void *DatasetHandle;
typedef void *BoosterHandle;

const char *LGBM_GetLastError();
int LGBM_DatasetCreateFromMat(const void *data, int data_type, int32_t nrow, int32_t ncol, int is_row_major,
                              const char *parameters, const DatasetHandle reference, DatasetHandle *out);
int LGBM_DatasetSetField(DatasetHandle handle, const char *field_name, const void *field_data, int num_element,
                         int type);
int LGBM_DatasetSetFeatureNames(DatasetHandle handle, const char **feature_names, int num_feature_names);
int LGBM_DatasetFree(DatasetHandle handle);
int LGBM_BoosterCreate(const DatasetHandle train_data, const char *parameters, BoosterHandle *out);
int LGBM_BoosterAddValidData(BoosterHandle handle, const DatasetHandle valid_data);
int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int *is_finished);
int LGBM_BoosterGetEval(BoosterHandle handle, int data_idx, int *out_len, double *out_results);
int LGBM_BoosterSaveModelToString(BoosterHandle handle, int start_iteration, int num_iteration,
                                  int feature_importance_type, int64_t buffer_len, int64_t *out_len, char *out_str);
int LGBM_BoosterFree(BoosterHandle handle);
}
#ifndef C_API_DTYPE_FLOAT64
#define C_API_DTYPE_FLOAT64 1
#endif
#ifndef C_API_DTYPE_FLOAT32
#define C_API_DTYPE_FLOAT32 0
#endif
#endif

namespace duckdb {
namespace duckboost {

namespace {

void EnsureRectangular(const vector<double> &y, const vector<vector<double>> &x) {
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot train native model on empty dataset");
	}
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch during native train");
	}
	const idx_t n_features = x[0].size();
	if (n_features == 0) {
		throw InvalidInputException("duckboost: native train requires at least one feature");
	}
	for (idx_t i = 0; i < x.size(); i++) {
		if (x[i].size() != n_features) {
			throw InvalidInputException("duckboost: jagged feature matrix at row %llu", (unsigned long long)i);
		}
	}
}

ImportOptions ImportOptionsFromTrain(const TrainOptions &options) {
	ImportOptions import_options;
	import_options.task = options.task;
	import_options.task_set = true;
	import_options.feature_names = options.feature_names;
	return import_options;
}

string ObjectiveForXGBoost(BoostTask task, BoostObjective objective, idx_t n_classes) {
	auto resolved = ResolveObjective(task, objective);
	switch (resolved) {
	case BoostObjective::LOGISTIC:
		return "binary:logistic";
	case BoostObjective::SOFTMAX:
		if (n_classes < 2) {
			throw InvalidInputException("duckboost: xgboost multiclass train requires n_classes >= 2");
		}
		return "multi:softprob";
	case BoostObjective::POISSON:
		return "count:poisson";
	case BoostObjective::HUBER:
		return "reg:pseudohubererror";
	case BoostObjective::QUANTILE:
		return "reg:quantileerror";
	case BoostObjective::SQUAREDERROR:
	case BoostObjective::AUTO:
	default:
		return "reg:squarederror";
	}
}

string ObjectiveForLightGBM(BoostTask task, BoostObjective objective, idx_t n_classes) {
	auto resolved = ResolveObjective(task, objective);
	switch (resolved) {
	case BoostObjective::LOGISTIC:
		return "binary";
	case BoostObjective::SOFTMAX:
		if (n_classes < 2) {
			throw InvalidInputException("duckboost: lightgbm multiclass train requires n_classes >= 2");
		}
		return "multiclass";
	case BoostObjective::POISSON:
		return "poisson";
	case BoostObjective::HUBER:
		return "huber";
	case BoostObjective::QUANTILE:
		return "quantile";
	case BoostObjective::SQUAREDERROR:
	case BoostObjective::AUTO:
	default:
		return "regression";
	}
}

string EvalMetricForObjective(BoostObjective objective) {
	switch (objective) {
	case BoostObjective::LOGISTIC:
		return "logloss";
	case BoostObjective::SOFTMAX:
		return "mlogloss";
	case BoostObjective::QUANTILE:
		return "mae";
	case BoostObjective::POISSON:
	case BoostObjective::HUBER:
	case BoostObjective::SQUAREDERROR:
	case BoostObjective::AUTO:
	default:
		return "rmse";
	}
}

struct RowSplit {
	vector<idx_t> train_rows;
	vector<idx_t> valid_rows;
};

RowSplit MakeValidationSplit(idx_t n_rows, const TrainOptions &options) {
	RowSplit split;
	split.train_rows.resize(n_rows);
	std::iota(split.train_rows.begin(), split.train_rows.end(), 0);
	if (!(options.validation_fraction > 0 && options.early_stopping_rounds > 0 && n_rows >= 4)) {
		return split;
	}
	uint64_t state = options.seed ? options.seed : 0x9e3779b97f4a7c15ULL;
	auto next = [&]() {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		return state;
	};
	auto shuffled = split.train_rows;
	for (idx_t i = 0; i < shuffled.size(); i++) {
		idx_t j = i + static_cast<idx_t>(next() % (shuffled.size() - i));
		std::swap(shuffled[i], shuffled[j]);
	}
	idx_t valid_n =
	    MaxValue<idx_t>(1, static_cast<idx_t>(std::floor(options.validation_fraction * static_cast<double>(n_rows))));
	valid_n = MinValue<idx_t>(valid_n, n_rows - 1);
	split.valid_rows.assign(shuffled.begin(), shuffled.begin() + valid_n);
	split.train_rows.assign(shuffled.begin() + valid_n, shuffled.end());
	std::sort(split.train_rows.begin(), split.train_rows.end());
	std::sort(split.valid_rows.begin(), split.valid_rows.end());
	return split;
}

void MaterializeRowSubset(const vector<vector<double>> &x, const vector<double> &y, const vector<float> &weights,
                          const vector<idx_t> &rows, idx_t ncol, vector<float> &flat, vector<float> &labels,
                          vector<float> &weight_out) {
	flat.resize(rows.size() * ncol);
	labels.resize(rows.size());
	weight_out.resize(rows.size());
	for (idx_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		labels[i] = static_cast<float>(y[row]);
		weight_out[i] = weights[row];
		for (idx_t j = 0; j < ncol; j++) {
			flat[i * ncol + j] = static_cast<float>(x[row][j]);
		}
	}
}

double RMSEFromPreds(const float *preds, const vector<float> &labels) {
	if (labels.empty()) {
		return std::numeric_limits<double>::infinity();
	}
	double sse = 0;
	for (idx_t i = 0; i < labels.size(); i++) {
		double err = static_cast<double>(preds[i]) - static_cast<double>(labels[i]);
		sse += err * err;
	}
	return std::sqrt(sse / static_cast<double>(labels.size()));
}

idx_t InferClassCount(const vector<double> &y, const TrainOptions &options) {
	if (options.task != BoostTask::MULTICLASS) {
		return 1;
	}
	double max_label = -1;
	for (auto v : y) {
		if (!std::isfinite(v) || v < 0) {
			throw InvalidInputException("duckboost: multiclass labels must be finite non-negative class indices");
		}
		max_label = MaxValue(max_label, v);
	}
	auto inferred = static_cast<idx_t>(max_label) + 1;
	if (inferred < 2) {
		throw InvalidInputException("duckboost: multiclass train requires at least 2 classes");
	}
	return inferred;
}

idx_t EffectiveMaxLeavesNative(const TrainOptions &options) {
	if (options.max_leaves > 0) {
		return options.max_leaves;
	}
	return 31;
}

vector<idx_t> ResolveNativeCatFeatures(const TrainOptions &options, idx_t n_features) {
	vector<idx_t> cats = options.cat_features;
	for (auto &token : options.cat_feature_tokens) {
		bool all_digits = !token.empty() && std::isdigit(static_cast<unsigned char>(token[0]));
		for (idx_t i = 1; all_digits && i < token.size(); i++) {
			if (!std::isdigit(static_cast<unsigned char>(token[i]))) {
				all_digits = false;
			}
		}
		if (all_digits) {
			continue;
		}
		bool found = false;
		for (idx_t i = 0; i < options.feature_names.size(); i++) {
			if (StringUtil::Lower(options.feature_names[i]) == StringUtil::Lower(token)) {
				cats.push_back(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InvalidInputException("duckboost: cat_features name '%s' not found in feature_names", token);
		}
	}
	std::sort(cats.begin(), cats.end());
	cats.erase(std::unique(cats.begin(), cats.end()), cats.end());
	for (auto idx : cats) {
		if (idx >= n_features) {
			throw InvalidInputException("duckboost: cat_features index %llu out of range for %llu features",
			                            (unsigned long long)idx, (unsigned long long)n_features);
		}
	}
	return cats;
}

vector<float> ResolveNativeWeights(const vector<double> &y, const TrainOptions &options, const vector<double> &weights,
                                   idx_t n_classes_for_weight) {
	vector<float> out(y.size(), 1.0f);
	if (!weights.empty()) {
		if (weights.size() != y.size()) {
			throw InvalidInputException("duckboost: sample weight count (%llu) must match row count (%llu)",
			                            (unsigned long long)weights.size(), (unsigned long long)y.size());
		}
		for (idx_t i = 0; i < y.size(); i++) {
			if (!std::isfinite(weights[i]) || weights[i] < 0) {
				throw InvalidInputException("duckboost: sample weights must be finite and >= 0");
			}
			out[i] = static_cast<float>(weights[i]);
		}
	}
	if (options.class_weight.empty() || n_classes_for_weight < 2) {
		return out;
	}
	vector<double> multipliers(n_classes_for_weight, 1.0);
	auto lower = StringUtil::Lower(options.class_weight);
	if (lower == "balanced") {
		vector<double> counts(n_classes_for_weight, 0);
		double total = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto label = static_cast<idx_t>(y[i]);
			if (label >= n_classes_for_weight) {
				continue;
			}
			counts[label] += out[i];
			total += out[i];
		}
		for (idx_t c = 0; c < n_classes_for_weight; c++) {
			multipliers[c] = counts[c] <= 0 ? 0 : total / (static_cast<double>(n_classes_for_weight) * counts[c]);
		}
	} else {
		auto parts = StringUtil::Split(options.class_weight, ',');
		if (parts.size() != n_classes_for_weight) {
			throw InvalidInputException("duckboost: class_weight list length (%llu) must match n_classes (%llu)",
			                            (unsigned long long)parts.size(), (unsigned long long)n_classes_for_weight);
		}
		for (idx_t c = 0; c < n_classes_for_weight; c++) {
			StringUtil::Trim(parts[c]);
			multipliers[c] = std::stod(parts[c]);
		}
	}
	for (idx_t i = 0; i < y.size(); i++) {
		auto label = static_cast<idx_t>(y[i]);
		if (label < n_classes_for_weight) {
			out[i] = static_cast<float>(static_cast<double>(out[i]) * multipliers[label]);
		}
	}
	return out;
}

#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)

[[noreturn]] void ThrowXGBoostError(const char *context) {
	const char *err = XGBGetLastError();
	throw InvalidInputException("duckboost: xgboost %s failed: %s", context, err ? err : "unknown error");
}

string DumpXGBoostJSON(BoosterHandle booster, idx_t ncol, const vector<idx_t> &cat_indices,
                       const TrainOptions &options) {
	bst_ulong out_len = 0;
	const char **out_dump = nullptr;
	vector<string> dummy_names;
	vector<const char *> fnames(ncol);
	vector<const char *> ftypes(ncol, "q");
	for (auto idx : cat_indices) {
		ftypes[idx] = "c";
	}
	if (!options.feature_names.empty() && options.feature_names.size() == ncol) {
		for (idx_t i = 0; i < ncol; i++) {
			fnames[i] = options.feature_names[i].c_str();
		}
	} else {
		dummy_names.resize(ncol);
		for (idx_t i = 0; i < ncol; i++) {
			dummy_names[i] = "f" + std::to_string(i);
			fnames[i] = dummy_names[i].c_str();
		}
	}
	int dump_rc = XGBoosterDumpModelExWithFeatures(booster, static_cast<int>(ncol), fnames.data(), ftypes.data(), 0,
	                                               "json", &out_len, &out_dump);
	if (dump_rc != 0 || !out_dump) {
		dump_rc = XGBoosterDumpModelEx(booster, "", 0, "json", &out_len, &out_dump);
	}
	if (dump_rc != 0 || !out_dump) {
		return string();
	}
	string dump = "[";
	for (bst_ulong i = 0; i < out_len; i++) {
		if (i > 0) {
			dump += ",";
		}
		dump += out_dump[i] ? out_dump[i] : "{}";
	}
	dump += "]";
	return dump;
}

BoostModel TrainWithXGBoost(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                            const vector<double> &weights) {
	EnsureRectangular(y, x);
	const idx_t nrow = y.size();
	const idx_t ncol = x[0].size();
	const idx_t n_classes = InferClassCount(y, options);
	const auto objective = ResolveObjective(options.task, options.objective);
	const idx_t weight_classes = options.task == BoostTask::BINARY ? 2 : n_classes;
	auto cat_indices = ResolveNativeCatFeatures(options, ncol);
	auto weight_f = ResolveNativeWeights(y, options, weights, weight_classes);
	auto split = MakeValidationSplit(nrow, options);

	vector<float> train_flat, train_labels, train_weights;
	MaterializeRowSubset(x, y, weight_f, split.train_rows, ncol, train_flat, train_labels, train_weights);

	DMatrixHandle dtrain = nullptr;
	if (XGDMatrixCreateFromMat(train_flat.data(), static_cast<bst_ulong>(split.train_rows.size()),
	                           static_cast<bst_ulong>(ncol), std::numeric_limits<float>::quiet_NaN(), &dtrain) != 0) {
		ThrowXGBoostError("XGDMatrixCreateFromMat(train)");
	}
	if (XGDMatrixSetFloatInfo(dtrain, "label", train_labels.data(), static_cast<bst_ulong>(train_labels.size())) != 0) {
		XGDMatrixFree(dtrain);
		ThrowXGBoostError("XGDMatrixSetFloatInfo(label)");
	}
	if (XGDMatrixSetFloatInfo(dtrain, "weight", train_weights.data(), static_cast<bst_ulong>(train_weights.size())) !=
	    0) {
		XGDMatrixFree(dtrain);
		ThrowXGBoostError("XGDMatrixSetFloatInfo(weight)");
	}

	DMatrixHandle dvalid = nullptr;
	vector<float> valid_flat, valid_labels, valid_weights;
	if (!split.valid_rows.empty()) {
		MaterializeRowSubset(x, y, weight_f, split.valid_rows, ncol, valid_flat, valid_labels, valid_weights);
		if (XGDMatrixCreateFromMat(valid_flat.data(), static_cast<bst_ulong>(split.valid_rows.size()),
		                           static_cast<bst_ulong>(ncol), std::numeric_limits<float>::quiet_NaN(),
		                           &dvalid) != 0) {
			XGDMatrixFree(dtrain);
			ThrowXGBoostError("XGDMatrixCreateFromMat(valid)");
		}
		if (XGDMatrixSetFloatInfo(dvalid, "label", valid_labels.data(), static_cast<bst_ulong>(valid_labels.size())) !=
		    0) {
			XGDMatrixFree(dvalid);
			XGDMatrixFree(dtrain);
			ThrowXGBoostError("XGDMatrixSetFloatInfo(valid label)");
		}
	}

	BoosterHandle booster = nullptr;
	const DMatrixHandle dmats[] = {dtrain};
	if (XGBoosterCreate(dmats, 1, &booster) != 0) {
		if (dvalid) {
			XGDMatrixFree(dvalid);
		}
		XGDMatrixFree(dtrain);
		ThrowXGBoostError("XGBoosterCreate");
	}

	auto cleanup = [&]() {
		XGBoosterFree(booster);
		if (dvalid) {
			XGDMatrixFree(dvalid);
		}
		XGDMatrixFree(dtrain);
	};

	auto set_param = [&](const char *name, const string &value) {
		if (XGBoosterSetParam(booster, name, value.c_str()) != 0) {
			cleanup();
			ThrowXGBoostError((string("XGBoosterSetParam(") + name + ")").c_str());
		}
	};

	set_param("verbosity", "0");
	set_param("max_depth", std::to_string(options.max_depth));
	set_param("max_bin", std::to_string(MaxValue<idx_t>(options.max_bins, 2)));
	set_param("eta", std::to_string(options.learning_rate));
	set_param("min_child_weight", std::to_string(options.min_child_weight));
	set_param("lambda", std::to_string(options.reg_lambda));
	set_param("alpha", std::to_string(options.reg_alpha));
	set_param("gamma", std::to_string(options.min_split_gain));
	set_param("subsample", std::to_string(options.subsample));
	set_param("colsample_bytree", std::to_string(options.colsample_bytree));
	set_param("seed", std::to_string(options.seed));
	set_param("objective", ObjectiveForXGBoost(options.task, options.objective, n_classes));
	set_param("eval_metric", EvalMetricForObjective(objective));
	if (objective == BoostObjective::HUBER) {
		set_param("huber_slope", std::to_string(options.huber_delta));
	}
	if (objective == BoostObjective::QUANTILE) {
		set_param("quantile_alpha", std::to_string(options.quantile_alpha));
	}
	if (options.grow_policy == GrowPolicy::LEAF) {
		set_param("grow_policy", "lossguide");
		set_param("max_leaves", std::to_string(EffectiveMaxLeavesNative(options)));
	} else {
		set_param("grow_policy", "depthwise");
	}
	if (options.task == BoostTask::MULTICLASS) {
		set_param("num_class", std::to_string(n_classes));
	}

	string best_dump;
	double best_metric = std::numeric_limits<double>::infinity();
	idx_t rounds_since_improve = 0;

	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		if (XGBoosterUpdateOneIter(booster, static_cast<int>(iter), dtrain) != 0) {
			cleanup();
			ThrowXGBoostError("XGBoosterUpdateOneIter");
		}
		if (!dvalid) {
			continue;
		}
		bst_ulong pred_len = 0;
		const float *preds = nullptr;
		if (XGBoosterPredict(booster, dvalid, 0, 0, 0, &pred_len, &preds) != 0 || !preds) {
			cleanup();
			ThrowXGBoostError("XGBoosterPredict(valid)");
		}
		// Multiclass softprob expands to n_classes scores/row; use first-score RMSE proxy only for regression-like.
		double metric = std::numeric_limits<double>::infinity();
		if (options.task == BoostTask::MULTICLASS && pred_len == valid_labels.size() * n_classes) {
			idx_t correct = 0;
			for (idx_t i = 0; i < valid_labels.size(); i++) {
				idx_t best = 0;
				for (idx_t c = 1; c < n_classes; c++) {
					if (preds[i * n_classes + c] > preds[i * n_classes + best]) {
						best = c;
					}
				}
				if (static_cast<float>(best) == valid_labels[i]) {
					correct++;
				}
			}
			metric = 1.0 - static_cast<double>(correct) / static_cast<double>(valid_labels.size());
		} else if (pred_len >= valid_labels.size()) {
			metric = RMSEFromPreds(preds, valid_labels);
		}
		if (metric < best_metric - 1e-12) {
			best_metric = metric;
			rounds_since_improve = 0;
			best_dump = DumpXGBoostJSON(booster, ncol, cat_indices, options);
			if (best_dump.empty()) {
				cleanup();
				ThrowXGBoostError("XGBoosterDumpModelEx");
			}
		} else {
			rounds_since_improve++;
			if (rounds_since_improve >= options.early_stopping_rounds) {
				break;
			}
		}
	}

	string dump = best_dump;
	if (dump.empty()) {
		dump = DumpXGBoostJSON(booster, ncol, cat_indices, options);
		if (dump.empty()) {
			cleanup();
			ThrowXGBoostError("XGBoosterDumpModelEx");
		}
	}

	cleanup();

	auto import_options = ImportOptionsFromTrain(options);
	if (options.task == BoostTask::MULTICLASS) {
		import_options.n_classes = n_classes;
		import_options.n_classes_set = true;
	}
	return ImportXGBoostJSON(dump, import_options);
}

#endif // XGBoost linked

#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)

[[noreturn]] void ThrowLightGBMError(const char *context) {
	const char *err = LGBM_GetLastError();
	throw InvalidInputException("duckboost: lightgbm %s failed: %s", context, err ? err : "unknown error");
}

void MaterializeRowSubsetDouble(const vector<vector<double>> &x, const vector<double> &y, const vector<float> &weights,
                                const vector<idx_t> &rows, idx_t ncol, vector<double> &flat, vector<float> &labels,
                                vector<float> &weight_out) {
	flat.resize(rows.size() * ncol);
	labels.resize(rows.size());
	weight_out.resize(rows.size());
	for (idx_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		labels[i] = static_cast<float>(y[row]);
		weight_out[i] = weights[row];
		for (idx_t j = 0; j < ncol; j++) {
			flat[i * ncol + j] = x[row][j];
		}
	}
}

string SaveLightGBMModel(BoosterHandle booster, int num_iteration) {
	int64_t out_len = 0;
	if (LGBM_BoosterSaveModelToString(booster, 0, num_iteration, 0, 0, &out_len, nullptr) != 0) {
		return string();
	}
	vector<char> buffer(static_cast<idx_t>(out_len) + 1);
	if (LGBM_BoosterSaveModelToString(booster, 0, num_iteration, 0, out_len, &out_len, buffer.data()) != 0) {
		return string();
	}
	return string(buffer.data());
}

BoostModel TrainWithLightGBM(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                             const vector<double> &weights) {
	EnsureRectangular(y, x);
	const idx_t nrow = y.size();
	const idx_t ncol = x[0].size();
	const idx_t n_classes = InferClassCount(y, options);
	const auto objective = ResolveObjective(options.task, options.objective);
	const idx_t weight_classes = options.task == BoostTask::BINARY ? 2 : n_classes;
	auto cat_indices = ResolveNativeCatFeatures(options, ncol);
	auto weight_f = ResolveNativeWeights(y, options, weights, weight_classes);
	auto split = MakeValidationSplit(nrow, options);

	vector<double> train_flat;
	vector<float> train_labels, train_weights;
	MaterializeRowSubsetDouble(x, y, weight_f, split.train_rows, ncol, train_flat, train_labels, train_weights);

	string dataset_params = "max_bin=" + std::to_string(MaxValue<idx_t>(options.max_bins, 2));
	if (!cat_indices.empty()) {
		dataset_params += " categorical_feature=";
		for (idx_t i = 0; i < cat_indices.size(); i++) {
			if (i > 0) {
				dataset_params += ',';
			}
			dataset_params += std::to_string(cat_indices[i]);
		}
	}

	auto train_nrow = NumericCast<int32_t>(split.train_rows.size());
	auto ncol_i = NumericCast<int32_t>(ncol);

	DatasetHandle dataset = nullptr;
	if (LGBM_DatasetCreateFromMat(train_flat.data(), C_API_DTYPE_FLOAT64, train_nrow, ncol_i, 1, dataset_params.c_str(),
	                              nullptr, &dataset) != 0) {
		ThrowLightGBMError("LGBM_DatasetCreateFromMat");
	}
	if (LGBM_DatasetSetField(dataset, "label", train_labels.data(), train_nrow, C_API_DTYPE_FLOAT32) != 0) {
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_DatasetSetField(label)");
	}
	if (LGBM_DatasetSetField(dataset, "weight", train_weights.data(), train_nrow, C_API_DTYPE_FLOAT32) != 0) {
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_DatasetSetField(weight)");
	}
	if (!options.feature_names.empty() && options.feature_names.size() == ncol) {
		vector<const char *> fnames(ncol);
		for (idx_t i = 0; i < ncol; i++) {
			fnames[i] = options.feature_names[i].c_str();
		}
		if (LGBM_DatasetSetFeatureNames(dataset, fnames.data(), ncol_i) != 0) {
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_DatasetSetFeatureNames");
		}
	}

	DatasetHandle valid_dataset = nullptr;
	vector<double> valid_flat;
	vector<float> valid_labels, valid_weights;
	if (!split.valid_rows.empty()) {
		MaterializeRowSubsetDouble(x, y, weight_f, split.valid_rows, ncol, valid_flat, valid_labels, valid_weights);
		auto valid_nrow = NumericCast<int32_t>(split.valid_rows.size());
		if (LGBM_DatasetCreateFromMat(valid_flat.data(), C_API_DTYPE_FLOAT64, valid_nrow, ncol_i, 1,
		                              dataset_params.c_str(), dataset, &valid_dataset) != 0) {
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_DatasetCreateFromMat(valid)");
		}
		if (LGBM_DatasetSetField(valid_dataset, "label", valid_labels.data(), valid_nrow, C_API_DTYPE_FLOAT32) != 0) {
			LGBM_DatasetFree(valid_dataset);
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_DatasetSetField(valid label)");
		}
	}

	idx_t num_leaves = options.grow_policy == GrowPolicy::LEAF
	                       ? EffectiveMaxLeavesNative(options)
	                       : MaxValue<idx_t>(2, 1ULL << MinValue<idx_t>(options.max_depth, 10));
	if (options.max_leaves > 0) {
		num_leaves = options.max_leaves;
	}
	string params = StringUtil::Format(
	    "objective=%s learning_rate=%g num_leaves=%llu max_depth=%llu min_data_in_leaf=%llu "
	    "min_sum_hessian_in_leaf=%g lambda_l2=%g lambda_l1=%g min_gain_to_split=%g "
	    "bagging_fraction=%g feature_fraction=%g bagging_freq=1 seed=%llu verbosity=-1 force_col_wise=true "
	    "metric=%s",
	    ObjectiveForLightGBM(options.task, options.objective, n_classes), options.learning_rate,
	    (unsigned long long)num_leaves, (unsigned long long)options.max_depth,
	    (unsigned long long)MaxValue<idx_t>(options.min_samples_leaf, 1), options.min_child_weight, options.reg_lambda,
	    options.reg_alpha, options.min_split_gain, options.subsample, options.colsample_bytree,
	    (unsigned long long)options.seed, EvalMetricForObjective(objective));
	if (options.task == BoostTask::MULTICLASS) {
		params += " num_class=" + std::to_string(n_classes);
	}
	if (objective == BoostObjective::HUBER) {
		params += " alpha=" + std::to_string(options.huber_delta);
	} else if (objective == BoostObjective::QUANTILE) {
		params += " alpha=" + std::to_string(options.quantile_alpha);
	}

	BoosterHandle booster = nullptr;
	if (LGBM_BoosterCreate(dataset, params.c_str(), &booster) != 0) {
		if (valid_dataset) {
			LGBM_DatasetFree(valid_dataset);
		}
		LGBM_DatasetFree(dataset);
		ThrowLightGBMError("LGBM_BoosterCreate");
	}
	if (valid_dataset) {
		if (LGBM_BoosterAddValidData(booster, valid_dataset) != 0) {
			LGBM_BoosterFree(booster);
			LGBM_DatasetFree(valid_dataset);
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_BoosterAddValidData");
		}
	}

	idx_t best_rounds = 0;
	double best_metric = std::numeric_limits<double>::infinity();
	idx_t rounds_since_improve = 0;

	for (idx_t iter = 0; iter < options.n_estimators; iter++) {
		int is_finished = 0;
		if (LGBM_BoosterUpdateOneIter(booster, &is_finished) != 0) {
			LGBM_BoosterFree(booster);
			if (valid_dataset) {
				LGBM_DatasetFree(valid_dataset);
			}
			LGBM_DatasetFree(dataset);
			ThrowLightGBMError("LGBM_BoosterUpdateOneIter");
		}
		if (valid_dataset) {
			int eval_len = 0;
			double eval_buf[8];
			// data_idx 1 = first validation set
			if (LGBM_BoosterGetEval(booster, 1, &eval_len, eval_buf) == 0 && eval_len > 0) {
				double metric = eval_buf[0];
				if (metric < best_metric - 1e-12) {
					best_metric = metric;
					best_rounds = iter + 1;
					rounds_since_improve = 0;
				} else {
					rounds_since_improve++;
					if (rounds_since_improve >= options.early_stopping_rounds) {
						break;
					}
				}
			}
		}
		if (is_finished) {
			break;
		}
	}

	int save_iters = best_rounds > 0 ? static_cast<int>(best_rounds) : -1;
	string dump = SaveLightGBMModel(booster, save_iters);
	LGBM_BoosterFree(booster);
	if (valid_dataset) {
		LGBM_DatasetFree(valid_dataset);
	}
	LGBM_DatasetFree(dataset);
	if (dump.empty()) {
		ThrowLightGBMError("LGBM_BoosterSaveModelToString");
	}

	auto import_options = ImportOptionsFromTrain(options);
	if (options.task == BoostTask::MULTICLASS) {
		import_options.n_classes = n_classes;
		import_options.n_classes_set = true;
	}
	return ImportLightGBMText(dump, import_options);
}

#endif // LightGBM linked

} // namespace

bool NativeTrainerCompiled(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM)
		return true;
#else
		return false;
#endif
	case BoostBackend::CATBOOST:
#if defined(DUCKBOOST_WITH_CATBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::REFERENCE:
	default:
		return false;
	}
}

bool NativeTrainerLinked(BoostBackend backend) {
#if defined(DUCKBOOST_NATIVE_STUB)
	(void)backend;
	return false;
#else
	switch (backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM)
		return true;
#else
		return false;
#endif
	case BoostBackend::CATBOOST:
		// CatBoost ships a model-application C API, not a public in-process training C API.
		return false;
	case BoostBackend::REFERENCE:
	default:
		return false;
	}
#endif
}

BoostModel TrainNative(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                       const vector<double> &weights) {
	if (!NativeTrainerCompiled(options.backend)) {
		throw NotImplementedException("duckboost: native training for backend '%s' is not linked in this build. "
		                              "Configure with -DDUCKBOOST_WITH_%s=ON (and install the vendor library), "
		                              "or use backend='reference' / duckboost_import().",
		                              BackendToString(options.backend),
		                              StringUtil::Upper(BackendToString(options.backend)));
	}

#if defined(DUCKBOOST_NATIVE_STUB)
	throw NotImplementedException("duckboost: native trainer for backend '%s' is compiled as a stub "
	                              "(DUCKBOOST_NATIVE_STUB_ONLY). Rebuild with the vendor library linked, "
	                              "or use duckboost_import() / backend='reference'.",
	                              BackendToString(options.backend));
#endif

	switch (options.backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST) && !defined(DUCKBOOST_NATIVE_STUB)
		return TrainWithXGBoost(y, x, options, weights);
#else
		break;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM) && !defined(DUCKBOOST_NATIVE_STUB)
		return TrainWithLightGBM(y, x, options, weights);
#else
		break;
#endif
	case BoostBackend::CATBOOST:
		throw NotImplementedException(
		    "duckboost: CatBoost has no public in-process training C API. "
		    "Train with the CatBoost CLI/Python API and load the JSON dump via duckboost_import('catboost', ...).");
	default:
		break;
	}

	throw NotImplementedException("duckboost: native trainer for backend '%s' is not available in this build. "
	                              "Use duckboost_import() or backend='reference'.",
	                              BackendToString(options.backend));
}

} // namespace duckboost
} // namespace duckdb
