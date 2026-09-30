#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace duckdb {
namespace duckboost {

namespace {

struct GradStat {
	double g = 0;
	double h = 0;

	void Add(double gg, double hh) {
		g += gg;
		h += hh;
	}

	void Add(const GradStat &other) {
		g += other.g;
		h += other.h;
	}

	GradStat Without(const GradStat &other) const {
		return {g - other.g, h - other.h};
	}
};

struct SplitCandidate {
	idx_t feature = 0;
	double threshold = 0;
	double gain = -std::numeric_limits<double>::infinity();
	bool default_left = true;
	SplitCompare compare = SplitCompare::LESS;
};

struct SimpleRng {
	uint64_t state;

	explicit SimpleRng(uint64_t seed) : state(seed ? seed : 0x9e3779b97f4a7c15ULL) {
	}

	uint64_t Next() {
		state = state * 6364136223846793005ULL + 1;
		return state;
	}

	double Uniform() {
		return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0);
	}

	idx_t Bounded(idx_t n) {
		if (n <= 1) {
			return 0;
		}
		return static_cast<idx_t>(Next() % n);
	}
};

bool IsMissing(double value) {
	return std::isnan(value);
}

double SoftThreshold(double g, double alpha) {
	if (g > alpha) {
		return g - alpha;
	}
	if (g < -alpha) {
		return g + alpha;
	}
	return 0;
}

double LeafWeight(const GradStat &stat, const TrainOptions &options) {
	if (stat.h <= 0) {
		return 0;
	}
	return -SoftThreshold(stat.g, options.reg_alpha) / (stat.h + options.reg_lambda);
}

double LeafScore(const GradStat &stat, const TrainOptions &options) {
	auto g = SoftThreshold(stat.g, options.reg_alpha);
	return (g * g) / (stat.h + options.reg_lambda);
}

double SplitGain(const GradStat &left, const GradStat &right, const GradStat &parent, const TrainOptions &options) {
	return 0.5 * (LeafScore(left, options) + LeafScore(right, options) - LeafScore(parent, options)) -
	       options.min_split_gain;
}

GradStat SumStats(const vector<double> &gradients, const vector<double> &hessians, const vector<idx_t> &rows) {
	GradStat sum;
	for (auto row : rows) {
		sum.Add(gradients[row], hessians[row]);
	}
	return sum;
}

vector<double> CandidateThresholds(const vector<double> &sorted_present_values, idx_t max_bins) {
	if (sorted_present_values.size() <= 1) {
		return {};
	}
	vector<double> unique;
	unique.reserve(sorted_present_values.size());
	for (auto value : sorted_present_values) {
		if (unique.empty() || value != unique.back()) {
			unique.push_back(value);
		}
	}
	if (unique.size() <= 1) {
		return {};
	}
	vector<double> thresholds;
	if (unique.size() <= max_bins + 1) {
		thresholds.reserve(unique.size() - 1);
		for (idx_t i = 0; i + 1 < unique.size(); i++) {
			thresholds.push_back(0.5 * (unique[i] + unique[i + 1]));
		}
		return thresholds;
	}
	for (idx_t b = 1; b <= max_bins; b++) {
		double q = static_cast<double>(b) / static_cast<double>(max_bins + 1);
		idx_t idx = static_cast<idx_t>(q * static_cast<double>(unique.size() - 1));
		if (idx + 1 < unique.size()) {
			thresholds.push_back(0.5 * (unique[idx] + unique[idx + 1]));
		}
	}
	std::sort(thresholds.begin(), thresholds.end());
	thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
	return thresholds;
}

vector<double> UniquePresentValues(const vector<double> &sorted_present_values, idx_t max_bins) {
	vector<double> unique;
	for (auto value : sorted_present_values) {
		if (unique.empty() || value != unique.back()) {
			unique.push_back(value);
		}
	}
	if (unique.size() <= max_bins) {
		return unique;
	}
	vector<double> sampled;
	sampled.reserve(max_bins);
	for (idx_t b = 0; b < max_bins; b++) {
		idx_t idx = static_cast<idx_t>((static_cast<double>(b) + 0.5) * static_cast<double>(unique.size()) /
		                               static_cast<double>(max_bins));
		idx = MinValue<idx_t>(idx, unique.size() - 1);
		sampled.push_back(unique[idx]);
	}
	std::sort(sampled.begin(), sampled.end());
	sampled.erase(std::unique(sampled.begin(), sampled.end()), sampled.end());
	return sampled;
}

bool ChildFeasible(const GradStat &stat, idx_t row_count, const TrainOptions &options) {
	return row_count >= options.min_samples_leaf && stat.h >= options.min_child_weight;
}

void ConsiderSplit(SplitCandidate &best, idx_t feature, double threshold, bool default_left, SplitCompare compare,
                   const GradStat &left, const GradStat &right, const GradStat &parent, idx_t left_rows,
                   idx_t right_rows, const TrainOptions &options) {
	if (!ChildFeasible(left, left_rows, options) || !ChildFeasible(right, right_rows, options)) {
		return;
	}
	auto gain = SplitGain(left, right, parent, options);
	if (std::isfinite(gain) && gain > best.gain) {
		best.gain = gain;
		best.feature = feature;
		best.threshold = threshold;
		best.default_left = default_left;
		best.compare = compare;
	}
}

bool FeatureIsCategorical(idx_t feature, const unordered_set<idx_t> &cat_set) {
	return cat_set.find(feature) != cat_set.end();
}

SplitCandidate FindBestSplit(const vector<vector<double>> &x, const vector<double> &gradients,
                             const vector<double> &hessians, const vector<idx_t> &rows,
                             const vector<idx_t> &feature_subset, const unordered_set<idx_t> &cat_set,
                             const TrainOptions &options) {
	SplitCandidate best;
	if (rows.size() < 2 * options.min_samples_leaf) {
		return best;
	}
	auto parent = SumStats(gradients, hessians, rows);
	for (auto f : feature_subset) {
		vector<idx_t> present;
		GradStat missing_stat;
		idx_t missing_count = 0;
		present.reserve(rows.size());
		for (auto row : rows) {
			auto value = x[row][f];
			if (IsMissing(value)) {
				missing_stat.Add(gradients[row], hessians[row]);
				missing_count++;
			} else {
				present.push_back(row);
			}
		}
		if (present.size() < 2) {
			continue;
		}
		std::sort(present.begin(), present.end(), [&](idx_t a, idx_t b) { return x[a][f] < x[b][f]; });
		vector<double> present_values;
		present_values.reserve(present.size());
		for (auto row : present) {
			present_values.push_back(x[row][f]);
		}

		if (FeatureIsCategorical(f, cat_set)) {
			auto levels = UniquePresentValues(present_values, options.max_bins);
			for (auto level : levels) {
				GradStat equal_stat;
				idx_t equal_count = 0;
				for (auto row : present) {
					if (x[row][f] == level) {
						equal_stat.Add(gradients[row], hessians[row]);
						equal_count++;
					}
				}
				idx_t neq_count = present.size() - equal_count;
				if (equal_count == 0 || neq_count == 0) {
					continue;
				}
				auto neq_stat = parent.Without(equal_stat).Without(missing_stat);
				// EQUAL: match → right, else → left (same as EvalTree / CatBoost OneHot).
				GradStat left_m = neq_stat;
				left_m.Add(missing_stat);
				ConsiderSplit(best, f, level, true, SplitCompare::EQUAL, left_m, equal_stat, parent,
				              neq_count + missing_count, equal_count, options);
				GradStat right_m = equal_stat;
				right_m.Add(missing_stat);
				ConsiderSplit(best, f, level, false, SplitCompare::EQUAL, neq_stat, right_m, parent, neq_count,
				              equal_count + missing_count, options);
			}
			continue;
		}

		auto thresholds = CandidateThresholds(present_values, options.max_bins);
		if (thresholds.empty()) {
			continue;
		}

		GradStat left_present;
		idx_t left_count = 0;
		idx_t cursor = 0;
		for (auto threshold : thresholds) {
			while (cursor < present.size() && x[present[cursor]][f] < threshold) {
				left_present.Add(gradients[present[cursor]], hessians[present[cursor]]);
				left_count++;
				cursor++;
			}
			auto right_present = parent.Without(left_present).Without(missing_stat);
			idx_t right_count = present.size() - left_count;
			if (left_count == 0 || right_count == 0) {
				continue;
			}

			GradStat left_m = left_present;
			left_m.Add(missing_stat);
			ConsiderSplit(best, f, threshold, true, SplitCompare::LESS, left_m, right_present, parent,
			              left_count + missing_count, right_count, options);
			GradStat right_m = right_present;
			right_m.Add(missing_stat);
			ConsiderSplit(best, f, threshold, false, SplitCompare::LESS, left_present, right_m, parent, left_count,
			              right_count + missing_count, options);
		}
	}
	return best;
}

idx_t BuildLeaf(BoostTree &tree, double value) {
	TreeNode node;
	node.is_leaf = true;
	node.value = value;
	tree.nodes.push_back(node);
	return tree.nodes.size() - 1;
}

void PartitionRows(const vector<vector<double>> &x, const vector<idx_t> &rows, const SplitCandidate &split,
                   vector<idx_t> &left_rows, vector<idx_t> &right_rows) {
	left_rows.clear();
	right_rows.clear();
	left_rows.reserve(rows.size());
	right_rows.reserve(rows.size());
	for (auto row : rows) {
		auto value = x[row][split.feature];
		bool go_left;
		if (IsMissing(value)) {
			go_left = split.default_left;
		} else if (split.compare == SplitCompare::EQUAL) {
			go_left = value != split.threshold;
		} else {
			go_left = value < split.threshold;
		}
		if (go_left) {
			left_rows.push_back(row);
		} else {
			right_rows.push_back(row);
		}
	}
}

idx_t BuildTree(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &gradients,
                const vector<double> &hessians, const vector<idx_t> &rows, const vector<idx_t> &feature_subset,
                const unordered_set<idx_t> &cat_set, idx_t depth, const TrainOptions &options) {
	auto parent = SumStats(gradients, hessians, rows);
	if (depth >= options.max_depth || rows.size() < 2 * options.min_samples_leaf) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}
	auto split = FindBestSplit(x, gradients, hessians, rows, feature_subset, cat_set, options);
	if (!std::isfinite(split.gain) || split.gain <= 0) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}
	vector<idx_t> left_rows;
	vector<idx_t> right_rows;
	PartitionRows(x, rows, split, left_rows, right_rows);
	if (left_rows.empty() || right_rows.empty()) {
		return BuildLeaf(tree, LeafWeight(parent, options));
	}

	TreeNode node;
	node.is_leaf = false;
	node.feature = split.feature;
	node.threshold = split.threshold;
	node.default_left = split.default_left;
	node.compare = split.compare;
	auto node_idx = tree.nodes.size();
	tree.nodes.push_back(node);
	tree.nodes[node_idx].left =
	    BuildTree(tree, x, gradients, hessians, left_rows, feature_subset, cat_set, depth + 1, options);
	tree.nodes[node_idx].right =
	    BuildTree(tree, x, gradients, hessians, right_rows, feature_subset, cat_set, depth + 1, options);
	return node_idx;
}

vector<idx_t> SampleRows(idx_t n_rows, double subsample, SimpleRng &rng) {
	vector<idx_t> all(n_rows);
	std::iota(all.begin(), all.end(), 0);
	if (subsample >= 1.0 || n_rows == 0) {
		return all;
	}
	idx_t keep = MaxValue<idx_t>(1, static_cast<idx_t>(std::ceil(subsample * static_cast<double>(n_rows))));
	keep = MinValue<idx_t>(keep, n_rows);
	for (idx_t i = 0; i < keep; i++) {
		idx_t j = i + rng.Bounded(n_rows - i);
		std::swap(all[i], all[j]);
	}
	all.resize(keep);
	std::sort(all.begin(), all.end());
	return all;
}

vector<idx_t> SampleFeatures(idx_t n_features, double colsample, SimpleRng &rng) {
	vector<idx_t> all(n_features);
	std::iota(all.begin(), all.end(), 0);
	if (colsample >= 1.0 || n_features == 0) {
		return all;
	}
	idx_t keep = MaxValue<idx_t>(1, static_cast<idx_t>(std::ceil(colsample * static_cast<double>(n_features))));
	keep = MinValue<idx_t>(keep, n_features);
	for (idx_t i = 0; i < keep; i++) {
		idx_t j = i + rng.Bounded(n_features - i);
		std::swap(all[i], all[j]);
	}
	all.resize(keep);
	std::sort(all.begin(), all.end());
	return all;
}

double ApplyTree(const BoostTree &tree, const vector<double> &features) {
	idx_t node_idx = 0;
	while (true) {
		auto &node = tree.nodes[node_idx];
		if (node.is_leaf) {
			return node.value;
		}
		auto value = features[node.feature];
		if (IsMissing(value)) {
			node_idx = node.default_left ? node.left : node.right;
		} else if (node.compare == SplitCompare::EQUAL) {
			node_idx = value == node.threshold ? node.right : node.left;
		} else {
			node_idx = value < node.threshold ? node.left : node.right;
		}
	}
}

vector<double> NormalizeWeights(const vector<double> &y, const vector<double> &weights_in) {
	vector<double> weights(y.size(), 1.0);
	if (!weights_in.empty()) {
		if (weights_in.size() != y.size()) {
			throw InvalidInputException("duckboost: sample weight count (%llu) must match row count (%llu)",
			                            (unsigned long long)weights_in.size(), (unsigned long long)y.size());
		}
		for (idx_t i = 0; i < y.size(); i++) {
			if (!std::isfinite(weights_in[i]) || weights_in[i] < 0) {
				throw InvalidInputException("duckboost: sample weights must be finite and >= 0");
			}
			weights[i] = weights_in[i];
		}
	}
	return weights;
}

void ApplyClassWeights(vector<double> &weights, const vector<double> &y, const TrainOptions &options, idx_t n_classes) {
	if (options.class_weight.empty()) {
		return;
	}
	vector<double> multipliers(n_classes, 1.0);
	auto lower = StringUtil::Lower(options.class_weight);
	if (lower == "balanced") {
		vector<double> counts(n_classes, 0);
		double total = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto label = static_cast<idx_t>(y[i]);
			if (label >= n_classes) {
				continue;
			}
			counts[label] += weights[i];
			total += weights[i];
		}
		for (idx_t c = 0; c < n_classes; c++) {
			if (counts[c] <= 0) {
				multipliers[c] = 0;
			} else {
				multipliers[c] = total / (static_cast<double>(n_classes) * counts[c]);
			}
		}
	} else {
		auto parts = StringUtil::Split(options.class_weight, ',');
		if (parts.size() != n_classes) {
			throw InvalidInputException("duckboost: class_weight list length (%llu) must match n_classes (%llu)",
			                            (unsigned long long)parts.size(), (unsigned long long)n_classes);
		}
		for (idx_t c = 0; c < n_classes; c++) {
			StringUtil::Trim(parts[c]);
			multipliers[c] = std::stod(parts[c]);
			if (!(multipliers[c] >= 0) || !std::isfinite(multipliers[c])) {
				throw InvalidInputException("duckboost: class_weight values must be finite and >= 0");
			}
		}
	}
	for (idx_t i = 0; i < y.size(); i++) {
		auto label = static_cast<idx_t>(y[i]);
		if (label < n_classes) {
			weights[i] *= multipliers[label];
		}
	}
}

idx_t InferNClasses(const vector<double> &y, const TrainOptions &options) {
	if (options.n_classes >= 2) {
		return options.n_classes;
	}
	double max_label = -1;
	for (auto v : y) {
		if (!std::isfinite(v) || v < 0 || std::floor(v) != v) {
			throw InvalidInputException(
			    "duckboost: multiclass labels must be finite non-negative integer class indices");
		}
		max_label = MaxValue(max_label, v);
	}
	auto inferred = static_cast<idx_t>(max_label) + 1;
	if (inferred < 2) {
		throw InvalidInputException("duckboost: multiclass train requires at least 2 classes");
	}
	return inferred;
}

unordered_set<idx_t> ResolveCatFeatures(const TrainOptions &options, idx_t n_features,
                                        const vector<string> &feature_names) {
	unordered_set<idx_t> cat_set;
	for (auto idx : options.cat_features) {
		if (idx >= n_features) {
			throw InvalidInputException("duckboost: cat_features index %llu out of range for %llu features",
			                            (unsigned long long)idx, (unsigned long long)n_features);
		}
		cat_set.insert(idx);
	}
	for (auto &token : options.cat_feature_tokens) {
		bool all_digits = !token.empty() && std::isdigit(static_cast<unsigned char>(token[0]));
		for (idx_t i = 1; all_digits && i < token.size(); i++) {
			if (!std::isdigit(static_cast<unsigned char>(token[i]))) {
				all_digits = false;
			}
		}
		if (all_digits) {
			continue; // already in cat_features
		}
		bool found = false;
		for (idx_t i = 0; i < feature_names.size(); i++) {
			if (StringUtil::Lower(feature_names[i]) == StringUtil::Lower(token)) {
				cat_set.insert(i);
				found = true;
				break;
			}
		}
		if (!found) {
			throw InvalidInputException("duckboost: cat_features name '%s' not found in feature_names", token);
		}
	}
	return cat_set;
}

double EvalMetricSingle(BoostTask task, const vector<double> &y, const vector<double> &prediction,
                        const vector<idx_t> &rows, const vector<double> &weights) {
	if (rows.empty()) {
		return std::numeric_limits<double>::infinity();
	}
	double weight_sum = 0;
	for (auto row : rows) {
		weight_sum += weights[row];
	}
	if (weight_sum <= 0) {
		return std::numeric_limits<double>::infinity();
	}
	if (task == BoostTask::BINARY) {
		double loss = 0;
		for (auto row : rows) {
			double p = 1.0 / (1.0 + std::exp(-prediction[row]));
			p = std::min(1.0 - 1e-15, std::max(1e-15, p));
			loss += weights[row] * -(y[row] * std::log(p) + (1.0 - y[row]) * std::log(1.0 - p));
		}
		return loss / weight_sum;
	}
	double sse = 0;
	for (auto row : rows) {
		auto err = prediction[row] - y[row];
		sse += weights[row] * err * err;
	}
	return std::sqrt(sse / weight_sum);
}

double EvalMetricMulti(const vector<double> &y, const vector<vector<double>> &prediction, idx_t n_classes,
                       const vector<idx_t> &rows, const vector<double> &weights) {
	if (rows.empty()) {
		return std::numeric_limits<double>::infinity();
	}
	double weight_sum = 0;
	double loss = 0;
	for (auto row : rows) {
		weight_sum += weights[row];
		double max_score = prediction[row][0];
		for (idx_t c = 1; c < n_classes; c++) {
			max_score = MaxValue(max_score, prediction[row][c]);
		}
		double sum_exp = 0;
		for (idx_t c = 0; c < n_classes; c++) {
			sum_exp += std::exp(prediction[row][c] - max_score);
		}
		auto label = static_cast<idx_t>(y[row]);
		auto p = std::exp(prediction[row][label] - max_score) / sum_exp;
		p = std::min(1.0 - 1e-15, std::max(1e-15, p));
		loss += weights[row] * -std::log(p);
	}
	if (weight_sum <= 0) {
		return std::numeric_limits<double>::infinity();
	}
	return loss / weight_sum;
}

vector<idx_t> SelectGrowRows(const vector<idx_t> &train_rows, double subsample, SimpleRng &rng) {
	if (subsample >= 1.0) {
		return train_rows;
	}
	auto bag = SampleRows(train_rows.size(), subsample, rng);
	vector<idx_t> grow_rows;
	grow_rows.reserve(bag.size());
	for (auto idx : bag) {
		grow_rows.push_back(train_rows[idx]);
	}
	std::sort(grow_rows.begin(), grow_rows.end());
	return grow_rows;
}

BoostModel TrainReference(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                          const vector<double> &weights_in) {
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch");
	}
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot train on empty dataset");
	}
	idx_t n_features = x[0].size();
	for (auto &row : x) {
		if (row.size() != n_features) {
			throw InvalidInputException("duckboost: jagged feature rows are not supported");
		}
	}

	BoostModel model;
	model.backend = BoostBackend::REFERENCE;
	model.task = options.task;
	model.learning_rate = options.learning_rate;
	model.n_features = n_features;
	model.feature_names = options.feature_names;
	if (model.feature_names.empty()) {
		for (idx_t i = 0; i < n_features; i++) {
			model.feature_names.push_back("f" + std::to_string(i));
		}
	} else if (model.feature_names.size() != n_features) {
		throw InvalidInputException("duckboost: feature_names count (%llu) must match feature width (%llu)",
		                            (unsigned long long)model.feature_names.size(), (unsigned long long)n_features);
	}

	auto weights = NormalizeWeights(y, weights_in);
	auto cat_set = ResolveCatFeatures(options, n_features, model.feature_names);

	SimpleRng rng(options.seed);
	vector<idx_t> all_rows(y.size());
	std::iota(all_rows.begin(), all_rows.end(), 0);
	vector<idx_t> train_rows = all_rows;
	vector<idx_t> valid_rows;
	if (options.validation_fraction > 0 && options.early_stopping_rounds > 0 && y.size() >= 4) {
		auto shuffled = all_rows;
		for (idx_t i = 0; i < shuffled.size(); i++) {
			idx_t j = i + rng.Bounded(shuffled.size() - i);
			std::swap(shuffled[i], shuffled[j]);
		}
		idx_t valid_n = MaxValue<idx_t>(
		    1, static_cast<idx_t>(std::floor(options.validation_fraction * static_cast<double>(y.size()))));
		valid_n = MinValue<idx_t>(valid_n, y.size() - 1);
		valid_rows.assign(shuffled.begin(), shuffled.begin() + valid_n);
		train_rows.assign(shuffled.begin() + valid_n, shuffled.end());
		std::sort(train_rows.begin(), train_rows.end());
		std::sort(valid_rows.begin(), valid_rows.end());
	}

	double best_valid = std::numeric_limits<double>::infinity();
	idx_t best_rounds = 0;
	idx_t rounds_since_improve = 0;

	if (options.task == BoostTask::MULTICLASS) {
		idx_t n_classes = InferNClasses(y, options);
		for (auto label : y) {
			if (static_cast<idx_t>(label) >= n_classes) {
				throw InvalidInputException("duckboost: multiclass label %g >= n_classes %llu", label,
				                            (unsigned long long)n_classes);
			}
		}
		model.n_classes = n_classes;
		ApplyClassWeights(weights, y, options, n_classes);

		vector<double> class_weight_sum(n_classes, 0);
		double total_w = 0;
		for (auto row : train_rows) {
			auto label = static_cast<idx_t>(y[row]);
			class_weight_sum[label] += weights[row];
			total_w += weights[row];
		}
		model.base_scores.assign(n_classes, 0);
		for (idx_t c = 0; c < n_classes; c++) {
			auto prior = class_weight_sum[c] / MaxValue(total_w, 1e-12);
			prior = std::min(1.0 - 1e-6, std::max(1e-6, prior));
			model.base_scores[c] = std::log(prior);
		}
		model.base_score = model.base_scores[0];

		vector<vector<double>> prediction(y.size(), model.base_scores);
		for (idx_t round = 0; round < options.n_estimators; round++) {
			auto grow_rows = SelectGrowRows(train_rows, options.subsample, rng);
			auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);
			for (idx_t c = 0; c < n_classes; c++) {
				vector<double> gradients(y.size());
				vector<double> hessians(y.size());
				for (idx_t i = 0; i < y.size(); i++) {
					double max_score = prediction[i][0];
					for (idx_t k = 1; k < n_classes; k++) {
						max_score = MaxValue(max_score, prediction[i][k]);
					}
					double sum_exp = 0;
					for (idx_t k = 0; k < n_classes; k++) {
						sum_exp += std::exp(prediction[i][k] - max_score);
					}
					double p = std::exp(prediction[i][c] - max_score) / sum_exp;
					double target = (static_cast<idx_t>(y[i]) == c) ? 1.0 : 0.0;
					gradients[i] = weights[i] * (p - target);
					hessians[i] = weights[i] * std::max(p * (1.0 - p), 1e-6);
				}
				BoostTree tree;
				BuildTree(tree, x, gradients, hessians, grow_rows, feature_subset, cat_set, 0, options);
				for (idx_t i = 0; i < y.size(); i++) {
					prediction[i][c] += options.learning_rate * ApplyTree(tree, x[i]);
				}
				model.trees.push_back(std::move(tree));
			}
			if (!valid_rows.empty()) {
				auto metric = EvalMetricMulti(y, prediction, n_classes, valid_rows, weights);
				if (metric < best_valid - 1e-12) {
					best_valid = metric;
					best_rounds = round + 1;
					rounds_since_improve = 0;
				} else {
					rounds_since_improve++;
					if (rounds_since_improve >= options.early_stopping_rounds) {
						break;
					}
				}
			}
		}
		if (!valid_rows.empty() && best_rounds > 0 && best_rounds * n_classes < model.trees.size()) {
			model.trees.resize(best_rounds * n_classes);
		}
		return model;
	}

	if (options.task == BoostTask::BINARY) {
		ApplyClassWeights(weights, y, options, 2);
	} else if (!options.class_weight.empty()) {
		throw InvalidInputException("duckboost: class_weight is only supported for binary and multiclass tasks");
	}

	vector<double> prediction(y.size(), 0);
	if (options.task == BoostTask::REGRESSION) {
		double sum = 0;
		double wsum = 0;
		for (auto row : train_rows) {
			sum += weights[row] * y[row];
			wsum += weights[row];
		}
		model.base_score = wsum > 0 ? sum / wsum : 0;
		std::fill(prediction.begin(), prediction.end(), model.base_score);
		model.n_classes = 1;
	} else {
		for (auto label : y) {
			if (!(label == 0.0 || label == 1.0)) {
				throw InvalidInputException("duckboost: binary task requires labels in {0, 1}");
			}
		}
		double pos = 0;
		double wsum = 0;
		for (auto row : train_rows) {
			pos += weights[row] * y[row];
			wsum += weights[row];
		}
		pos = wsum > 0 ? pos / wsum : 0.5;
		pos = std::min(1.0 - 1e-6, std::max(1e-6, pos));
		model.base_score = std::log(pos / (1.0 - pos));
		std::fill(prediction.begin(), prediction.end(), model.base_score);
		model.n_classes = 1;
	}

	for (idx_t round = 0; round < options.n_estimators; round++) {
		vector<double> gradients(y.size());
		vector<double> hessians(y.size());
		if (options.task == BoostTask::REGRESSION) {
			for (idx_t i = 0; i < y.size(); i++) {
				gradients[i] = weights[i] * (prediction[i] - y[i]);
				hessians[i] = weights[i];
			}
		} else {
			for (idx_t i = 0; i < y.size(); i++) {
				double p = 1.0 / (1.0 + std::exp(-prediction[i]));
				gradients[i] = weights[i] * (p - y[i]);
				hessians[i] = weights[i] * std::max(p * (1.0 - p), 1e-6);
			}
		}

		auto grow_rows = SelectGrowRows(train_rows, options.subsample, rng);
		auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);

		BoostTree tree;
		BuildTree(tree, x, gradients, hessians, grow_rows, feature_subset, cat_set, 0, options);
		for (idx_t i = 0; i < y.size(); i++) {
			prediction[i] += options.learning_rate * ApplyTree(tree, x[i]);
		}
		model.trees.push_back(std::move(tree));

		if (!valid_rows.empty()) {
			auto metric = EvalMetricSingle(options.task, y, prediction, valid_rows, weights);
			if (metric < best_valid - 1e-12) {
				best_valid = metric;
				best_rounds = model.trees.size();
				rounds_since_improve = 0;
			} else {
				rounds_since_improve++;
				if (rounds_since_improve >= options.early_stopping_rounds) {
					break;
				}
			}
		}
	}

	if (!valid_rows.empty() && best_rounds > 0 && best_rounds < model.trees.size()) {
		model.trees.resize(best_rounds);
	}
	return model;
}

} // namespace

BoostModel TrainModel(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options,
                      const vector<double> &weights) {
	if (options.backend == BoostBackend::REFERENCE) {
		return TrainReference(y, x, options, weights);
	}
	if (NativeTrainerCompiled(options.backend)) {
		return TrainNative(y, x, options);
	}
	throw NotImplementedException(
	    "duckboost: native training for backend '%s' is not linked in this build. "
	    "Use backend='reference' to train in-process, or duckboost_import() with a vendor dump. "
	    "Optional CMake flags: DUCKBOOST_WITH_XGBOOST / DUCKBOOST_WITH_LIGHTGBM / DUCKBOOST_WITH_CATBOOST "
	    "(add DUCKBOOST_NATIVE_STUB_ONLY=ON to compile stubs without vendor libs).",
	    BackendToString(options.backend));
}

double EvaluateModel(const BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const EvalOptions &options) {
	if (y.size() != x.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch during evaluate");
	}
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot evaluate on empty dataset");
	}
	auto metric = options.metric;
	if (metric.empty() || metric == "auto") {
		if (model.task == BoostTask::BINARY || model.task == BoostTask::MULTICLASS) {
			metric = "accuracy";
		} else {
			metric = "rmse";
		}
	}

	if (metric == "rmse") {
		double sse = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto err = model.Predict(x[i]) - y[i];
			sse += err * err;
		}
		return std::sqrt(sse / static_cast<double>(y.size()));
	}
	if (metric == "mae") {
		double sae = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			sae += std::fabs(model.Predict(x[i]) - y[i]);
		}
		return sae / static_cast<double>(y.size());
	}
	if (metric == "accuracy") {
		idx_t correct = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			double pred;
			if (model.task == BoostTask::MULTICLASS) {
				pred = model.Predict(x[i]);
			} else {
				pred = model.Predict(x[i]) >= 0.5 ? 1.0 : 0.0;
			}
			if (pred == y[i]) {
				correct++;
			}
		}
		return static_cast<double>(correct) / static_cast<double>(y.size());
	}
	if (metric == "logloss") {
		if (model.task == BoostTask::MULTICLASS) {
			double loss = 0;
			for (idx_t i = 0; i < y.size(); i++) {
				auto proba = model.PredictProba(x[i]);
				auto label = static_cast<idx_t>(y[i]);
				if (label >= proba.size()) {
					throw InvalidInputException("duckboost: multiclass label out of range during logloss");
				}
				auto p = std::min(1.0 - 1e-15, std::max(1e-15, proba[label]));
				loss += -std::log(p);
			}
			return loss / static_cast<double>(y.size());
		}
		double loss = 0;
		for (idx_t i = 0; i < y.size(); i++) {
			auto p = std::min(1.0 - 1e-15, std::max(1e-15, model.Predict(x[i])));
			loss += -(y[i] * std::log(p) + (1.0 - y[i]) * std::log(1.0 - p));
		}
		return loss / static_cast<double>(y.size());
	}
	throw InvalidInputException("duckboost: unknown metric '%s' (expected auto, rmse, mae, accuracy, logloss)",
	                            options.metric);
}

} // namespace duckboost
} // namespace duckdb
