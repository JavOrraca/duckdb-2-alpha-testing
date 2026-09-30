#include "duckboost/model.hpp"
#include "duckboost/native_train.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unordered_map.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

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

		// Gradient histogram: bin present values, then O(bins) prefix-sum split search.
		auto bin_uppers = CandidateThresholds(present_values, options.max_bins);
		if (bin_uppers.empty()) {
			continue;
		}
		const idx_t n_bins = bin_uppers.size() + 1;
		vector<GradStat> hist(n_bins);
		vector<idx_t> hist_count(n_bins, 0);
		for (auto row : present) {
			auto value = x[row][f];
			idx_t bin =
			    static_cast<idx_t>(std::upper_bound(bin_uppers.begin(), bin_uppers.end(), value) - bin_uppers.begin());
			hist[bin].Add(gradients[row], hessians[row]);
			hist_count[bin]++;
		}
		GradStat left_present;
		idx_t left_count = 0;
		for (idx_t b = 0; b + 1 < n_bins; b++) {
			left_present.Add(hist[b]);
			left_count += hist_count[b];
			auto right_present = parent.Without(left_present).Without(missing_stat);
			idx_t right_count = present.size() - left_count;
			if (left_count == 0 || right_count == 0) {
				continue;
			}
			auto threshold = bin_uppers[b];
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

idx_t EffectiveMaxLeaves(const TrainOptions &options) {
	if (options.max_leaves > 0) {
		return options.max_leaves;
	}
	return 31;
}

idx_t BuildTreeLeafWise(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &gradients,
                        const vector<double> &hessians, const vector<idx_t> &rows, const vector<idx_t> &feature_subset,
                        const unordered_set<idx_t> &cat_set, const TrainOptions &options) {
	struct Expandable {
		idx_t node_idx = 0;
		idx_t depth = 0;
		vector<idx_t> rows;
		SplitCandidate split;
	};

	auto parent = SumStats(gradients, hessians, rows);
	idx_t root = BuildLeaf(tree, LeafWeight(parent, options));
	if (rows.size() < 2 * options.min_samples_leaf) {
		return root;
	}

	vector<Expandable> frontier;
	auto enqueue = [&](idx_t node_idx, idx_t depth, vector<idx_t> node_rows) {
		if (depth >= options.max_depth || node_rows.size() < 2 * options.min_samples_leaf) {
			return;
		}
		auto split = FindBestSplit(x, gradients, hessians, node_rows, feature_subset, cat_set, options);
		if (!std::isfinite(split.gain) || split.gain <= 0) {
			return;
		}
		Expandable entry;
		entry.node_idx = node_idx;
		entry.depth = depth;
		entry.rows = std::move(node_rows);
		entry.split = split;
		frontier.push_back(std::move(entry));
	};
	enqueue(root, 0, rows);

	idx_t leaf_count = 1;
	const idx_t max_leaves = EffectiveMaxLeaves(options);
	while (leaf_count < max_leaves && !frontier.empty()) {
		idx_t best_i = 0;
		for (idx_t i = 1; i < frontier.size(); i++) {
			if (frontier[i].split.gain > frontier[best_i].split.gain) {
				best_i = i;
			}
		}
		auto cur = std::move(frontier[best_i]);
		frontier.erase(frontier.begin() + static_cast<int64_t>(best_i));

		vector<idx_t> left_rows;
		vector<idx_t> right_rows;
		PartitionRows(x, cur.rows, cur.split, left_rows, right_rows);
		if (left_rows.empty() || right_rows.empty()) {
			continue;
		}

		auto left_stat = SumStats(gradients, hessians, left_rows);
		auto right_stat = SumStats(gradients, hessians, right_rows);
		idx_t left_idx = BuildLeaf(tree, LeafWeight(left_stat, options));
		idx_t right_idx = BuildLeaf(tree, LeafWeight(right_stat, options));

		auto &node = tree.nodes[cur.node_idx];
		node.is_leaf = false;
		node.value = 0;
		node.feature = cur.split.feature;
		node.threshold = cur.split.threshold;
		node.default_left = cur.split.default_left;
		node.compare = cur.split.compare;
		node.left = left_idx;
		node.right = right_idx;

		leaf_count++;
		enqueue(left_idx, cur.depth + 1, std::move(left_rows));
		enqueue(right_idx, cur.depth + 1, std::move(right_rows));
	}
	return root;
}

idx_t GrowTree(BoostTree &tree, const vector<vector<double>> &x, const vector<double> &gradients,
               const vector<double> &hessians, const vector<idx_t> &rows, const vector<idx_t> &feature_subset,
               const unordered_set<idx_t> &cat_set, const TrainOptions &options) {
	if (options.grow_policy == GrowPolicy::LEAF) {
		return BuildTreeLeafWise(tree, x, gradients, hessians, rows, feature_subset, cat_set, options);
	}
	return BuildTree(tree, x, gradients, hessians, rows, feature_subset, cat_set, 0, options);
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

double LinkedPrediction(BoostObjective objective, double raw) {
	if (objective == BoostObjective::POISSON) {
		return std::exp(raw);
	}
	if (objective == BoostObjective::LOGISTIC) {
		return 1.0 / (1.0 + std::exp(-raw));
	}
	return raw;
}

double EvalMetricSingle(BoostObjective objective, const vector<double> &y, const vector<double> &prediction,
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
	if (objective == BoostObjective::LOGISTIC) {
		double loss = 0;
		for (auto row : rows) {
			double p = LinkedPrediction(objective, prediction[row]);
			p = std::min(1.0 - 1e-15, std::max(1e-15, p));
			loss += weights[row] * -(y[row] * std::log(p) + (1.0 - y[row]) * std::log(1.0 - p));
		}
		return loss / weight_sum;
	}
	if (objective == BoostObjective::QUANTILE) {
		// Pinball loss uses alpha from the model path; early-stop uses MAE on linked preds as a proxy.
		double sae = 0;
		for (auto row : rows) {
			sae += weights[row] * std::fabs(LinkedPrediction(objective, prediction[row]) - y[row]);
		}
		return sae / weight_sum;
	}
	double sse = 0;
	for (auto row : rows) {
		auto err = LinkedPrediction(objective, prediction[row]) - y[row];
		sse += weights[row] * err * err;
	}
	return std::sqrt(sse / weight_sum);
}

void FillGradients(BoostObjective objective, const vector<double> &y, const vector<double> &prediction,
                   const vector<double> &weights, double huber_delta, double quantile_alpha, vector<double> &gradients,
                   vector<double> &hessians) {
	const idx_t n = y.size();
	gradients.resize(n);
	hessians.resize(n);
	switch (objective) {
	case BoostObjective::POISSON:
		for (idx_t i = 0; i < n; i++) {
			double mu = std::exp(prediction[i]);
			mu = std::min(1e12, std::max(1e-12, mu));
			gradients[i] = weights[i] * (mu - y[i]);
			hessians[i] = weights[i] * std::max(mu, 1e-6);
		}
		break;
	case BoostObjective::HUBER:
		for (idx_t i = 0; i < n; i++) {
			double err = prediction[i] - y[i];
			if (std::fabs(err) <= huber_delta) {
				gradients[i] = weights[i] * err;
				hessians[i] = weights[i];
			} else {
				gradients[i] = weights[i] * huber_delta * (err > 0 ? 1.0 : -1.0);
				hessians[i] = weights[i] * 1e-6;
			}
		}
		break;
	case BoostObjective::QUANTILE:
		for (idx_t i = 0; i < n; i++) {
			// Pinball ∂L/∂pred: -α if y >= pred else (1-α).
			gradients[i] = weights[i] * ((y[i] >= prediction[i]) ? -quantile_alpha : (1.0 - quantile_alpha));
			hessians[i] = weights[i];
		}
		break;
	case BoostObjective::LOGISTIC:
		for (idx_t i = 0; i < n; i++) {
			double p = 1.0 / (1.0 + std::exp(-prediction[i]));
			gradients[i] = weights[i] * (p - y[i]);
			hessians[i] = weights[i] * std::max(p * (1.0 - p), 1e-6);
		}
		break;
	case BoostObjective::SQUAREDERROR:
	default:
		for (idx_t i = 0; i < n; i++) {
			gradients[i] = weights[i] * (prediction[i] - y[i]);
			hessians[i] = weights[i];
		}
		break;
	}
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

double RankGain(double relevance) {
	return std::pow(2.0, relevance) - 1.0;
}

double RankDiscount(idx_t pos) {
	return 1.0 / std::log2(static_cast<double>(pos) + 2.0);
}

struct RankItem {
	idx_t row;
	double relevance;
	double score;
};

double IdealDCG(vector<double> relevances, idx_t k) {
	std::sort(relevances.begin(), relevances.end(), std::greater<double>());
	idx_t n = k == 0 ? relevances.size() : MinValue<idx_t>(k, relevances.size());
	double dcg = 0;
	for (idx_t i = 0; i < n; i++) {
		dcg += RankGain(relevances[i]) * RankDiscount(i);
	}
	return dcg;
}

double NDCGForGroup(vector<RankItem> items, idx_t k) {
	if (items.empty()) {
		return 0;
	}
	vector<double> relevances;
	relevances.reserve(items.size());
	for (auto &item : items) {
		relevances.push_back(item.relevance);
	}
	double idcg = IdealDCG(relevances, k);
	if (idcg <= 0) {
		return 0;
	}
	std::sort(items.begin(), items.end(), [](const RankItem &a, const RankItem &b) {
		if (a.score != b.score) {
			return a.score > b.score;
		}
		return a.row < b.row;
	});
	idx_t n = k == 0 ? items.size() : MinValue<idx_t>(k, items.size());
	double dcg = 0;
	for (idx_t i = 0; i < n; i++) {
		dcg += RankGain(items[i].relevance) * RankDiscount(i);
	}
	return dcg / idcg;
}

double AveragePrecisionForGroup(vector<RankItem> items) {
	if (items.empty()) {
		return 0;
	}
	std::sort(items.begin(), items.end(), [](const RankItem &a, const RankItem &b) {
		if (a.score != b.score) {
			return a.score > b.score;
		}
		return a.row < b.row;
	});
	double hits = 0;
	double sum_prec = 0;
	for (idx_t i = 0; i < items.size(); i++) {
		if (items[i].relevance > 0) {
			hits += 1;
			sum_prec += hits / static_cast<double>(i + 1);
		}
	}
	return hits <= 0 ? 0 : sum_prec / hits;
}

unordered_map<int64_t, vector<idx_t>> GroupRowsById(const vector<idx_t> &rows, const vector<int64_t> &groups) {
	unordered_map<int64_t, vector<idx_t>> by_group;
	for (auto row : rows) {
		by_group[groups[row]].push_back(row);
	}
	return by_group;
}

void SplitRowsByGroups(const vector<int64_t> &groups, double validation_fraction, SimpleRng &rng,
                       vector<idx_t> &train_rows, vector<idx_t> &valid_rows) {
	unordered_map<int64_t, vector<idx_t>> by_group;
	for (idx_t i = 0; i < groups.size(); i++) {
		by_group[groups[i]].push_back(i);
	}
	vector<int64_t> group_ids;
	group_ids.reserve(by_group.size());
	for (auto &entry : by_group) {
		group_ids.push_back(entry.first);
	}
	for (idx_t i = 0; i < group_ids.size(); i++) {
		idx_t j = i + rng.Bounded(group_ids.size() - i);
		std::swap(group_ids[i], group_ids[j]);
	}
	idx_t valid_g =
	    MaxValue<idx_t>(1, static_cast<idx_t>(std::floor(validation_fraction * static_cast<double>(group_ids.size()))));
	valid_g = MinValue<idx_t>(valid_g, group_ids.size() - 1);
	train_rows.clear();
	valid_rows.clear();
	for (idx_t g = 0; g < group_ids.size(); g++) {
		auto &rows = by_group[group_ids[g]];
		if (g < valid_g) {
			valid_rows.insert(valid_rows.end(), rows.begin(), rows.end());
		} else {
			train_rows.insert(train_rows.end(), rows.begin(), rows.end());
		}
	}
	std::sort(train_rows.begin(), train_rows.end());
	std::sort(valid_rows.begin(), valid_rows.end());
}

double EvalRankingMetric(const vector<double> &y, const vector<double> &prediction, const vector<idx_t> &rows,
                         const vector<int64_t> &groups, idx_t ndcg_at, bool use_map) {
	auto by_group = GroupRowsById(rows, groups);
	if (by_group.empty()) {
		return std::numeric_limits<double>::infinity();
	}
	double sum = 0;
	for (auto &entry : by_group) {
		vector<RankItem> items;
		items.reserve(entry.second.size());
		for (auto row : entry.second) {
			items.push_back({row, y[row], prediction[row]});
		}
		sum += use_map ? AveragePrecisionForGroup(std::move(items)) : NDCGForGroup(std::move(items), ndcg_at);
	}
	// Return loss-style metric (lower is better) for early stopping.
	return 1.0 - sum / static_cast<double>(by_group.size());
}

void FillLambdaRankGradients(BoostObjective objective, const vector<double> &y, const vector<double> &prediction,
                             const vector<double> &weights, const vector<int64_t> &groups, const vector<idx_t> &rows,
                             idx_t ndcg_at, vector<double> &gradients, vector<double> &hessians) {
	gradients.assign(y.size(), 0);
	hessians.assign(y.size(), 0);
	auto by_group = GroupRowsById(rows, groups);
	const bool use_delta_ndcg = objective == BoostObjective::LAMBDARANK;

	for (auto &entry : by_group) {
		auto &group_rows = entry.second;
		if (group_rows.size() < 2) {
			continue;
		}
		vector<RankItem> items;
		items.reserve(group_rows.size());
		for (auto row : group_rows) {
			items.push_back({row, y[row], prediction[row]});
		}
		std::sort(items.begin(), items.end(), [](const RankItem &a, const RankItem &b) {
			if (a.score != b.score) {
				return a.score > b.score;
			}
			return a.row < b.row;
		});
		unordered_map<idx_t, idx_t> position;
		for (idx_t i = 0; i < items.size(); i++) {
			position[items[i].row] = i;
		}
		vector<double> relevances;
		relevances.reserve(items.size());
		for (auto &item : items) {
			relevances.push_back(item.relevance);
		}
		double idcg = IdealDCG(relevances, ndcg_at);
		if (idcg <= 0) {
			idcg = 1.0;
		}

		for (idx_t i = 0; i < group_rows.size(); i++) {
			for (idx_t j = i + 1; j < group_rows.size(); j++) {
				idx_t ri = group_rows[i];
				idx_t rj = group_rows[j];
				if (y[ri] == y[rj]) {
					continue;
				}
				idx_t high = y[ri] > y[rj] ? ri : rj;
				idx_t low = y[ri] > y[rj] ? rj : ri;
				double score_diff = prediction[high] - prediction[low];
				// Pairwise logistic: push high-label scores up via leaf = -G/H.
				double rho = 1.0 / (1.0 + std::exp(score_diff));
				double weight = std::sqrt(weights[high] * weights[low]);
				double delta = 1.0;
				if (use_delta_ndcg) {
					idx_t pos_high = position[high];
					idx_t pos_low = position[low];
					double gain_high = RankGain(y[high]);
					double gain_low = RankGain(y[low]);
					delta = std::fabs((gain_high - gain_low) * (RankDiscount(pos_high) - RankDiscount(pos_low))) / idcg;
					if (!(delta > 0)) {
						delta = 1e-6;
					}
				}
				// g_high = -ρ·Δ so leaf = -G/H raises scores for higher-label docs.
				double lambda = weight * rho * delta;
				double hess = weight * std::max(rho * (1.0 - rho), 1e-3) * delta;
				gradients[high] -= lambda;
				gradients[low] += lambda;
				hessians[high] += hess;
				hessians[low] += hess;
			}
		}
	}
	for (idx_t i = 0; i < hessians.size(); i++) {
		if (hessians[i] <= 0) {
			hessians[i] = 1e-3;
		}
	}
}

vector<string> ParseCtrTypes(const string &raw) {
	vector<string> types;
	auto parts = StringUtil::Split(raw, ',');
	for (auto &part : parts) {
		StringUtil::Trim(part);
		if (part.empty()) {
			continue;
		}
		auto lower = StringUtil::Lower(part);
		if (lower == "borders" || lower == "border" || lower == "buckets") {
			types.push_back("Borders");
		} else if (lower == "counter" || lower == "featurefreq" || lower == "freq") {
			types.push_back("Counter");
		} else {
			throw InvalidInputException("duckboost: unknown ctr_types token '%s' (expected Borders or Counter)", part);
		}
	}
	return types;
}

void FillCtrHashTables(CtrFeatureSpec &spec, const vector<double> &y, const vector<vector<double>> &x,
                       const vector<idx_t> &train_rows, const vector<double> &weights, idx_t cat_idx, bool is_borders) {
	unordered_map<uint64_t, int64_t> primary;
	unordered_map<uint64_t, int64_t> secondary;
	double weight_sum = 0;
	for (auto row : train_rows) {
		auto key = FeatureHashU64(x[row][cat_idx]);
		weight_sum += weights[row];
		if (is_borders) {
			if (y[row] >= 0.5) {
				secondary[key] += 1;
			} else {
				primary[key] += 1;
			}
		} else {
			primary[key] += 1;
		}
	}
	spec.hash_keys.clear();
	spec.hash_values.clear();
	spec.hash_values_alt.clear();
	spec.hash_keys.reserve(primary.size() + secondary.size());
	if (is_borders) {
		unordered_map<uint64_t, bool> seen;
		for (auto &entry : primary) {
			spec.hash_keys.push_back(entry.first);
			spec.hash_values.push_back(entry.second);
			spec.hash_values_alt.push_back(secondary[entry.first]);
			seen[entry.first] = true;
		}
		for (auto &entry : secondary) {
			if (seen[entry.first]) {
				continue;
			}
			spec.hash_keys.push_back(entry.first);
			spec.hash_values.push_back(0);
			spec.hash_values_alt.push_back(entry.second);
		}
	} else {
		for (auto &entry : primary) {
			spec.hash_keys.push_back(entry.first);
			spec.hash_values.push_back(entry.second);
		}
		spec.counter_denominator = static_cast<int64_t>(std::llround(weight_sum > 0 ? weight_sum : train_rows.size()));
	}
}

double CtrValueForRow(const CtrFeatureSpec &spec, const vector<double> &raw_row,
                      const unordered_map<uint64_t, idx_t> &lookup, bool is_borders, CtrTargetLeakage leakage,
                      double label) {
	auto key = CombineCtrHash(spec, raw_row);
	int64_t primary = 0;
	int64_t secondary = 0;
	auto it = lookup.find(key);
	if (it != lookup.end()) {
		primary = spec.hash_values[it->second];
		if (it->second < spec.hash_values_alt.size()) {
			secondary = spec.hash_values_alt[it->second];
		}
	}
	if (is_borders && leakage == CtrTargetLeakage::LEAVE_ONE_OUT) {
		if (label >= 0.5) {
			secondary = MaxValue<int64_t>(0, secondary - 1);
		} else {
			primary = MaxValue<int64_t>(0, primary - 1);
		}
	}
	return CtrValueFromCounts(spec, primary, secondary);
}

vector<vector<double>> ExpandFeaturesWithCtr(BoostModel &model, const vector<double> &y,
                                             const vector<vector<double>> &x, const vector<idx_t> &train_rows,
                                             const vector<double> &weights, const unordered_set<idx_t> &cat_set,
                                             const TrainOptions &options) {
	auto types = ParseCtrTypes(options.ctr_types);
	if (types.empty()) {
		model.n_raw_features = x[0].size();
		return x;
	}
	if (cat_set.empty()) {
		throw InvalidInputException("duckboost: ctr_types requires cat_features");
	}
	const bool need_binary = std::any_of(types.begin(), types.end(), [](const string &t) { return t == "Borders"; });
	if (need_binary) {
		if (options.task != BoostTask::BINARY &&
		    ResolveObjective(options.task, options.objective) != BoostObjective::LOGISTIC) {
			throw InvalidInputException("duckboost: Borders CTR requires task 'binary'");
		}
		for (auto row : train_rows) {
			if (!(y[row] == 0.0 || y[row] == 1.0)) {
				throw InvalidInputException("duckboost: Borders CTR requires labels in {0, 1}");
			}
		}
	}

	vector<idx_t> cats(cat_set.begin(), cat_set.end());
	std::sort(cats.begin(), cats.end());

	const idx_t n_raw = x[0].size();
	model.n_raw_features = n_raw;
	model.n_features = n_raw;
	model.ctr_features.clear();

	vector<vector<double>> expanded = x;
	for (auto &row : expanded) {
		row.resize(n_raw);
	}

	for (auto cat_idx : cats) {
		for (auto &type : types) {
			CtrFeatureSpec spec;
			spec.feature_index = model.n_features++;
			spec.ctr_type = type;
			spec.prior_numerator = options.ctr_prior_numerator;
			spec.prior_denominator = options.ctr_prior_denominator;
			spec.scale = type == "Counter" && options.ctr_scale == 1.0 ? 15.0 : options.ctr_scale;
			spec.shift = options.ctr_shift;
			CtrCombineElement element;
			element.kind = CtrElementKind::CAT_FEATURE_VALUE;
			element.feature_index = cat_idx;
			spec.elements.push_back(element);
			spec.cat_feature_indices.push_back(cat_idx);

			const bool is_borders = type == "Borders";
			if (options.ctr_target_leakage == CtrTargetLeakage::EXPANDING && is_borders) {
				// Expanding: fill row values from prefix counts; store full-train tables for predict.
				unordered_map<uint64_t, int64_t> fail;
				unordered_map<uint64_t, int64_t> success;
				vector<idx_t> order = train_rows;
				std::sort(order.begin(), order.end());
				unordered_map<idx_t, double> train_ctr;
				for (auto row : order) {
					auto key = FeatureHashU64(x[row][cat_idx]);
					train_ctr[row] = CtrValueFromCounts(spec, fail[key], success[key]);
					if (y[row] >= 0.5) {
						success[key] += 1;
					} else {
						fail[key] += 1;
					}
				}
				FillCtrHashTables(spec, y, x, train_rows, weights, cat_idx, true);
				unordered_map<uint64_t, idx_t> lookup;
				for (idx_t i = 0; i < spec.hash_keys.size(); i++) {
					lookup[spec.hash_keys[i]] = i;
				}
				for (auto &row : expanded) {
					row.resize(model.n_features, 0);
				}
				for (idx_t i = 0; i < y.size(); i++) {
					auto it = train_ctr.find(i);
					if (it != train_ctr.end()) {
						expanded[i][spec.feature_index] = it->second;
					} else {
						expanded[i][spec.feature_index] =
						    CtrValueForRow(spec, x[i], lookup, true, CtrTargetLeakage::NONE, y[i]);
					}
				}
			} else {
				FillCtrHashTables(spec, y, x, train_rows, weights, cat_idx, is_borders);
				unordered_map<uint64_t, idx_t> lookup;
				for (idx_t i = 0; i < spec.hash_keys.size(); i++) {
					lookup[spec.hash_keys[i]] = i;
				}
				for (auto &row : expanded) {
					row.resize(model.n_features, 0);
				}
				for (idx_t i = 0; i < y.size(); i++) {
					auto leakage = is_borders ? options.ctr_target_leakage : CtrTargetLeakage::NONE;
					// Valid / non-train rows use full train tables (no LOO).
					bool in_train = false;
					// train_rows is sorted; binary search
					in_train = std::binary_search(train_rows.begin(), train_rows.end(), i);
					if (!in_train) {
						leakage = CtrTargetLeakage::NONE;
					}
					expanded[i][spec.feature_index] = CtrValueForRow(spec, x[i], lookup, is_borders, leakage, y[i]);
				}
			}
			model.ctr_features.push_back(std::move(spec));
		}
	}

	// Append synthetic feature names for tree debugging / SQL.
	while (model.feature_names.size() < model.n_features) {
		model.feature_names.push_back("_duckboost_ctr_" + std::to_string(model.feature_names.size()));
	}
	return expanded;
}

BoostModel TrainReference(const vector<double> &y, const vector<vector<double>> &x_in, const TrainOptions &options,
                          const vector<double> &weights_in, const vector<int64_t> &groups) {
	if (y.size() != x_in.size()) {
		throw InvalidInputException("duckboost: y/x row count mismatch");
	}
	if (y.empty()) {
		throw InvalidInputException("duckboost: cannot train on empty dataset");
	}
	idx_t n_raw_features = x_in[0].size();
	for (auto &row : x_in) {
		if (row.size() != n_raw_features) {
			throw InvalidInputException("duckboost: jagged feature rows are not supported");
		}
	}

	auto objective = ResolveObjective(options.task, options.objective);

	BoostModel model;
	model.backend = BoostBackend::REFERENCE;
	model.task = options.task;
	model.objective = options.objective == BoostObjective::AUTO ? BoostObjective::AUTO : objective;
	model.huber_delta = options.huber_delta;
	model.quantile_alpha = options.quantile_alpha;
	model.learning_rate = options.learning_rate;
	model.n_features = n_raw_features;
	model.n_raw_features = n_raw_features;
	model.feature_names = options.feature_names;
	if (model.feature_names.empty()) {
		for (idx_t i = 0; i < n_raw_features; i++) {
			model.feature_names.push_back("f" + std::to_string(i));
		}
	} else if (model.feature_names.size() != n_raw_features) {
		throw InvalidInputException("duckboost: feature_names count (%llu) must match feature width (%llu)",
		                            (unsigned long long)model.feature_names.size(), (unsigned long long)n_raw_features);
	}

	auto weights = NormalizeWeights(y, weights_in);
	auto cat_set = ResolveCatFeatures(options, n_raw_features, model.feature_names);

	const bool is_ranking = objective == BoostObjective::LAMBDARANK || objective == BoostObjective::PAIRWISE;
	if (is_ranking) {
		if (groups.size() != y.size()) {
			throw InvalidInputException("duckboost: ranking requires a group id for every row");
		}
		for (auto label : y) {
			if (!std::isfinite(label) || label < 0) {
				throw InvalidInputException("duckboost: ranking labels must be finite and >= 0");
			}
		}
		model.ndcg_at = options.ndcg_at;
	} else if (!groups.empty() && groups.size() != y.size()) {
		throw InvalidInputException("duckboost: group count (%llu) must match row count (%llu)",
		                            (unsigned long long)groups.size(), (unsigned long long)y.size());
	}

	SimpleRng rng(options.seed);
	vector<idx_t> all_rows(y.size());
	std::iota(all_rows.begin(), all_rows.end(), 0);
	vector<idx_t> train_rows = all_rows;
	vector<idx_t> valid_rows;
	if (options.validation_fraction > 0 && options.early_stopping_rounds > 0 && y.size() >= 4) {
		if (is_ranking) {
			unordered_map<int64_t, idx_t> group_count;
			for (auto g : groups) {
				group_count[g]++;
			}
			if (group_count.size() >= 2) {
				SplitRowsByGroups(groups, options.validation_fraction, rng, train_rows, valid_rows);
			}
		} else {
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
	}

	auto x = ExpandFeaturesWithCtr(model, y, x_in, train_rows, weights, cat_set, options);
	idx_t n_features = model.n_features;

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
				GrowTree(tree, x, gradients, hessians, grow_rows, feature_subset, cat_set, options);
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

	if (is_ranking) {
		if (!options.class_weight.empty()) {
			throw InvalidInputException("duckboost: class_weight is not supported for ranking");
		}
		model.base_score = 0;
		model.n_classes = 1;
		vector<double> prediction(y.size(), 0);
		for (idx_t round = 0; round < options.n_estimators; round++) {
			vector<double> gradients;
			vector<double> hessians;
			FillLambdaRankGradients(objective, y, prediction, weights, groups, train_rows, options.ndcg_at, gradients,
			                        hessians);
			auto grow_rows = SelectGrowRows(train_rows, options.subsample, rng);
			auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);
			BoostTree tree;
			GrowTree(tree, x, gradients, hessians, grow_rows, feature_subset, cat_set, options);
			for (idx_t i = 0; i < y.size(); i++) {
				prediction[i] += options.learning_rate * ApplyTree(tree, x[i]);
			}
			model.trees.push_back(std::move(tree));
			if (!valid_rows.empty()) {
				auto metric = EvalRankingMetric(y, prediction, valid_rows, groups, options.ndcg_at, /*use_map=*/false);
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

	if (options.task == BoostTask::BINARY) {
		ApplyClassWeights(weights, y, options, 2);
	} else if (!options.class_weight.empty()) {
		throw InvalidInputException("duckboost: class_weight is only supported for binary and multiclass tasks");
	}

	vector<double> prediction(y.size(), 0);
	if (objective == BoostObjective::LOGISTIC) {
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
	} else if (objective == BoostObjective::POISSON) {
		for (auto label : y) {
			if (!std::isfinite(label) || label < 0) {
				throw InvalidInputException("duckboost: poisson objective requires non-negative labels");
			}
		}
		double sum = 0;
		double wsum = 0;
		for (auto row : train_rows) {
			sum += weights[row] * y[row];
			wsum += weights[row];
		}
		double mean = wsum > 0 ? sum / wsum : 1.0;
		mean = std::max(mean, 1e-6);
		model.base_score = std::log(mean);
		std::fill(prediction.begin(), prediction.end(), model.base_score);
		model.n_classes = 1;
	} else {
		// Squared error / Huber / quantile: initialize at weighted mean (quantile uses mean as a stable start).
		double sum = 0;
		double wsum = 0;
		for (auto row : train_rows) {
			sum += weights[row] * y[row];
			wsum += weights[row];
		}
		model.base_score = wsum > 0 ? sum / wsum : 0;
		std::fill(prediction.begin(), prediction.end(), model.base_score);
		model.n_classes = 1;
	}

	for (idx_t round = 0; round < options.n_estimators; round++) {
		vector<double> gradients;
		vector<double> hessians;
		FillGradients(objective, y, prediction, weights, options.huber_delta, options.quantile_alpha, gradients,
		              hessians);

		auto grow_rows = SelectGrowRows(train_rows, options.subsample, rng);
		auto feature_subset = SampleFeatures(n_features, options.colsample_bytree, rng);

		BoostTree tree;
		GrowTree(tree, x, gradients, hessians, grow_rows, feature_subset, cat_set, options);
		for (idx_t i = 0; i < y.size(); i++) {
			prediction[i] += options.learning_rate * ApplyTree(tree, x[i]);
		}
		model.trees.push_back(std::move(tree));

		if (!valid_rows.empty()) {
			auto metric = EvalMetricSingle(objective, y, prediction, valid_rows, weights);
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
                      const vector<double> &weights, const vector<int64_t> &groups) {
	if (options.backend == BoostBackend::REFERENCE) {
		return TrainReference(y, x, options, weights, groups);
	}
	if (NativeTrainerCompiled(options.backend)) {
		return TrainNative(y, x, options, weights, groups);
	}
	throw NotImplementedException(
	    "duckboost: native training for backend '%s' is not linked in this build. "
	    "Use backend='reference' to train in-process, or duckboost_import() with a vendor dump. "
	    "Optional CMake flags: DUCKBOOST_WITH_XGBOOST / DUCKBOOST_WITH_LIGHTGBM / DUCKBOOST_WITH_CATBOOST "
	    "(add DUCKBOOST_NATIVE_STUB_ONLY=ON to compile stubs without vendor libs).",
	    BackendToString(options.backend));
}

double EvaluateModel(const BoostModel &model, const vector<double> &y, const vector<vector<double>> &x,
                     const EvalOptions &options, const vector<int64_t> &groups) {
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
		} else if (model.task == BoostTask::RANKING) {
			metric = "ndcg";
		} else {
			metric = "rmse";
		}
	}

	if (metric == "ndcg" || metric == "map") {
		if (groups.size() != y.size()) {
			throw InvalidInputException("duckboost: metric '%s' requires a group id for every row", metric);
		}
		idx_t k = options.ndcg_at > 0 ? options.ndcg_at : model.ndcg_at;
		vector<double> scores(y.size());
		for (idx_t i = 0; i < y.size(); i++) {
			scores[i] = model.Predict(x[i]);
		}
		vector<idx_t> all_rows(y.size());
		std::iota(all_rows.begin(), all_rows.end(), 0);
		// EvalRankingMetric returns 1 - mean metric; invert for user-facing NDCG/MAP.
		return 1.0 - EvalRankingMetric(y, scores, all_rows, groups, k, metric == "map");
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
	throw InvalidInputException(
	    "duckboost: unknown metric '%s' (expected auto, rmse, mae, accuracy, logloss, ndcg, map)", options.metric);
}

} // namespace duckboost
} // namespace duckdb
