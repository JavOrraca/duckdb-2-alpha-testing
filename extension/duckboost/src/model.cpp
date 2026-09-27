#include "duckboost/model.hpp"
#include "duckboost/json_util.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <cmath>
#include <sstream>

namespace duckdb {
namespace duckboost {

namespace {

string QuoteIdent(const string &name) {
	return "\"" + StringUtil::Replace(name, "\"", "\"\"") + "\"";
}

string TreeToSQL(const BoostTree &tree, const vector<string> &feature_columns, idx_t node_idx) {
	if (node_idx >= tree.nodes.size()) {
		throw InvalidInputException("duckboost: invalid tree node index in SQL export");
	}
	auto &node = tree.nodes[node_idx];
	if (node.is_leaf) {
		return FormatDouble(node.value);
	}
	if (node.feature >= feature_columns.size()) {
		throw InvalidInputException("duckboost: feature index out of range in SQL export");
	}
	auto feature = QuoteIdent(feature_columns[node.feature]);
	auto left = TreeToSQL(tree, feature_columns, node.left);
	auto right = TreeToSQL(tree, feature_columns, node.right);
	return "CASE WHEN " + feature + " < " + FormatDouble(node.threshold) + " THEN " + left + " ELSE " + right + " END";
}

double Sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

} // namespace

string BackendToString(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::REFERENCE:
		return "reference";
	case BoostBackend::XGBOOST:
		return "xgboost";
	case BoostBackend::LIGHTGBM:
		return "lightgbm";
	case BoostBackend::CATBOOST:
		return "catboost";
	default:
		return "unknown";
	}
}

BoostBackend BackendFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "reference" || lower == "duckboost" || lower == "gbdt") {
		return BoostBackend::REFERENCE;
	}
	if (lower == "xgboost" || lower == "xgb") {
		return BoostBackend::XGBOOST;
	}
	if (lower == "lightgbm" || lower == "lgbm" || lower == "lgb") {
		return BoostBackend::LIGHTGBM;
	}
	if (lower == "catboost" || lower == "cb") {
		return BoostBackend::CATBOOST;
	}
	throw InvalidInputException(
	    "duckboost: unknown backend '%s' (expected reference, xgboost, lightgbm, or catboost)", name);
}

string TaskToString(BoostTask task) {
	switch (task) {
	case BoostTask::BINARY:
		return "binary";
	case BoostTask::MULTICLASS:
		return "multiclass";
	case BoostTask::REGRESSION:
	default:
		return "regression";
	}
}

BoostTask TaskFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "regression" || lower == "regressor" || lower == "mse") {
		return BoostTask::REGRESSION;
	}
	if (lower == "binary" || lower == "classification" || lower == "logistic" || lower == "binomial") {
		return BoostTask::BINARY;
	}
	if (lower == "multiclass" || lower == "multi_class" || lower == "multi-class" || lower == "softmax") {
		return BoostTask::MULTICLASS;
	}
	throw InvalidInputException("duckboost: unknown task '%s' (expected regression, binary, or multiclass)", name);
}

bool BackendTrainingSupported(BoostBackend backend) {
	return backend == BoostBackend::REFERENCE;
}

string BackendCapabilityNote(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::REFERENCE:
		return "in-process reference GBDT (train/predict/evaluate/to_sql)";
	case BoostBackend::XGBOOST:
		return "import dump_model JSON via duckboost_import; native train requires DUCKBOOST_WITH_XGBOOST";
	case BoostBackend::LIGHTGBM:
		return "import save_model text via duckboost_import; native train requires DUCKBOOST_WITH_LIGHTGBM";
	case BoostBackend::CATBOOST:
		return "import save_model JSON via duckboost_import; native train requires DUCKBOOST_WITH_CATBOOST";
	default:
		return "unknown";
	}
}

TrainOptions TrainOptions::FromMap(const unordered_map<string, string> &options) {
	TrainOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto &value = entry.second;
		if (key == "backend") {
			result.backend = BackendFromString(value);
		} else if (key == "task" || key == "objective") {
			result.task = TaskFromString(value);
		} else if (key == "n_estimators" || key == "num_boost_round" || key == "iterations") {
			result.n_estimators = static_cast<idx_t>(std::stoull(value));
		} else if (key == "max_depth" || key == "depth") {
			result.max_depth = static_cast<idx_t>(std::stoull(value));
		} else if (key == "learning_rate" || key == "eta" || key == "lr") {
			result.learning_rate = std::stod(value);
		} else if (key == "min_samples_leaf" || key == "min_data_in_leaf") {
			result.min_samples_leaf = static_cast<idx_t>(std::stoull(value));
		} else if (key == "max_bins") {
			result.max_bins = static_cast<idx_t>(std::stoull(value));
		} else if (key == "feature_names") {
			result.feature_names = StringUtil::Split(value, ',');
			for (auto &name : result.feature_names) {
				StringUtil::Trim(name);
			}
		} else {
			throw InvalidInputException("duckboost: unknown train option '%s'", entry.first);
		}
	}
	if (result.n_estimators == 0) {
		throw InvalidInputException("duckboost: n_estimators must be > 0");
	}
	if (result.max_depth == 0) {
		throw InvalidInputException("duckboost: max_depth must be > 0");
	}
	if (!(result.learning_rate > 0)) {
		throw InvalidInputException("duckboost: learning_rate must be > 0");
	}
	return result;
}

EvalOptions EvalOptions::FromMap(const unordered_map<string, string> &options) {
	EvalOptions result;
	result.metric = "auto";
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		if (key == "metric") {
			result.metric = StringUtil::Lower(entry.second);
		} else {
			throw InvalidInputException("duckboost: unknown evaluate option '%s'", entry.first);
		}
	}
	return result;
}

SqlExportOptions SqlExportOptions::FromMap(const unordered_map<string, string> &options) {
	SqlExportOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto value = StringUtil::Lower(entry.second);
		if (key == "separate_trees") {
			if (value == "true" || value == "1" || value == "yes") {
				result.separate_trees = true;
			} else if (value == "false" || value == "0" || value == "no") {
				result.separate_trees = false;
			} else {
				throw InvalidInputException("duckboost: separate_trees must be true/false");
			}
		} else if (key == "prediction_alias" || key == "alias") {
			result.prediction_alias = entry.second;
		} else {
			throw InvalidInputException("duckboost: unknown to_sql option '%s'", entry.first);
		}
	}
	return result;
}

string BoostModel::ToJSON() const {
	std::ostringstream out;
	out << "{\"duckboost_version\":" << duckboost_version;
	out << ",\"backend\":\"" << EscapeJSON(BackendToString(backend)) << "\"";
	out << ",\"task\":\"" << EscapeJSON(TaskToString(task)) << "\"";
	out << ",\"base_score\":" << FormatDouble(base_score);
	if (!base_scores.empty()) {
		out << ",\"base_scores\":[";
		for (idx_t i = 0; i < base_scores.size(); i++) {
			if (i > 0) {
				out << ',';
			}
			out << FormatDouble(base_scores[i]);
		}
		out << ']';
	}
	out << ",\"learning_rate\":" << FormatDouble(learning_rate);
	out << ",\"n_features\":" << n_features;
	out << ",\"n_classes\":" << n_classes;
	out << ",\"feature_names\":[";
	for (idx_t i = 0; i < feature_names.size(); i++) {
		if (i > 0) {
			out << ',';
		}
		out << '"' << EscapeJSON(feature_names[i]) << '"';
	}
	out << "],\"trees\":[";
	for (idx_t t = 0; t < trees.size(); t++) {
		if (t > 0) {
			out << ',';
		}
		out << "{\"nodes\":[";
		auto &tree = trees[t];
		for (idx_t n = 0; n < tree.nodes.size(); n++) {
			if (n > 0) {
				out << ',';
			}
			auto &node = tree.nodes[n];
			out << "{\"is_leaf\":" << (node.is_leaf ? "true" : "false");
			out << ",\"feature\":" << node.feature;
			out << ",\"threshold\":" << FormatDouble(node.threshold);
			out << ",\"left\":" << node.left;
			out << ",\"right\":" << node.right;
			out << ",\"value\":" << FormatDouble(node.value);
			out << '}';
		}
		out << "]}";
	}
	out << "]}";
	return out.str();
}

BoostModel BoostModel::FromJSON(const string &json) {
	JsonParser p(json);
	p.Expect('{');
	BoostModel model;
	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "duckboost_version") {
			model.duckboost_version = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "backend") {
			model.backend = BackendFromString(p.ParseString());
		} else if (key == "task") {
			model.task = TaskFromString(p.ParseString());
		} else if (key == "base_score") {
			model.base_score = p.ParseNumber();
		} else if (key == "base_scores") {
			p.Expect('[');
			bool first_score = true;
			while (!p.TryConsume(']')) {
				if (!first_score) {
					p.Expect(',');
				}
				first_score = false;
				model.base_scores.push_back(p.ParseNumber());
			}
		} else if (key == "learning_rate") {
			model.learning_rate = p.ParseNumber();
		} else if (key == "n_features") {
			model.n_features = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "n_classes") {
			model.n_classes = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "feature_names") {
			p.Expect('[');
			bool first_name = true;
			while (!p.TryConsume(']')) {
				if (!first_name) {
					p.Expect(',');
				}
				first_name = false;
				model.feature_names.push_back(p.ParseString());
			}
		} else if (key == "trees") {
			p.Expect('[');
			bool first_tree = true;
			while (!p.TryConsume(']')) {
				if (!first_tree) {
					p.Expect(',');
				}
				first_tree = false;
				p.Expect('{');
				BoostTree tree;
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto tree_key = p.ParseString();
					p.Expect(':');
					if (tree_key != "nodes") {
						p.SkipValue();
						continue;
					}
					p.Expect('[');
					bool first_node = true;
					while (!p.TryConsume(']')) {
						if (!first_node) {
							p.Expect(',');
						}
						first_node = false;
						p.Expect('{');
						TreeNode node;
						bool first_node_field = true;
						while (!p.TryConsume('}')) {
							if (!first_node_field) {
								p.Expect(',');
							}
							first_node_field = false;
							auto node_key = p.ParseString();
							p.Expect(':');
							if (node_key == "is_leaf") {
								node.is_leaf = p.ParseBool();
							} else if (node_key == "feature") {
								node.feature = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "threshold") {
								node.threshold = p.ParseNumber();
							} else if (node_key == "left") {
								node.left = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "right") {
								node.right = static_cast<idx_t>(p.ParseNumber());
							} else if (node_key == "value") {
								node.value = p.ParseNumber();
							} else {
								p.SkipValue();
							}
						}
						tree.nodes.push_back(node);
					}
				}
				if (tree.nodes.empty()) {
					throw InvalidInputException("duckboost: tree has no nodes");
				}
				model.trees.push_back(std::move(tree));
			}
		} else {
			p.SkipValue();
		}
	}
	if (model.n_features == 0 && !model.feature_names.empty()) {
		model.n_features = model.feature_names.size();
	}
	if (model.task == BoostTask::MULTICLASS) {
		if (model.n_classes < 2) {
			model.n_classes = MaxValue<idx_t>(model.base_scores.size(), 2);
		}
	} else if (model.n_classes == 0) {
		model.n_classes = 1;
	}
	return model;
}

double BoostModel::ClassBias(idx_t class_idx) const {
	if (class_idx < base_scores.size()) {
		return base_scores[class_idx];
	}
	return base_score;
}

double BoostModel::EvalTree(const BoostTree &tree, const vector<double> &features) const {
	idx_t node_idx = 0;
	while (true) {
		if (node_idx >= tree.nodes.size()) {
			throw InvalidInputException("duckboost: corrupt tree during prediction");
		}
		auto &node = tree.nodes[node_idx];
		if (node.is_leaf) {
			return node.value;
		}
		if (node.feature >= features.size()) {
			throw InvalidInputException("duckboost: feature index out of range during prediction");
		}
		node_idx = features[node.feature] < node.threshold ? node.left : node.right;
	}
}

double BoostModel::PredictRaw(const vector<double> &features) const {
	if (task == BoostTask::MULTICLASS) {
		throw InvalidInputException("duckboost: PredictRaw is not defined for multiclass; use Predict / PredictProba");
	}
	if (features.size() < n_features) {
		throw InvalidInputException("duckboost: expected at least %llu features, got %llu",
		                            (unsigned long long)n_features, (unsigned long long)features.size());
	}
	double score = ClassBias(0);
	for (auto &tree : trees) {
		score += learning_rate * EvalTree(tree, features);
	}
	return score;
}

vector<double> BoostModel::PredictRawMulti(const vector<double> &features) const {
	if (features.size() < n_features) {
		throw InvalidInputException("duckboost: expected at least %llu features, got %llu",
		                            (unsigned long long)n_features, (unsigned long long)features.size());
	}
	if (task != BoostTask::MULTICLASS) {
		return {PredictRaw(features)};
	}
	if (n_classes < 2) {
		throw InvalidInputException("duckboost: multiclass model requires n_classes >= 2");
	}
	if (trees.size() % n_classes != 0) {
		throw InvalidInputException("duckboost: multiclass tree count %llu is not divisible by n_classes %llu",
		                            (unsigned long long)trees.size(), (unsigned long long)n_classes);
	}
	vector<double> scores(n_classes);
	for (idx_t c = 0; c < n_classes; c++) {
		scores[c] = ClassBias(c);
	}
	const idx_t n_rounds = trees.size() / n_classes;
	for (idx_t round = 0; round < n_rounds; round++) {
		for (idx_t c = 0; c < n_classes; c++) {
			scores[c] += learning_rate * EvalTree(trees[round * n_classes + c], features);
		}
	}
	return scores;
}

double BoostModel::Predict(const vector<double> &features) const {
	if (task == BoostTask::MULTICLASS) {
		auto scores = PredictRawMulti(features);
		idx_t best = 0;
		for (idx_t c = 1; c < scores.size(); c++) {
			if (scores[c] > scores[best]) {
				best = c;
			}
		}
		return static_cast<double>(best);
	}
	auto raw = PredictRaw(features);
	if (task == BoostTask::BINARY) {
		return Sigmoid(raw);
	}
	return raw;
}

vector<double> BoostModel::PredictProba(const vector<double> &features) const {
	if (task == BoostTask::REGRESSION) {
		return {Predict(features)};
	}
	if (task == BoostTask::BINARY) {
		auto p = Predict(features);
		return {1.0 - p, p};
	}
	auto scores = PredictRawMulti(features);
	double max_score = scores[0];
	for (idx_t i = 1; i < scores.size(); i++) {
		max_score = MaxValue(max_score, scores[i]);
	}
	vector<double> proba(scores.size());
	double sum = 0;
	for (idx_t i = 0; i < scores.size(); i++) {
		proba[i] = std::exp(scores[i] - max_score);
		sum += proba[i];
	}
	if (sum > 0) {
		for (auto &p : proba) {
			p /= sum;
		}
	}
	return proba;
}

string ExportModelSQL(const BoostModel &model, const string &table_name, const vector<string> &feature_columns,
                      const SqlExportOptions &options) {
	if (table_name.empty()) {
		throw InvalidInputException("duckboost: table_name must not be empty");
	}
	vector<string> columns = feature_columns;
	if (columns.empty()) {
		columns = model.feature_names;
	}
	if (columns.size() < model.n_features) {
		throw InvalidInputException("duckboost: need %llu feature column names for SQL export",
		                            (unsigned long long)model.n_features);
	}
	auto alias = QuoteIdent(options.prediction_alias);

	if (model.task == BoostTask::MULTICLASS) {
		if (model.n_classes < 2 || model.trees.size() % model.n_classes != 0) {
			throw InvalidInputException("duckboost: invalid multiclass model for SQL export");
		}
		const idx_t n_rounds = model.trees.size() / model.n_classes;
		std::ostringstream inner;
		inner << "SELECT ";
		for (idx_t c = 0; c < model.n_classes; c++) {
			if (c > 0) {
				inner << ", ";
			}
			string expr = FormatDouble(model.ClassBias(c));
			for (idx_t round = 0; round < n_rounds; round++) {
				expr += " + " + FormatDouble(model.learning_rate) + " * (" +
				        TreeToSQL(model.trees[round * model.n_classes + c], columns, 0) + ")";
			}
			inner << "(" << expr << ") AS " << QuoteIdent("score_" + std::to_string(c));
		}
		inner << " FROM " << QuoteIdent(table_name);

		// Argmax over class scores (ties → lowest class index).
		string argmax = "0";
		for (idx_t c = 1; c < model.n_classes; c++) {
			string cond;
			for (idx_t prev = 0; prev < c; prev++) {
				if (prev > 0) {
					cond += " AND ";
				}
				cond += QuoteIdent("score_" + std::to_string(c)) + " > " + QuoteIdent("score_" + std::to_string(prev));
			}
			argmax = "CASE WHEN " + cond + " THEN " + std::to_string(c) + " ELSE " + argmax + " END";
		}
		return "SELECT (" + argmax + ") AS " + alias + " FROM (" + inner.str() + ") AS " +
		       QuoteIdent("_duckboost_scores");
	}

	if (!options.separate_trees) {
		string expr = FormatDouble(model.ClassBias(0));
		for (auto &tree : model.trees) {
			expr += " + " + FormatDouble(model.learning_rate) + " * (" + TreeToSQL(tree, columns, 0) + ")";
		}
		if (model.task == BoostTask::BINARY) {
			expr = "1.0 / (1.0 + EXP(-(" + expr + ")))";
		}
		return "SELECT (" + expr + ") AS " + alias + " FROM " + QuoteIdent(table_name);
	}

	std::ostringstream inner;
	inner << "SELECT ";
	for (idx_t t = 0; t < model.trees.size(); t++) {
		if (t > 0) {
			inner << ", ";
		}
		inner << "(" << TreeToSQL(model.trees[t], columns, 0) << ") AS " << QuoteIdent("tree_" + std::to_string(t));
	}
	if (model.trees.empty()) {
		inner << FormatDouble(model.ClassBias(0)) << " AS " << QuoteIdent("base_score");
	} else {
		inner << ", " << FormatDouble(model.ClassBias(0)) << " AS " << QuoteIdent("base_score");
	}
	inner << " FROM " << QuoteIdent(table_name);

	string sum_expr = QuoteIdent("base_score");
	for (idx_t t = 0; t < model.trees.size(); t++) {
		sum_expr += " + " + FormatDouble(model.learning_rate) + " * " + QuoteIdent("tree_" + std::to_string(t));
	}
	if (model.task == BoostTask::BINARY) {
		sum_expr = "1.0 / (1.0 + EXP(-(" + sum_expr + ")))";
	}
	return "SELECT (" + sum_expr + ") AS " + alias + " FROM (" + inner.str() + ") AS " + QuoteIdent("_duckboost_trees");
}

} // namespace duckboost
} // namespace duckdb
