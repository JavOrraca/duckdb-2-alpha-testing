#include "duckboost/model.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/exception/binder_exception.hpp"

#include <cctype>
#include <cstdio>
#include <sstream>

namespace duckdb {
namespace duckboost {

namespace {

string EscapeJSON(const string &input) {
	string out;
	out.reserve(input.size() + 8);
	for (auto c : input) {
		switch (c) {
		case '"':
			out += "\\\"";
			break;
		case '\\':
			out += "\\\\";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			out.push_back(c);
			break;
		}
	}
	return out;
}

string FormatDouble(double value) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.17g", value);
	return string(buf);
}

struct JsonParser {
	explicit JsonParser(const string &input_p) : input(input_p), pos(0) {
	}

	const string &input;
	idx_t pos;

	void SkipWs() {
		while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos]))) {
			pos++;
		}
	}

	char Peek() {
		SkipWs();
		if (pos >= input.size()) {
			throw InvalidInputException("duckboost: unexpected end of JSON");
		}
		return input[pos];
	}

	char Consume() {
		auto c = Peek();
		pos++;
		return c;
	}

	void Expect(char c) {
		auto got = Consume();
		if (got != c) {
			throw InvalidInputException("duckboost: expected '%c' in JSON, got '%c'", c, got);
		}
	}

	bool TryConsume(char c) {
		SkipWs();
		if (pos < input.size() && input[pos] == c) {
			pos++;
			return true;
		}
		return false;
	}

	string ParseString() {
		Expect('"');
		string out;
		while (pos < input.size()) {
			auto c = input[pos++];
			if (c == '"') {
				return out;
			}
			if (c == '\\') {
				if (pos >= input.size()) {
					throw InvalidInputException("duckboost: truncated escape in JSON string");
				}
				auto e = input[pos++];
				switch (e) {
				case '"':
				case '\\':
				case '/':
					out.push_back(e);
					break;
				case 'n':
					out.push_back('\n');
					break;
				case 'r':
					out.push_back('\r');
					break;
				case 't':
					out.push_back('\t');
					break;
				default:
					out.push_back(e);
					break;
				}
			} else {
				out.push_back(c);
			}
		}
		throw InvalidInputException("duckboost: unterminated JSON string");
	}

	double ParseNumber() {
		SkipWs();
		idx_t start = pos;
		if (pos < input.size() && (input[pos] == '-' || input[pos] == '+')) {
			pos++;
		}
		while (pos < input.size() &&
		       (std::isdigit(static_cast<unsigned char>(input[pos])) || input[pos] == '.' || input[pos] == 'e' ||
		        input[pos] == 'E' || input[pos] == '+' || input[pos] == '-')) {
			pos++;
		}
		if (start == pos) {
			throw InvalidInputException("duckboost: expected number in JSON");
		}
		return std::stod(input.substr(start, pos - start));
	}

	bool ParseBool() {
		SkipWs();
		if (input.compare(pos, 4, "true") == 0) {
			pos += 4;
			return true;
		}
		if (input.compare(pos, 5, "false") == 0) {
			pos += 5;
			return false;
		}
		throw InvalidInputException("duckboost: expected boolean in JSON");
	}

	void SkipValue() {
		SkipWs();
		auto c = Peek();
		if (c == '"') {
			ParseString();
		} else if (c == '{') {
			Expect('{');
			if (!TryConsume('}')) {
				do {
					ParseString();
					Expect(':');
					SkipValue();
				} while (TryConsume(','));
				Expect('}');
			}
		} else if (c == '[') {
			Expect('[');
			if (!TryConsume(']')) {
				do {
					SkipValue();
				} while (TryConsume(','));
				Expect(']');
			}
		} else if (c == 't' || c == 'f') {
			ParseBool();
		} else if (c == 'n') {
			if (input.compare(pos, 4, "null") != 0) {
				throw InvalidInputException("duckboost: expected null in JSON");
			}
			pos += 4;
		} else {
			ParseNumber();
		}
	}
};

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
	return task == BoostTask::BINARY ? "binary" : "regression";
}

BoostTask TaskFromString(const string &name) {
	auto lower = StringUtil::Lower(name);
	if (lower == "regression" || lower == "regressor" || lower == "mse") {
		return BoostTask::REGRESSION;
	}
	if (lower == "binary" || lower == "classification" || lower == "logistic" || lower == "binomial") {
		return BoostTask::BINARY;
	}
	throw InvalidInputException("duckboost: unknown task '%s' (expected regression or binary)", name);
}

bool BackendTrainingSupported(BoostBackend backend) {
	return backend == BoostBackend::REFERENCE;
}

string BackendCapabilityNote(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::REFERENCE:
		return "in-process reference GBDT (train/predict/evaluate/to_sql)";
	case BoostBackend::XGBOOST:
		return "selected backend; native train requires DUCKBOOST_WITH_XGBOOST (import + SQL export supported)";
	case BoostBackend::LIGHTGBM:
		return "selected backend; native train requires DUCKBOOST_WITH_LIGHTGBM (import + SQL export supported)";
	case BoostBackend::CATBOOST:
		return "selected backend; native train requires DUCKBOOST_WITH_CATBOOST (import + SQL export supported)";
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
	out << ",\"learning_rate\":" << FormatDouble(learning_rate);
	out << ",\"n_features\":" << n_features;
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
		} else if (key == "learning_rate") {
			model.learning_rate = p.ParseNumber();
		} else if (key == "n_features") {
			model.n_features = static_cast<idx_t>(p.ParseNumber());
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
	return model;
}

double BoostModel::PredictRaw(const vector<double> &features) const {
	if (features.size() < n_features) {
		throw InvalidInputException("duckboost: expected at least %llu features, got %llu",
		                            (unsigned long long)n_features, (unsigned long long)features.size());
	}
	double score = base_score;
	for (auto &tree : trees) {
		idx_t node_idx = 0;
		while (true) {
			if (node_idx >= tree.nodes.size()) {
				throw InvalidInputException("duckboost: corrupt tree during prediction");
			}
			auto &node = tree.nodes[node_idx];
			if (node.is_leaf) {
				score += learning_rate * node.value;
				break;
			}
			if (node.feature >= features.size()) {
				throw InvalidInputException("duckboost: feature index out of range during prediction");
			}
			node_idx = features[node.feature] < node.threshold ? node.left : node.right;
		}
	}
	return score;
}

double BoostModel::Predict(const vector<double> &features) const {
	auto raw = PredictRaw(features);
	if (task == BoostTask::BINARY) {
		return Sigmoid(raw);
	}
	return raw;
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
	if (!options.separate_trees) {
		string expr = FormatDouble(model.base_score);
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
		inner << FormatDouble(model.base_score) << " AS " << QuoteIdent("base_score");
	} else {
		inner << ", " << FormatDouble(model.base_score) << " AS " << QuoteIdent("base_score");
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
