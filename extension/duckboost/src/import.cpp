#include "duckboost/import.hpp"
#include "duckboost/json_util.hpp"

#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/string_util.hpp"

#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>

namespace duckdb {
namespace duckboost {

namespace {

double ThresholdForLessEqual(double border) {
	// duckboost uses feature < threshold; convert vendor "<=" / CatBoost False paths.
	if (!std::isfinite(border)) {
		return border;
	}
	auto next = std::nextafter(border, std::numeric_limits<double>::infinity());
	// Avoid subnormals: they round-trip poorly through text/JSON numeric parsers.
	if (std::fpclassify(next) == FP_SUBNORMAL || next == 0.0) {
		return std::numeric_limits<double>::min();
	}
	return next;
}

idx_t ParseFeatureIndex(const string &name, unordered_map<string, idx_t> &name_to_idx, idx_t &n_features) {
	auto it = name_to_idx.find(name);
	if (it != name_to_idx.end()) {
		return it->second;
	}
	if (!name.empty() && (name[0] == 'f' || name[0] == 'F')) {
		bool all_digits = true;
		for (idx_t i = 1; i < name.size(); i++) {
			if (!std::isdigit(static_cast<unsigned char>(name[i]))) {
				all_digits = false;
				break;
			}
		}
		if (all_digits && name.size() > 1) {
			auto idx = static_cast<idx_t>(std::stoull(name.substr(1)));
			n_features = MaxValue<idx_t>(n_features, idx + 1);
			name_to_idx[name] = idx;
			return idx;
		}
	}
	auto idx = n_features++;
	name_to_idx[name] = idx;
	return idx;
}

void ApplyImportOptions(BoostModel &model, const ImportOptions &options) {
	if (options.task_set) {
		model.task = options.task;
	}
	if (options.base_score_set) {
		model.base_score = options.base_score;
	}
	if (options.learning_rate_set) {
		model.learning_rate = options.learning_rate;
	}
	if (!options.feature_names.empty()) {
		model.feature_names = options.feature_names;
		model.n_features = MaxValue<idx_t>(model.n_features, model.feature_names.size());
	}
	if (model.feature_names.size() < model.n_features) {
		model.feature_names.resize(model.n_features);
	}
	for (idx_t i = 0; i < model.feature_names.size(); i++) {
		if (model.feature_names[i].empty()) {
			model.feature_names[i] = "f" + std::to_string(i);
		}
	}
}

struct XGBNode {
	bool is_leaf = false;
	idx_t nodeid = 0;
	idx_t yes = 0;
	idx_t no = 0;
	string split;
	double split_condition = 0;
	double leaf = 0;
	vector<XGBNode> children;
};

XGBNode ParseXGBNode(JsonParser &p) {
	p.Expect('{');
	XGBNode node;
	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "nodeid") {
			node.nodeid = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "leaf") {
			node.is_leaf = true;
			node.leaf = p.ParseNumber();
		} else if (key == "split") {
			node.split = p.ParseString();
		} else if (key == "split_condition") {
			node.split_condition = p.ParseNumber();
		} else if (key == "yes") {
			node.yes = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "no") {
			node.no = static_cast<idx_t>(p.ParseNumber());
		} else if (key == "children") {
			p.Expect('[');
			bool first_child = true;
			while (!p.TryConsume(']')) {
				if (!first_child) {
					p.Expect(',');
				}
				first_child = false;
				node.children.push_back(ParseXGBNode(p));
			}
		} else {
			p.SkipValue();
		}
	}
	return node;
}

idx_t ConvertXGBNode(const XGBNode &node, BoostTree &tree, unordered_map<string, idx_t> &name_to_idx,
                     idx_t &n_features) {
	if (node.is_leaf || node.children.empty()) {
		TreeNode leaf;
		leaf.is_leaf = true;
		leaf.value = node.leaf;
		tree.nodes.push_back(leaf);
		return tree.nodes.size() - 1;
	}
	TreeNode out;
	out.is_leaf = false;
	out.feature = ParseFeatureIndex(node.split, name_to_idx, n_features);
	out.threshold = node.split_condition;
	auto idx = tree.nodes.size();
	tree.nodes.push_back(out);

	const XGBNode *yes_child = nullptr;
	const XGBNode *no_child = nullptr;
	for (auto &child : node.children) {
		if (child.nodeid == node.yes) {
			yes_child = &child;
		} else if (child.nodeid == node.no) {
			no_child = &child;
		}
	}
	if (!yes_child || !no_child) {
		if (node.children.size() != 2) {
			throw InvalidInputException("duckboost: xgboost node %llu missing yes/no children",
			                            (unsigned long long)node.nodeid);
		}
		yes_child = &node.children[0];
		no_child = &node.children[1];
	}
	tree.nodes[idx].left = ConvertXGBNode(*yes_child, tree, name_to_idx, n_features);
	tree.nodes[idx].right = ConvertXGBNode(*no_child, tree, name_to_idx, n_features);
	return idx;
}

vector<string> SplitWhitespace(const string &line) {
	vector<string> parts;
	std::stringstream ss(line);
	string part;
	while (ss >> part) {
		parts.push_back(part);
	}
	return parts;
}

vector<double> ParseDoubleList(const string &value) {
	vector<double> out;
	if (value.empty()) {
		return out;
	}
	auto parts = StringUtil::Split(value, ' ');
	for (auto &part : parts) {
		StringUtil::Trim(part);
		if (part.empty()) {
			continue;
		}
		out.push_back(std::stod(part));
	}
	return out;
}

vector<int64_t> ParseIntList(const string &value) {
	vector<int64_t> out;
	if (value.empty()) {
		return out;
	}
	auto parts = StringUtil::Split(value, ' ');
	for (auto &part : parts) {
		StringUtil::Trim(part);
		if (part.empty()) {
			continue;
		}
		out.push_back(std::stoll(part));
	}
	return out;
}

idx_t ConvertLGBNode(idx_t internal_idx, const vector<int64_t> &split_feature, const vector<double> &threshold,
                     const vector<int64_t> &left_child, const vector<int64_t> &right_child,
                     const vector<double> &leaf_value, double shrinkage, BoostTree &tree) {
	TreeNode node;
	node.is_leaf = false;
	node.feature = static_cast<idx_t>(split_feature[internal_idx]);
	node.threshold = ThresholdForLessEqual(threshold[internal_idx]);
	auto idx = tree.nodes.size();
	tree.nodes.push_back(node);

	auto attach = [&](int64_t child) -> idx_t {
		if (child < 0) {
			TreeNode leaf;
			leaf.is_leaf = true;
			auto leaf_idx = static_cast<idx_t>(~child);
			if (leaf_idx >= leaf_value.size()) {
				throw InvalidInputException("duckboost: lightgbm leaf index out of range");
			}
			leaf.value = leaf_value[leaf_idx] * shrinkage;
			tree.nodes.push_back(leaf);
			return tree.nodes.size() - 1;
		}
		return ConvertLGBNode(static_cast<idx_t>(child), split_feature, threshold, left_child, right_child, leaf_value,
		                      shrinkage, tree);
	};

	tree.nodes[idx].left = attach(left_child[internal_idx]);
	tree.nodes[idx].right = attach(right_child[internal_idx]);
	return idx;
}

idx_t ExpandCatBoost(idx_t depth, idx_t path_bits, const vector<idx_t> &features, const vector<double> &thresholds,
                     const vector<double> &leaf_values, BoostTree &tree) {
	if (depth == features.size()) {
		if (path_bits >= leaf_values.size()) {
			throw InvalidInputException("duckboost: catboost leaf index out of range");
		}
		TreeNode leaf;
		leaf.is_leaf = true;
		leaf.value = leaf_values[path_bits];
		tree.nodes.push_back(leaf);
		return tree.nodes.size() - 1;
	}
	TreeNode node;
	node.is_leaf = false;
	node.feature = features[depth];
	node.threshold = thresholds[depth];
	auto idx = tree.nodes.size();
	tree.nodes.push_back(node);
	// False (feature <= border) → left / bit 0; True (feature > border) → right / bit 1
	tree.nodes[idx].left = ExpandCatBoost(depth + 1, path_bits << 1, features, thresholds, leaf_values, tree);
	tree.nodes[idx].right = ExpandCatBoost(depth + 1, (path_bits << 1) | 1, features, thresholds, leaf_values, tree);
	return idx;
}

} // namespace

ImportOptions ImportOptions::FromMap(const unordered_map<string, string> &options) {
	ImportOptions result;
	for (auto &entry : options) {
		auto key = StringUtil::Lower(entry.first);
		auto &value = entry.second;
		if (key == "task" || key == "objective") {
			result.task = TaskFromString(value);
			result.task_set = true;
		} else if (key == "base_score" || key == "bias") {
			result.base_score = std::stod(value);
			result.base_score_set = true;
		} else if (key == "learning_rate" || key == "eta" || key == "lr" || key == "scale") {
			result.learning_rate = std::stod(value);
			result.learning_rate_set = true;
		} else if (key == "feature_names") {
			result.feature_names = StringUtil::Split(value, ',');
			for (auto &name : result.feature_names) {
				StringUtil::Trim(name);
			}
		} else {
			throw InvalidInputException("duckboost: unknown import option '%s'", entry.first);
		}
	}
	return result;
}

bool LooksLikeDuckBoostJSON(const string &dump) {
	auto trimmed = dump;
	StringUtil::Trim(trimmed);
	return trimmed.find("\"duckboost_version\"") != string::npos;
}

BoostModel ImportXGBoostJSON(const string &dump, const ImportOptions &options) {
	JsonParser p(dump);
	p.Expect('[');
	BoostModel model;
	model.backend = BoostBackend::XGBOOST;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;
	unordered_map<string, idx_t> name_to_idx;
	bool first = true;
	while (!p.TryConsume(']')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto root = ParseXGBNode(p);
		BoostTree tree;
		ConvertXGBNode(root, tree, name_to_idx, model.n_features);
		if (tree.nodes.empty()) {
			throw InvalidInputException("duckboost: empty xgboost tree");
		}
		model.trees.push_back(std::move(tree));
	}
	for (auto &entry : name_to_idx) {
		if (model.feature_names.size() < model.n_features) {
			model.feature_names.resize(model.n_features);
		}
		model.feature_names[entry.second] = entry.first;
	}
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportLightGBMText(const string &dump, const ImportOptions &options) {
	BoostModel model;
	model.backend = BoostBackend::LIGHTGBM;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;

	auto lines = StringUtil::Split(dump, '\n');
	std::map<string, string> tree_fields;
	bool in_tree = false;
	auto flush_tree = [&]() {
		if (!in_tree) {
			return;
		}
		auto num_leaves = static_cast<idx_t>(std::stoull(tree_fields["num_leaves"]));
		if (num_leaves == 0) {
			throw InvalidInputException("duckboost: lightgbm tree has zero leaves");
		}
		auto split_feature = ParseIntList(tree_fields["split_feature"]);
		auto threshold = ParseDoubleList(tree_fields["threshold"]);
		auto left_child = ParseIntList(tree_fields["left_child"]);
		auto right_child = ParseIntList(tree_fields["right_child"]);
		auto leaf_value = ParseDoubleList(tree_fields["leaf_value"]);
		auto decision_type = ParseIntList(tree_fields.count("decision_type") ? tree_fields["decision_type"] : "");
		double shrinkage = 1.0;
		if (tree_fields.count("shrinkage")) {
			shrinkage = std::stod(tree_fields["shrinkage"]);
		}
		if (leaf_value.size() != num_leaves) {
			throw InvalidInputException("duckboost: lightgbm leaf_value size mismatch");
		}
		for (auto dt : decision_type) {
			if (dt & 1) {
				throw NotImplementedException("duckboost: lightgbm categorical splits are not supported");
			}
		}
		BoostTree tree;
		if (num_leaves == 1) {
			TreeNode leaf;
			leaf.is_leaf = true;
			leaf.value = leaf_value[0] * shrinkage;
			tree.nodes.push_back(leaf);
		} else {
			if (split_feature.size() != num_leaves - 1 || threshold.size() != num_leaves - 1 ||
			    left_child.size() != num_leaves - 1 || right_child.size() != num_leaves - 1) {
				throw InvalidInputException("duckboost: lightgbm split array size mismatch");
			}
			for (auto f : split_feature) {
				if (f < 0) {
					throw InvalidInputException("duckboost: lightgbm split_feature must be non-negative");
				}
				model.n_features = MaxValue<idx_t>(model.n_features, static_cast<idx_t>(f) + 1);
			}
			ConvertLGBNode(0, split_feature, threshold, left_child, right_child, leaf_value, shrinkage, tree);
		}
		model.trees.push_back(std::move(tree));
		tree_fields.clear();
		in_tree = false;
	};

	for (auto &raw_line : lines) {
		auto line = raw_line;
		StringUtil::Trim(line);
		if (line.empty()) {
			continue;
		}
		if (line.rfind("Tree=", 0) == 0) {
			flush_tree();
			in_tree = true;
			continue;
		}
		if (line == "end of trees" || line.rfind("tree_sizes=", 0) == 0) {
			flush_tree();
			continue;
		}
		auto eq = line.find('=');
		if (eq == string::npos) {
			continue;
		}
		auto key = line.substr(0, eq);
		auto value = line.substr(eq + 1);
		StringUtil::Trim(key);
		StringUtil::Trim(value);
		if (in_tree) {
			tree_fields[key] = value;
			continue;
		}
		if (key == "objective") {
			auto lower = StringUtil::Lower(value);
			if (lower.find("binary") != string::npos || lower.find("logistic") != string::npos) {
				model.task = BoostTask::BINARY;
			} else {
				model.task = BoostTask::REGRESSION;
			}
		} else if (key == "feature_names") {
			model.feature_names = SplitWhitespace(value);
			model.n_features = MaxValue<idx_t>(model.n_features, model.feature_names.size());
		} else if (key == "max_feature_idx") {
			model.n_features = MaxValue<idx_t>(model.n_features, static_cast<idx_t>(std::stoull(value)) + 1);
		}
	}
	flush_tree();
	if (model.trees.empty()) {
		throw InvalidInputException("duckboost: no trees found in lightgbm dump");
	}
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportCatBoostJSON(const string &dump, const ImportOptions &options) {
	JsonParser p(dump);
	p.Expect('{');
	BoostModel model;
	model.backend = BoostBackend::CATBOOST;
	model.learning_rate = 1.0;
	model.task = BoostTask::REGRESSION;

	struct FloatFeatureInfo {
		idx_t feature_index = 0;
		idx_t flat_feature_index = 0;
		vector<double> borders;
	};
	vector<FloatFeatureInfo> float_features;
	vector<std::pair<idx_t, double>> global_splits; // split_index → (feature, border)
	double scale = 1.0;
	double bias = 0.0;
	bool has_trees = false;

	struct TreeSplit {
		idx_t feature = 0;
		double border = 0;
		bool resolved = false;
		idx_t split_index = 0;
	};
	struct PendingTree {
		vector<double> leaf_values;
		vector<TreeSplit> splits;
	};
	vector<PendingTree> pending_trees;

	bool first = true;
	while (!p.TryConsume('}')) {
		if (!first) {
			p.Expect(',');
		}
		first = false;
		auto key = p.ParseString();
		p.Expect(':');
		if (key == "features_info") {
			p.Expect('{');
			bool first_fi = true;
			while (!p.TryConsume('}')) {
				if (!first_fi) {
					p.Expect(',');
				}
				first_fi = false;
				auto fi_key = p.ParseString();
				p.Expect(':');
				if (fi_key != "float_features") {
					p.SkipValue();
					continue;
				}
				p.Expect('[');
				bool first_feat = true;
				while (!p.TryConsume(']')) {
					if (!first_feat) {
						p.Expect(',');
					}
					first_feat = false;
					p.Expect('{');
					FloatFeatureInfo feat;
					bool has_feature_index = false;
					bool has_flat = false;
					bool first_field = true;
					while (!p.TryConsume('}')) {
						if (!first_field) {
							p.Expect(',');
						}
						first_field = false;
						auto fkey = p.ParseString();
						p.Expect(':');
						if (fkey == "feature_index") {
							feat.feature_index = static_cast<idx_t>(p.ParseNumber());
							has_feature_index = true;
						} else if (fkey == "flat_feature_index") {
							feat.flat_feature_index = static_cast<idx_t>(p.ParseNumber());
							has_flat = true;
						} else if (fkey == "borders") {
							p.Expect('[');
							bool first_border = true;
							while (!p.TryConsume(']')) {
								if (!first_border) {
									p.Expect(',');
								}
								first_border = false;
								feat.borders.push_back(p.ParseNumber());
							}
						} else {
							p.SkipValue();
						}
					}
					if (!has_flat && has_feature_index) {
						feat.flat_feature_index = feat.feature_index;
					}
					if (!has_feature_index && has_flat) {
						feat.feature_index = feat.flat_feature_index;
					}
					float_features.push_back(std::move(feat));
				}
			}
		} else if (key == "oblivious_trees") {
			has_trees = true;
			p.Expect('[');
			bool first_tree = true;
			while (!p.TryConsume(']')) {
				if (!first_tree) {
					p.Expect(',');
				}
				first_tree = false;
				p.Expect('{');
				PendingTree tree;
				bool first_field = true;
				while (!p.TryConsume('}')) {
					if (!first_field) {
						p.Expect(',');
					}
					first_field = false;
					auto tkey = p.ParseString();
					p.Expect(':');
					if (tkey == "leaf_values") {
						p.Expect('[');
						bool first_leaf = true;
						while (!p.TryConsume(']')) {
							if (!first_leaf) {
								p.Expect(',');
							}
							first_leaf = false;
							tree.leaf_values.push_back(p.ParseNumber());
						}
					} else if (tkey == "splits") {
						p.Expect('[');
						bool first_split = true;
						while (!p.TryConsume(']')) {
							if (!first_split) {
								p.Expect(',');
							}
							first_split = false;
							p.Expect('{');
							TreeSplit split;
							bool has_border = false;
							bool has_feature = false;
							bool first_sfield = true;
							while (!p.TryConsume('}')) {
								if (!first_sfield) {
									p.Expect(',');
								}
								first_sfield = false;
								auto skey = p.ParseString();
								p.Expect(':');
								if (skey == "split_index") {
									split.split_index = static_cast<idx_t>(p.ParseNumber());
								} else if (skey == "float_feature_index" || skey == "flat_feature_index") {
									split.feature = static_cast<idx_t>(p.ParseNumber());
									has_feature = true;
								} else if (skey == "border") {
									split.border = p.ParseNumber();
									has_border = true;
								} else if (skey == "split_type") {
									auto st = p.ParseString();
									if (st != "FloatFeature") {
										throw NotImplementedException(
										    "duckboost: catboost split_type '%s' is not supported (FloatFeature only)",
										    st);
									}
								} else {
									p.SkipValue();
								}
							}
							split.resolved = has_border && has_feature;
							tree.splits.push_back(split);
						}
					} else {
						p.SkipValue();
					}
				}
				pending_trees.push_back(std::move(tree));
			}
		} else if (key == "scale_and_bias") {
			p.Expect('[');
			scale = p.ParseNumber();
			if (p.TryConsume(',')) {
				if (p.Peek() == '[') {
					p.Expect('[');
					bias = p.ParseNumber();
					while (!p.TryConsume(']')) {
						p.Expect(',');
						p.SkipValue();
					}
				} else {
					bias = p.ParseNumber();
				}
			}
			p.Expect(']');
		} else if (key == "model_info") {
			p.Expect('{');
			bool first_mi = true;
			while (!p.TryConsume('}')) {
				if (!first_mi) {
					p.Expect(',');
				}
				first_mi = false;
				auto mi_key = p.ParseString();
				p.Expect(':');
				if (mi_key == "params" && p.Peek() == '{') {
					p.Expect('{');
					bool first_param = true;
					while (!p.TryConsume('}')) {
						if (!first_param) {
							p.Expect(',');
						}
						first_param = false;
						auto pk = p.ParseString();
						p.Expect(':');
						if ((pk == "loss_function" || pk == "objective") && p.Peek() == '"') {
							auto loss = StringUtil::Lower(p.ParseString());
							if (loss.find("logloss") != string::npos || loss.find("crossentropy") != string::npos) {
								model.task = BoostTask::BINARY;
							}
						} else {
							p.SkipValue();
						}
					}
				} else {
					p.SkipValue();
				}
			}
		} else {
			p.SkipValue();
		}
	}

	if (!has_trees) {
		throw InvalidInputException("duckboost: catboost JSON missing oblivious_trees");
	}

	for (auto &feat : float_features) {
		auto feature_id = feat.flat_feature_index;
		model.n_features = MaxValue<idx_t>(model.n_features, feature_id + 1);
		for (auto border : feat.borders) {
			global_splits.emplace_back(feature_id, border);
		}
	}

	for (auto &pending : pending_trees) {
		vector<idx_t> features;
		vector<double> thresholds;
		features.reserve(pending.splits.size());
		thresholds.reserve(pending.splits.size());
		for (auto &split : pending.splits) {
			idx_t feature = split.feature;
			double border = split.border;
			if (!split.resolved) {
				if (split.split_index >= global_splits.size()) {
					throw InvalidInputException("duckboost: catboost split_index %llu out of range",
					                            (unsigned long long)split.split_index);
				}
				feature = global_splits[split.split_index].first;
				border = global_splits[split.split_index].second;
			}
			model.n_features = MaxValue<idx_t>(model.n_features, feature + 1);
			features.push_back(feature);
			// CatBoost True when feature > border; duckboost left uses feature < threshold.
			thresholds.push_back(ThresholdForLessEqual(border));
		}
		idx_t expected_leaves = idx_t(1) << features.size();
		if (pending.leaf_values.size() != expected_leaves) {
			throw NotImplementedException(
			    "duckboost: catboost tree leaf_values size %llu != 2^depth (%llu); multiclass not supported yet",
			    (unsigned long long)pending.leaf_values.size(), (unsigned long long)expected_leaves);
		}
		for (auto &value : pending.leaf_values) {
			value *= scale;
		}
		BoostTree tree;
		ExpandCatBoost(0, 0, features, thresholds, pending.leaf_values, tree);
		model.trees.push_back(std::move(tree));
	}

	model.base_score = bias;
	ApplyImportOptions(model, options);
	return model;
}

BoostModel ImportModel(BoostBackend backend, const string &dump, const ImportOptions &options) {
	if (dump.empty()) {
		throw InvalidInputException("duckboost: empty model dump");
	}
	if (LooksLikeDuckBoostJSON(dump) || backend == BoostBackend::REFERENCE) {
		auto model = BoostModel::FromJSON(dump);
		if (backend != BoostBackend::REFERENCE) {
			model.backend = backend;
		}
		ApplyImportOptions(model, options);
		return model;
	}
	switch (backend) {
	case BoostBackend::XGBOOST:
		return ImportXGBoostJSON(dump, options);
	case BoostBackend::LIGHTGBM:
		return ImportLightGBMText(dump, options);
	case BoostBackend::CATBOOST:
		return ImportCatBoostJSON(dump, options);
	case BoostBackend::REFERENCE:
	default:
		throw InvalidInputException("duckboost: cannot import vendor dump with backend 'reference'");
	}
}

} // namespace duckboost
} // namespace duckdb
