# duckboost

Experimental DuckDB extension for **in-database gradient boosting**: train and evaluate tree ensembles inside DuckDB, then export pure SQL for orbital-style in-database inference.

## Motivation

[Orbital](https://posit-dev.github.io/orbital/) converts trained sklearn / tidymodels pipelines into SQL so scoring needs no Python runtime. `duckboost` brings a similar loop fully into DuckDB:

1. **Train** with a selectable boosting backend
2. **Evaluate** fit quality in SQL
3. **Export** model trees to DuckDB SQL (`CASE WHEN` ensembles, optional `separate_trees`)
4. **Score** with either `duckboost_predict` or the exported SQL (no extension required at inference time)

## Build

```bash
DUCKDB_EXTENSIONS='duckboost' make reldebug
# or
EXTENSION_CONFIGS=.github/config/extensions/duckboost.cmake make reldebug
# or
BUILD_DUCKBOOST=1 make reldebug
# out-of-tree style (same sources via local extension_config.cmake)
EXTENSION_CONFIGS=extension/duckboost/extension_config.cmake make reldebug
```

Optional native trainer flags (XGBoost / LightGBM train via vendor C API → dump → import):

```bash
# Link real libraries (pip wheels work; set ROOT or rely on auto-detect under ~/.local)
EXTRA_CMAKE_VARIABLES='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_WITH_LIGHTGBM=ON' \
  DUCKDB_EXTENSIONS='duckboost' make reldebug

# Compile #ifdef paths without linking vendor libraries
EXTRA_CMAKE_VARIABLES='-DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_NATIVE_STUB_ONLY=ON' \
  DUCKDB_EXTENSIONS='duckboost' make reldebug
```

CatBoost has no public in-process training C API — use `duckboost_import('catboost', ...)`.

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Native train tests (linked builds only):

```bash
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/xgboost/lib:$HOME/.local/lib/python3.12/site-packages/lightgbm/lib:${LD_LIBRARY_PATH}"
DUCKBOOST_NATIVE_TRAIN_TEST=1 build/reldebug/test/unittest test/sql/duckboost/native_train.test
```

Then:

```bash
build/reldebug/test/unittest test/sql/duckboost/*
```

See [`PACKAGING.md`](PACKAGING.md) for community / out-of-tree extraction.

## SQL API

```sql
LOAD duckboost;

-- Available backends
SELECT * FROM duckboost_backends();

-- Train (reference GBDT backend)
CREATE TABLE models AS
SELECT duckboost_train(
	y,
	[x1, x2],
	MAP {
		'backend': 'reference',
		'task': 'regression',
		'n_estimators': '20',
		'max_depth': '3',
		'learning_rate': '0.1',
		'feature_names': 'x1,x2'
	}
) AS model
FROM train;

-- In-process predict
SELECT duckboost_predict(model, [x1, x2]) FROM models, test;

-- Dataset-level metrics
SELECT duckboost_evaluate_agg(model, y, [x1, x2], MAP {'metric': 'rmse'})
FROM models, test;

-- Orbital-style SQL export (trees as separate columns for DuckDB parallelism)
SELECT duckboost_to_sql(
	model,
	'test',
	['x1', 'x2'],
	MAP {'separate_trees': 'true', 'prediction_alias': 'pred'}
) FROM models;

-- Import a vendor dump trained outside DuckDB
SELECT duckboost_import('xgboost', xgb_dump_json, MAP {'task': 'binary', 'base_score': '0.0'});
SELECT duckboost_import('lightgbm', lgb_model_txt);
SELECT duckboost_import('catboost', catboost_model_json);

-- Table macros (query_table wrappers; pass table names as VARCHAR literals)
CREATE TABLE models AS
FROM duckboost_fit('train', y, [x1, x2], options := MAP {'n_estimators': '20', 'feature_names': 'x1,x2'});

SELECT * FROM duckboost_score((SELECT model FROM models), 'test', [x1, x2]);

-- Multiclass: predict returns argmax class index; predict_proba returns softmax LIST
SELECT duckboost_predict(model, features), duckboost_predict_proba(model, features);
```

### Backends

| Backend | Train in this build | Dump import | Predict / evaluate / `to_sql` |
| --- | --- | --- | --- |
| `reference` | Yes (built-in GBDT) | duckboost JSON | Yes |
| `xgboost` | Optional (`DUCKBOOST_WITH_XGBOOST`) | `dump_model(..., dump_format='json')` | Yes |
| `lightgbm` | Optional (`DUCKBOOST_WITH_LIGHTGBM`) | `booster_.save_model()` text | Yes |
| `catboost` | Optional (`DUCKBOOST_WITH_CATBOOST`) | `save_model(..., format='json')` float trees | Yes |

Native XGBoost / LightGBM linking is opt-in via `DUCKBOOST_WITH_*`. Linked builds train in-process through the vendor C API, then convert the dump into duckboost JSON. CatBoost remains import-only. Prefer `duckboost_import()` when you already train outside DuckDB; use `backend='reference'` for dependency-free experiments (including feature NULLs/NaNs).

### Reference trainer options

| Key | Default | Notes |
| --- | --- | --- |
| `task` | `regression` | `regression`, `binary`, or `multiclass` |
| `n_estimators` | `10` | Boosting rounds |
| `max_depth` | `3` | Depth-wise trees |
| `learning_rate` | `0.1` | Shrinkage (`eta` / `lr`) |
| `max_bins` | `256` | Gradient-histogram bin budget for numeric splits |
| `grow_policy` | `depth` | `depth` (depth-wise) or `leaf` / `lossguide` (leaf-wise) |
| `max_leaves` / `num_leaves` | `0`→`31` | Leaf budget when `grow_policy=leaf` |
| `min_samples_leaf` | `1` | Min rows per child |
| `min_child_weight` | `1` | Min hessian sum per child |
| `reg_lambda` / `reg_alpha` | `1` / `0` | L2 / L1 on leaf weights |
| `gamma` | `0` | Min split gain |
| `subsample` / `colsample_bytree` | `1` / `1` | Row / column bagging per tree |
| `early_stopping_rounds` | `0` | With `validation_fraction` (default `0.2` when early stopping set) |
| `seed` | `0` | RNG seed for sampling / valid split |
| `n_classes` | inferred | Multiclass class count override |
| `cat_features` | _(none)_ | Comma-separated indices or `feature_names` for EQUAL splits |
| `class_weight` | _(none)_ | `balanced` or comma-separated per-class multipliers |

Optional sample weights: `duckboost_train(y, features, weight [, options])`.

SQL `NULL` and IEEE NaN in feature lists are treated as missing. The reference trainer learns a per-split default direction (serialized as `default_left`), matching XGBoost’s missing-child behavior. Target `y` must still be non-NULL. Multiclass uses softmax (`task: multiclass`); categorical columns use `compare: equal` splits.

### Dump import notes

- Imported models set `learning_rate = 1.0` and bake vendor shrinkage/scale into leaf values (XGBoost dump leaves already include η; LightGBM `shrinkage` and CatBoost `scale_and_bias` are applied at import).
- LightGBM numerical `<=` splits are converted to duckboost `<` via `nextafter(threshold, +∞)`.
- CatBoost support: `FloatFeature`, `OneHotFeature`, and `OnlineCtr` (Counter/Borders). CTR combinations may include `cat_feature_value`, `float_feature`, and `cat_feature_exact_value`. Pass categorical CityHash values as numeric features.
- Multiclass CatBoost JSON uses class-blocked `leaf_values` (`2^depth` values per class). Import expands each oblivious tree into `n_classes` duckboost trees (layout `[round][class]`).
- OnlineCtr requires `ctr_data` in the dump (`save_model(..., pool=...)`). `duckboost_to_sql` inlines CTR hash lookups via `UHUGEINT` modular arithmetic.
- Optional import map keys: `task`, `base_score`, `learning_rate`, `feature_names`, `n_classes`.

## Model format

Models are opaque `VARCHAR` JSON documents:

```json
{
  "duckboost_version": 1,
  "backend": "reference",
  "task": "regression",
  "base_score": 0.0,
  "learning_rate": 0.1,
  "n_features": 2,
  "n_classes": 1,
  "feature_names": ["x1", "x2"],
  "trees": [{"nodes": [{"is_leaf": false, "feature": 0, "threshold": 1.5, "left": 1, "right": 2, "value": 0.0}, ...]}]
}
```

For `task: "multiclass"`, `n_classes >= 2`, optional `base_scores` holds per-class bias, and trees are stored as `round * n_classes + class`. `duckboost_predict` returns the argmax class index; `duckboost_to_sql` exports an argmax over per-class score expressions.

## Design notes

- **Intended home**: out-of-tree community extension (heavy optional deps + ML surface area). Prototyped in-tree here for DuckDB 2.0 development.
- **SQL export** mirrors orbital's `separate_trees` idea so DuckDB can evaluate ensemble members as independent columns.
- **Reference trainer** is a second-order histogram GBDT (squared error, logistic, softmax) with
  XGBoost-style missing-value defaults, categorical EQUAL splits, depth- or leaf-wise growth,
  sample/class weights, L1/L2/`gamma`, row/column subsample, and optional early stopping. Linked
  XGBoost/LightGBM bridges forward weights, categoricals, and leaf-wise knobs when available.
- **Table macros** `duckboost_fit` / `duckboost_score` wrap `duckboost_train` / `duckboost_predict` with `query_table` for a compact SQL workflow.

## Roadmap

- [x] Native trainer scaffolding behind `DUCKBOOST_WITH_*` / `DUCKBOOST_NATIVE_STUB_ONLY` + `duckboost_build_info()`
- [x] Community packaging docs (`PACKAGING.md`, local `extension_config.cmake`)
- [x] Vendor C API bridges for XGBoost / LightGBM (train → dump → import into `BoostModel`)
- [x] Community publish kit (`community/description.yml`, `scripts/extract_duckboost_oot.sh`, dual-mode CMake)
- [x] Orphan publish surface `cursor/duckboost-community-oot-0c09` + submit-ready descriptor (`community/SUBMIT.md`)
- [ ] Open PR on `duckdb/community-extensions` with `extensions/duckboost/description.yml`
