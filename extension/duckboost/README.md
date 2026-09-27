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
```

Then:

```bash
build/reldebug/test/unittest test/sql/duckboost/*
```

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
```

### Backends

| Backend | Train in this build | Dump import | Predict / evaluate / `to_sql` |
| --- | --- | --- | --- |
| `reference` | Yes (built-in GBDT) | duckboost JSON | Yes |
| `xgboost` | Optional (`DUCKBOOST_WITH_XGBOOST`) | `dump_model(..., dump_format='json')` | Yes |
| `lightgbm` | Optional (`DUCKBOOST_WITH_LIGHTGBM`) | `booster_.save_model()` text | Yes |
| `catboost` | Optional (`DUCKBOOST_WITH_CATBOOST`) | `save_model(..., format='json')` float trees | Yes |

Native XGBoost / LightGBM / CatBoost linking is intentionally opt-in. Prefer training outside DuckDB and importing dumps when you need production booster quality.

### Dump import notes

- Imported models set `learning_rate = 1.0` and bake vendor shrinkage/scale into leaf values (XGBoost dump leaves already include η; LightGBM `shrinkage` and CatBoost `scale_and_bias` are applied at import).
- LightGBM numerical `<=` splits are converted to duckboost `<` via `nextafter(threshold, +∞)`.
- CatBoost support is float/`FloatFeature` oblivious trees only (no OneHot/CTR/multiclass yet).
- Optional import map keys: `task`, `base_score`, `learning_rate`, `feature_names`.

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
  "feature_names": ["x1", "x2"],
  "trees": [{"nodes": [{"is_leaf": false, "feature": 0, "threshold": 1.5, "left": 1, "right": 2, "value": 0.0}, ...]}]
}
```

## Design notes

- **Intended home**: out-of-tree community extension (heavy optional deps + ML surface area). Prototyped in-tree here for DuckDB 2.0 development.
- **SQL export** mirrors orbital's `separate_trees` idea so DuckDB can evaluate ensemble members as independent columns.
- **Reference trainer** is a didactic histogram/quantile-split GBDT (squared error + logistic). It is not a replacement for production XGBoost/LightGBM/CatBoost quality, but it exercises the full train → evaluate → SQL path.

## Roadmap

- Native trainers behind `DUCKBOOST_WITH_*` CMake options
- CatBoost OneHot/CTR + multiclass dump support
- Model catalog table macros (`duckboost_fit`, `duckboost_score`)
- Community extension packaging
