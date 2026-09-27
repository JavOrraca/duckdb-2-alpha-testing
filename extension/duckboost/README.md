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
```

### Backends

| Backend | Train in this build | Predict / evaluate / `to_sql` |
| --- | --- | --- |
| `reference` | Yes (built-in GBDT) | Yes |
| `xgboost` | Optional (`DUCKBOOST_WITH_XGBOOST`) | Via `duckboost_import` of duckboost JSON |
| `lightgbm` | Optional (`DUCKBOOST_WITH_LIGHTGBM`) | Via `duckboost_import` |
| `catboost` | Optional (`DUCKBOOST_WITH_CATBOOST`) | Via `duckboost_import` |

Native XGBoost / LightGBM / CatBoost linking is intentionally opt-in so the extension builds without those heavyweight dependencies. The unified duckboost JSON model format is the interchange layer; adapters that convert vendor dumps into that format are the next integration step.

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
- Importers for XGBoost JSON / LightGBM text / CatBoost JSON dumps
- Multiclass + ranking objectives
- Model catalog table macros (`duckboost_fit`, `duckboost_score`)
- Community extension packaging
