# Packaging duckboost as a community / out-of-tree extension

`duckboost` is prototyped in-tree under `extension/duckboost` for DuckDB 2.0 development. The intended long-term home is a **community (out-of-tree) extension** so optional ML vendor libraries do not weigh down core DuckDB builds.

## In-tree build (current)

```bash
DUCKDB_EXTENSIONS='duckboost' make reldebug
# or
EXTENSION_CONFIGS=.github/config/extensions/duckboost.cmake make reldebug
# or
BUILD_DUCKBOOST=1 make reldebug
```

## Out-of-tree style load from this directory

`extension_config.cmake` in this folder mirrors the [extension-template](https://github.com/duckdb/extension-template) pattern:

```bash
EXTENSION_CONFIGS=extension/duckboost/extension_config.cmake make reldebug
```

## Extracting a standalone repository

Suggested layout (compatible with DuckDB `extension_external/`):

```text
duckboost/
  CMakeLists.txt
  duckboost_extension.cpp
  extension_config.cmake
  include/duckboost/...
  src/...
  test/sql/...
  README.md
  PACKAGING.md
```

Steps:

1. Copy `extension/duckboost/**` into a new git repo (preserve `include/`, `src/`, tests).
2. Point DuckDB at it:

```cmake
# extension/extension_config_local.cmake (gitignored in DuckDB)
duckdb_extension_load(duckboost
    SOURCE_DIR /path/to/duckboost
    LOAD_TESTS
)
```

Or place the repo at `duckdb/extension_external/duckboost` and load by name.

3. Publish via [DuckDB community extensions](https://duckdb.org/community_extensions/) once the API stabilizes.

## Native trainer flags

| CMake option | Env hint | Effect |
| --- | --- | --- |
| `DUCKBOOST_WITH_XGBOOST=ON` | `XGBOOST_ROOT` | Compile/link XGBoost trainer path |
| `DUCKBOOST_WITH_LIGHTGBM=ON` | `LIGHTGBM_ROOT` | Compile/link LightGBM trainer path |
| `DUCKBOOST_WITH_CATBOOST=ON` | `CATBOOST_ROOT` | Compile/link CatBoost trainer path |
| `DUCKBOOST_NATIVE_STUB_ONLY=ON` | — | Compile `#ifdef` paths without linking vendor libs |

Inspect the active build:

```sql
SELECT * FROM duckboost_build_info();
SELECT * FROM duckboost_backends();
```

Until the vendor C API bridges are finished, prefer `duckboost_import()` for production boosters and `backend='reference'` for in-process experiments.
