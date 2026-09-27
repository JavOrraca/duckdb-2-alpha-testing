# Experimental in-tree duckboost extension (gradient boosting + SQL export).
# Build with: DUCKDB_EXTENSIONS='duckboost' make reldebug
# Or: EXTENSION_CONFIGS=.github/config/extensions/duckboost.cmake make reldebug
duckdb_extension_load(duckboost LOAD_TESTS)
