cmake_minimum_required(VERSION 2.8.12...3.29)

project(DuckboostExtension)

include_directories(include)

set(DUCKBOOST_EXTENSION_FILES
    duckboost_extension.cpp
    src/model.cpp
    src/trainer.cpp
    src/import.cpp
    src/duckboost_functions.cpp)

build_extension_library(duckboost ${DUCKBOOST_EXTENSION_FILES})

# Optional native library backends (future):
#   -DDUCKBOOST_WITH_XGBOOST=1
#   -DDUCKBOOST_WITH_LIGHTGBM=1
#   -DDUCKBOOST_WITH_CATBOOST=1
option(DUCKBOOST_WITH_XGBOOST "Link XGBoost for native training" OFF)
option(DUCKBOOST_WITH_LIGHTGBM "Link LightGBM for native training" OFF)
option(DUCKBOOST_WITH_CATBOOST "Link CatBoost for native training" OFF)

if(DUCKBOOST_WITH_XGBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_XGBOOST=1)
endif()
if(DUCKBOOST_WITH_LIGHTGBM)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_LIGHTGBM=1)
endif()
if(DUCKBOOST_WITH_CATBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_CATBOOST=1)
endif()

install(
  TARGETS duckboost_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
