cmake_minimum_required(VERSION 2.8.12...3.29)

project(DuckboostExtension)

include_directories(include)

set(DUCKBOOST_EXTENSION_FILES
    duckboost_extension.cpp
    src/model.cpp
    src/trainer.cpp
    src/native_train.cpp
    src/import.cpp
    src/duckboost_functions.cpp)

build_extension_library(duckboost ${DUCKBOOST_EXTENSION_FILES})

# Optional native library backends:
#   -DDUCKBOOST_WITH_XGBOOST=ON -DXGBOOST_ROOT=/path
#   -DDUCKBOOST_WITH_LIGHTGBM=ON -DLIGHTGBM_ROOT=/path
#   -DDUCKBOOST_WITH_CATBOOST=ON -DCATBOOST_ROOT=/path
# Stub-only (compile #ifdef paths without linking vendor libs):
#   -DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_NATIVE_STUB_ONLY=ON
option(DUCKBOOST_WITH_XGBOOST "Enable XGBoost native trainer" OFF)
option(DUCKBOOST_WITH_LIGHTGBM "Enable LightGBM native trainer" OFF)
option(DUCKBOOST_WITH_CATBOOST "Enable CatBoost native trainer" OFF)
option(DUCKBOOST_NATIVE_STUB_ONLY "Compile native trainer stubs without linking vendor libs" OFF)

if(DUCKBOOST_WITH_XGBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_XGBOOST=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_XGBOOST enabled (stub only)")
  else()
    find_path(XGBOOST_INCLUDE_DIR NAMES xgboost/c_api.h HINTS ${XGBOOST_ROOT} ENV XGBOOST_ROOT PATH_SUFFIXES include)
    find_library(XGBOOST_LIBRARY NAMES xgboost HINTS ${XGBOOST_ROOT} ENV XGBOOST_ROOT PATH_SUFFIXES lib lib64)
    if(NOT XGBOOST_INCLUDE_DIR OR NOT XGBOOST_LIBRARY)
      message(FATAL_ERROR "duckboost: DUCKBOOST_WITH_XGBOOST=ON but XGBoost not found (set XGBOOST_ROOT or use "
                          "DUCKBOOST_NATIVE_STUB_ONLY=ON)")
    endif()
    target_include_directories(duckboost_extension PRIVATE ${XGBOOST_INCLUDE_DIR})
    target_link_libraries(duckboost_extension PRIVATE ${XGBOOST_LIBRARY})
    message(STATUS "duckboost: linking XGBoost (${XGBOOST_LIBRARY})")
  endif()
endif()

if(DUCKBOOST_WITH_LIGHTGBM)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_LIGHTGBM=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_LIGHTGBM enabled (stub only)")
  else()
    find_path(LIGHTGBM_INCLUDE_DIR NAMES LightGBM/c_api.h HINTS ${LIGHTGBM_ROOT} ENV LIGHTGBM_ROOT PATH_SUFFIXES include)
    find_library(LIGHTGBM_LIBRARY NAMES _lightgbm lib_lightgbm lightgbm HINTS ${LIGHTGBM_ROOT} ENV LIGHTGBM_ROOT
                 PATH_SUFFIXES lib lib64)
    if(NOT LIGHTGBM_INCLUDE_DIR OR NOT LIGHTGBM_LIBRARY)
      message(FATAL_ERROR "duckboost: DUCKBOOST_WITH_LIGHTGBM=ON but LightGBM not found (set LIGHTGBM_ROOT or use "
                          "DUCKBOOST_NATIVE_STUB_ONLY=ON)")
    endif()
    target_include_directories(duckboost_extension PRIVATE ${LIGHTGBM_INCLUDE_DIR})
    target_link_libraries(duckboost_extension PRIVATE ${LIGHTGBM_LIBRARY})
    message(STATUS "duckboost: linking LightGBM (${LIGHTGBM_LIBRARY})")
  endif()
endif()

if(DUCKBOOST_WITH_CATBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_CATBOOST=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_CATBOOST enabled (stub only)")
  else()
    find_path(CATBOOST_INCLUDE_DIR NAMES catboost/libs/model_interface/c_api.h c_api.h HINTS ${CATBOOST_ROOT}
              ENV CATBOOST_ROOT PATH_SUFFIXES include)
    find_library(CATBOOST_LIBRARY NAMES catboostmodel catboost HINTS ${CATBOOST_ROOT} ENV CATBOOST_ROOT
                 PATH_SUFFIXES lib lib64)
    if(NOT CATBOOST_INCLUDE_DIR OR NOT CATBOOST_LIBRARY)
      message(FATAL_ERROR "duckboost: DUCKBOOST_WITH_CATBOOST=ON but CatBoost not found (set CATBOOST_ROOT or use "
                          "DUCKBOOST_NATIVE_STUB_ONLY=ON)")
    endif()
    target_include_directories(duckboost_extension PRIVATE ${CATBOOST_INCLUDE_DIR})
    target_link_libraries(duckboost_extension PRIVATE ${CATBOOST_LIBRARY})
    message(STATUS "duckboost: linking CatBoost (${CATBOOST_LIBRARY})")
  endif()
endif()

install(
  TARGETS duckboost_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
