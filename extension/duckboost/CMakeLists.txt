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

# Optional native library backends (C API train → dump → duckboost_import):
#   -DDUCKBOOST_WITH_XGBOOST=ON -DXGBOOST_ROOT=/path
#   -DDUCKBOOST_WITH_LIGHTGBM=ON -DLIGHTGBM_ROOT=/path
#   -DDUCKBOOST_WITH_CATBOOST=ON  (compile flag only; no public train C API)
# Stub-only (compile #ifdef paths without linking vendor libs):
#   -DDUCKBOOST_WITH_XGBOOST=ON -DDUCKBOOST_NATIVE_STUB_ONLY=ON
#
# Pip wheels often ship the shared library without headers; duckboost embeds the
# small C API declarations it needs, so only the library must be found.
option(DUCKBOOST_WITH_XGBOOST "Enable XGBoost native trainer" OFF)
option(DUCKBOOST_WITH_LIGHTGBM "Enable LightGBM native trainer" OFF)
option(DUCKBOOST_WITH_CATBOOST "Enable CatBoost native trainer compile flag" OFF)
option(DUCKBOOST_NATIVE_STUB_ONLY "Compile native trainer stubs without linking vendor libs" OFF)

if(DUCKBOOST_WITH_XGBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_XGBOOST=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_XGBOOST enabled (stub only)")
  else()
    find_library(XGBOOST_LIBRARY NAMES xgboost HINTS ${XGBOOST_ROOT} ENV XGBOOST_ROOT PATH_SUFFIXES lib lib64)
    if(NOT XGBOOST_LIBRARY)
      file(GLOB XGBOOST_LIBRARY_CANDIDATES
           "$ENV{HOME}/.local/lib/python*/site-packages/xgboost/lib/libxgboost.so"
           "/usr/local/lib/libxgboost.so"
           "/usr/lib/libxgboost.so")
      if(XGBOOST_LIBRARY_CANDIDATES)
        list(GET XGBOOST_LIBRARY_CANDIDATES 0 XGBOOST_LIBRARY)
      endif()
    endif()
    if(NOT XGBOOST_LIBRARY)
      message(FATAL_ERROR "duckboost: DUCKBOOST_WITH_XGBOOST=ON but libxgboost not found (set XGBOOST_ROOT or use "
                          "DUCKBOOST_NATIVE_STUB_ONLY=ON)")
    endif()
    get_filename_component(XGBOOST_LIB_DIR "${XGBOOST_LIBRARY}" DIRECTORY)
    # Pip wheels ship libxgboost.so with SONAME libxgboost.so.3; stage a soname link for the linker.
    set(DUCKBOOST_XGB_STAGE "${CMAKE_BINARY_DIR}/duckboost_native_libs")
    file(MAKE_DIRECTORY "${DUCKBOOST_XGB_STAGE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${XGBOOST_LIBRARY}"
                            "${DUCKBOOST_XGB_STAGE}/libxgboost.so")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${XGBOOST_LIBRARY}"
                            "${DUCKBOOST_XGB_STAGE}/libxgboost.so.3")
    # build_extension_library uses the plain target_link_libraries signature.
    target_link_libraries(duckboost_extension "${DUCKBOOST_XGB_STAGE}/libxgboost.so")
    set_property(TARGET duckboost_extension APPEND PROPERTY BUILD_RPATH "${DUCKBOOST_XGB_STAGE};${XGBOOST_LIB_DIR}")
    set_property(TARGET duckboost_extension APPEND PROPERTY INSTALL_RPATH "${XGBOOST_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${DUCKBOOST_XGB_STAGE};${XGBOOST_LIB_DIR}")
    link_directories("${DUCKBOOST_XGB_STAGE}" "${XGBOOST_LIB_DIR}")
    message(STATUS "duckboost: linking XGBoost (${XGBOOST_LIBRARY})")
  endif()
endif()

if(DUCKBOOST_WITH_LIGHTGBM)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_LIGHTGBM=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_LIGHTGBM enabled (stub only)")
  else()
    find_library(LIGHTGBM_LIBRARY NAMES _lightgbm lib_lightgbm lightgbm HINTS ${LIGHTGBM_ROOT} ENV LIGHTGBM_ROOT
                 PATH_SUFFIXES lib lib64)
    if(NOT LIGHTGBM_LIBRARY)
      file(GLOB LIGHTGBM_LIBRARY_CANDIDATES
           "$ENV{HOME}/.local/lib/python*/site-packages/lightgbm/lib/lib_lightgbm.so"
           "/usr/local/lib/lib_lightgbm.so"
           "/usr/lib/lib_lightgbm.so")
      if(LIGHTGBM_LIBRARY_CANDIDATES)
        list(GET LIGHTGBM_LIBRARY_CANDIDATES 0 LIGHTGBM_LIBRARY)
      endif()
    endif()
    if(NOT LIGHTGBM_LIBRARY)
      message(FATAL_ERROR "duckboost: DUCKBOOST_WITH_LIGHTGBM=ON but lib_lightgbm not found (set LIGHTGBM_ROOT or use "
                          "DUCKBOOST_NATIVE_STUB_ONLY=ON)")
    endif()
    target_link_libraries(duckboost_extension ${LIGHTGBM_LIBRARY})
    get_filename_component(LIGHTGBM_LIB_DIR "${LIGHTGBM_LIBRARY}" DIRECTORY)
    set_property(TARGET duckboost_extension APPEND PROPERTY BUILD_RPATH "${LIGHTGBM_LIB_DIR}")
    set_property(TARGET duckboost_extension APPEND PROPERTY INSTALL_RPATH "${LIGHTGBM_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "${CMAKE_BUILD_RPATH};${LIGHTGBM_LIB_DIR}")
    link_directories("${LIGHTGBM_LIB_DIR}")
    message(STATUS "duckboost: linking LightGBM (${LIGHTGBM_LIBRARY})")
  endif()
endif()

if(DUCKBOOST_WITH_CATBOOST)
  target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_WITH_CATBOOST=1)
  if(DUCKBOOST_NATIVE_STUB_ONLY)
    target_compile_definitions(duckboost_extension PRIVATE DUCKBOOST_NATIVE_STUB=1)
    message(STATUS "duckboost: DUCKBOOST_WITH_CATBOOST enabled (stub only)")
  else()
    # CatBoost has no public training C API; keep the compile flag for capability reporting.
    message(STATUS "duckboost: DUCKBOOST_WITH_CATBOOST enabled (import-only; no train C API)")
  endif()
endif()

install(
  TARGETS duckboost_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
