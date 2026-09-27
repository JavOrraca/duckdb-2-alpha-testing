#include "duckboost/native_train.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {
namespace duckboost {

bool NativeTrainerCompiled(BoostBackend backend) {
	switch (backend) {
	case BoostBackend::XGBOOST:
#if defined(DUCKBOOST_WITH_XGBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::LIGHTGBM:
#if defined(DUCKBOOST_WITH_LIGHTGBM)
		return true;
#else
		return false;
#endif
	case BoostBackend::CATBOOST:
#if defined(DUCKBOOST_WITH_CATBOOST)
		return true;
#else
		return false;
#endif
	case BoostBackend::REFERENCE:
	default:
		return false;
	}
}

BoostModel TrainNative(const vector<double> &y, const vector<vector<double>> &x, const TrainOptions &options) {
	(void)y;
	(void)x;
	if (!NativeTrainerCompiled(options.backend)) {
		throw NotImplementedException(
		    "duckboost: native training for backend '%s' is not linked in this build. "
		    "Configure with -DDUCKBOOST_WITH_%s=ON (and install the vendor library), "
		    "or use backend='reference' / duckboost_import().",
		    BackendToString(options.backend), StringUtil::Upper(BackendToString(options.backend)));
	}

	// Linked/stub builds: C API bridge is the next step once vendor libs are available.
	// Intended flow: train with vendor C API → dump model → ImportXGBoostJSON / ImportLightGBMText /
	// ImportCatBoostJSON into BoostModel.
	throw NotImplementedException(
	    "duckboost: native trainer for backend '%s' is compiled into this build but the vendor C API bridge "
	    "is not implemented yet. Use duckboost_import() with a vendor dump, or backend='reference'.",
	    BackendToString(options.backend));
}

} // namespace duckboost
} // namespace duckdb
