#!/usr/bin/env bash
# Extract extension/duckboost into a standalone DuckDB extension-template layout.
#
# Usage:
#   scripts/extract_duckboost_oot.sh [output_dir]
#
# Default output_dir: ./build/duckboost_oot
#
# After extraction:
#   cd "$OUT"
#   git init && git add . && git commit -m "Initial duckboost OOT extract"
#   # Add duckdb + extension-ci-tools submodules (or let `make` clone them)
#   make
# Then publish the repo and submit community/description.yml to duckdb/community-extensions.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/extension/duckboost"
OUT="${1:-${ROOT}/build/duckboost_oot}"

if [[ ! -d "${SRC}" ]]; then
	echo "error: missing ${SRC}" >&2
	exit 1
fi

echo "Extracting duckboost OOT layout -> ${OUT}"
rm -rf "${OUT}"
mkdir -p "${OUT}"

# Prefer content copy without failing on mount permission bits (e.g. artifact dirs).
copy_tree() {
	local from="$1"
	local to="$2"
	if cp -R --no-preserve=mode,ownership "${from}" "${to}" 2>/dev/null; then
		return 0
	fi
	cp -R "${from}" "${to}"
}

# Core sources / docs (keep in-tree layout; dual-mode CMakeLists supports OOT builds)
cp "${SRC}/CMakeLists.txt" "${OUT}/"
cp "${SRC}/duckboost_extension.cpp" "${OUT}/"
cp "${SRC}/extension_config.cmake" "${OUT}/"
cp "${SRC}/LICENSE" "${OUT}/"
cp "${SRC}/README.md" "${OUT}/"
cp "${SRC}/PACKAGING.md" "${OUT}/"
copy_tree "${SRC}/include" "${OUT}/include"
copy_tree "${SRC}/src" "${OUT}/src"

# Keep test/sql/duckboost/... so read_text('test/sql/duckboost/data/...') paths work.
mkdir -p "${OUT}/test/sql"
if [[ -d "${ROOT}/test/sql/duckboost" ]]; then
	copy_tree "${ROOT}/test/sql/duckboost" "${OUT}/test/sql/duckboost"
elif [[ -d "${SRC}/test" ]]; then
	copy_tree "${SRC}/test" "${OUT}/test"
fi

# Community / template scaffolding
cp "${SRC}/community/oot/Makefile" "${OUT}/"
cp "${SRC}/community/oot/vcpkg.json" "${OUT}/"
cp "${SRC}/community/oot/.gitignore" "${OUT}/"
mkdir -p "${OUT}/.github/workflows"
cp "${SRC}/community/oot/.github/workflows/MainDistributionPipeline.yml" "${OUT}/.github/workflows/"
mkdir -p "${OUT}/docs"
cp "${SRC}/community/description.yml" "${OUT}/docs/community_extensions_description.yml"

# Minimal README pointer for the standalone repo
cat > "${OUT}/docs/COMMUNITY_PUBLISH.md" <<'EOF'
# Publishing duckboost as a community extension

1. Push this repository to GitHub (public), e.g. `JavOrraca/duckboost`.
2. Note the commit SHA to publish.
3. Copy `docs/community_extensions_description.yml` to a fork of
   [duckdb/community-extensions](https://github.com/duckdb/community-extensions) as
   `extensions/duckboost/description.yml`, set `repo.ref` to that SHA, and open a PR.
4. After merge, users can:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

See also `PACKAGING.md` in the repository root.
EOF

# Record provenance
cat > "${OUT}/EXTRACT_SOURCE.txt" <<EOF
Extracted from ${ROOT}
Source path: extension/duckboost
Git describe: $(git -C "${ROOT}" describe --always --dirty 2>/dev/null || echo unknown)
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "OK: ${OUT}"
echo "Next: cd ${OUT} && git init && make   # builds via extension-ci-tools"
