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

# Core sources / docs (keep in-tree layout; dual-mode CMakeLists supports OOT builds)
cp -a "${SRC}/CMakeLists.txt" "${OUT}/"
cp -a "${SRC}/duckboost_extension.cpp" "${OUT}/"
cp -a "${SRC}/extension_config.cmake" "${OUT}/"
cp -a "${SRC}/LICENSE" "${OUT}/"
cp -a "${SRC}/README.md" "${OUT}/"
cp -a "${SRC}/PACKAGING.md" "${OUT}/"
cp -a "${SRC}/include" "${OUT}/"
cp -a "${SRC}/src" "${OUT}/"

# Keep test/sql/duckboost/... so read_text('test/sql/duckboost/data/...') paths work.
mkdir -p "${OUT}/test/sql"
if [[ -d "${ROOT}/test/sql/duckboost" ]]; then
	cp -a "${ROOT}/test/sql/duckboost" "${OUT}/test/sql/"
elif [[ -d "${SRC}/test" ]]; then
	cp -a "${SRC}/test/." "${OUT}/test/"
fi

# Community / template scaffolding
cp -a "${SRC}/community/oot/Makefile" "${OUT}/"
cp -a "${SRC}/community/oot/vcpkg.json" "${OUT}/"
cp -a "${SRC}/community/oot/.gitignore" "${OUT}/"
mkdir -p "${OUT}/.github/workflows"
cp -a "${SRC}/community/oot/.github/workflows/MainDistributionPipeline.yml" "${OUT}/.github/workflows/"
mkdir -p "${OUT}/docs"
cp -a "${SRC}/community/description.yml" "${OUT}/docs/community_extensions_description.yml"

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
