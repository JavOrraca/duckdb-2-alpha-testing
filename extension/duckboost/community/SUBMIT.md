# Submit duckboost to DuckDB community extensions

## Status

| Item | Value |
| --- | --- |
| Publish surface | orphan branch `cursor/duckboost-community-oot-0c09` |
| Repository | `JavOrraca/duckdb-2-alpha-testing` |
| Commit | `2bcdcae3d5fc8375b026d367da051a78416809db` |
| Descriptor | [`description.yml`](description.yml) |

Do **not** merge the orphan branch into `v2.0-cyanoptera`.

## One-shot submit

1. Fork [duckdb/community-extensions](https://github.com/duckdb/community-extensions).
2. Copy `extension/duckboost/community/description.yml` to
   `extensions/duckboost/description.yml` in the fork (that single file only).
3. Open a PR against `duckdb/community-extensions` with title like:
   `Add duckboost community extension`.
4. After merge + CI, install with:

```sql
INSTALL duckboost FROM community;
LOAD duckboost;
```

## Refreshing the publish surface

When duckboost sources change on `v2.0-cyanoptera`:

```bash
scripts/extract_duckboost_oot.sh /tmp/duckboost_oot
cd /tmp/duckboost_oot
git init -b cursor/duckboost-community-oot-0c09
git add -A && git commit -m "Refresh duckboost community extract"
git remote add origin https://github.com/JavOrraca/duckdb-2-alpha-testing.git
git push -f origin cursor/duckboost-community-oot-0c09
# Update community/description.yml repo.ref to the new SHA, then re-submit / bump
```

## Optional: dedicated repository

If you later create `JavOrraca/duckboost`, change `repo.github` in `description.yml`
to that repo and point `repo.ref` at its commit instead of the orphan branch.
