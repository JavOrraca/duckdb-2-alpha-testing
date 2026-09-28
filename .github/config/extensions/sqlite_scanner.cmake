# Static linking on windows does not properly work due to symbol collision
if (WIN32)
	set(STATIC_LINK_SQLITE "DONT_LINK")
	# Attach/storage tests are unreliable on Windows file locking.
	set(LOAD_SQLITE_TESTS "")
else ()
	set(STATIC_LINK_SQLITE "")
	set(LOAD_SQLITE_TESTS "LOAD_TESTS")
endif()
duckdb_extension_load(sqlite_scanner
		${STATIC_LINK_SQLITE} ${LOAD_SQLITE_TESTS}
		GIT_URL https://github.com/duckdb/duckdb-sqlite
		GIT_TAG 9bc53cf6552461da57b2dad25f35090136633370
		SUBMODULES database-connector
		)
