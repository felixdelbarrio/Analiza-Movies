# Security Audit Report

## Executive Summary

Audit date: 2026-03-14

I reviewed the current FastAPI backend, React frontend integration points, and GitHub Actions workflow posture with a focus on the active PR checks and CodeQL findings. The PR conversation currently shows GitHub Advanced Security review events only; there are no human review comments on PR #152 at this time.

I fixed the security findings that were actionable in code:

1. Arbitrary file deletion from client-controlled paths.
2. SPA static file path traversal risk.
3. Clear-text persistence of Plex and OMDb secrets in runtime JSON.
4. Excessive exception detail exposure in API responses.
5. Missing explicit GitHub Actions permissions in the desktop build workflow.

## High Severity

### SEC-001: Client-controlled file deletion

- Severity: High
- Location: `src/server/api/services/file_actions.py:17-129`
- Evidence:
  - The delete flow now authorizes deletion only if the requested file exists in application-generated reports loaded from `report_filtered.csv` / `report_all.csv`.
  - The client-provided `file` value is no longer turned directly into an executable filesystem path for deletion.
- Impact:
  - Before the fix, a malicious caller could submit arbitrary paths to `/actions/delete` and attempt to delete files outside the app’s intended scope.
- Fix:
  - Added `_load_authorized_files(...)` and changed `delete_files_from_rows(...)` to delete only trusted paths materialized from report files.
  - Added endpoint regression tests in `tests/test_server_actions_endpoints.py`.

### SEC-002: SPA asset path traversal

- Severity: High
- Location: `src/server/api/app.py:165-182`, `src/server/api/app.py:220-232`
- Evidence:
  - Added `_safe_spa_asset_path(...)` to normalize the requested path, reject `..`, reject absolute paths, and enforce `candidate.relative_to(dist_root)`.
- Impact:
  - Without this guard, a crafted path could try to escape `web/dist` and serve unexpected local files.
- Fix:
  - All asset serving now goes through `_safe_spa_asset_path(...)`.
  - Added regression coverage in `tests/test_server_health_endpoints.py`.

### SEC-003: Clear-text storage of runtime secrets

- Severity: High
- Location: `src/shared/runtime_profiles.py:215-317`, `src/server/api/services/runtime_secrets.py:1-58`
- Evidence:
  - `RuntimeConfig.to_dict(..., include_secrets=False)` now strips `omdb_api_keys` and `plex_token` from persisted payloads.
  - `save_runtime_config(...)` writes only the redacted form.
  - Runtime secrets are now held in memory via `runtime_secrets.py`.
- Impact:
  - Persisting Plex tokens and OMDb keys in JSON exposed them to disk disclosure and triggered CodeQL.
- Fix:
  - Introduced in-memory secret handling for Plex tokens and OMDb keys.
  - Added regression assertions in `tests/test_shared_runtime_profiles.py`.

## Medium Severity

### SEC-004: Internal exception details returned to API callers

- Severity: Medium
- Location: `src/server/api/routers/configuration.py:163-229`, `src/server/api/routers/meta.py:56-68`, `src/server/api/routers/omdb.py:15-19`, `src/server/api/routers/wiki.py:15-19`
- Evidence:
  - Replaced `detail=f"...{exc!r}"` and raw `repr(exc)` style responses with generic client messages plus server-side logging.
- Impact:
  - Raw exception text can reveal internal module names, paths, or implementation details that help an attacker.
- Fix:
  - API responses now return generic failure messages while preserving diagnostics in server logs.

### SEC-005: Workflow token permissions not explicitly constrained

- Severity: Medium
- Location: `.github/workflows/desktop-build.yml:9-10`
- Evidence:
  - Added explicit `permissions: contents: read`.
- Impact:
  - Relying on repository defaults makes the workflow less portable and weaker under least-privilege review.
- Fix:
  - Constrained the workflow token to read-only contents access.

## Additional Validation

- `ruff check .` passed.
- `black --check .` passed.
- `mypy src/backend src/desktop src/server src/shared` passed.
- `pyright -p pyrightconfig.json` passed.
- `pytest -q` passed with 83 tests.
- `npm --prefix web run build` passed.
- `python -m build` passed.
- Wheel/sdist validation confirmed `backend/py.typed`, `desktop/py.typed`, `server/api/services/py.typed`, and `shared/py.typed`.

## Residual Risk / Notes

- CodeQL itself was not run locally; the fixes above target the currently open alerts reported by GitHub.
- Runtime OMDb/Plex secrets are no longer persisted to disk. This improves security posture but means persistence now depends on environment variables or the live application session.
- I did not find React app-side raw HTML sinks such as `dangerouslySetInnerHTML`, `innerHTML`, or `eval` in the application source under `web/src`.
