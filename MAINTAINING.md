# Maintaining

Operational notes for maintainers of this repository. Contributors from the
outside should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## Cutting a PyPI release

Releases publish to PyPI (https://pypi.org/project/strands-decider/) via
[`.github/workflows/release-pypi.yml`](.github/workflows/release-pypi.yml).

The version is chosen at dispatch time; there is no need to bump
`pyproject.toml` first. `setuptools-scm` stamps the wheel with
`SETUPTOOLS_SCM_PRETEND_VERSION`, which the workflow sets from your input.

Steps:

1. Go to **Actions → release-pypi → Run workflow**.
2. Pick the branch to release from (usually `staging`).
3. Enter the version, e.g. `0.1.0`, `0.1.0.dev1`, or `0.1.0.rc1`. Format is
   `<int>.<int>.<int>` with an optional `.devN` / `.rcN` / `.aN` / `.bN`
   suffix. The workflow rejects anything else.
4. Run.
5. The **build** job builds sdist and wheel, then smoke-tests the wheel in a
   venv outside the repository (installs it, imports `strands_decider`, runs
   `strands-decider --help`, and confirms the installed version matches the
   input).
6. The **publish** job is gated by the `pypi` GitHub environment. Approve it
   when the build passes. Publication uses PyPI Trusted Publishing (OIDC);
   no API token is stored in the repository.

PyPI rejects re-uploading an existing version. Bump the number (or add a
`.devN` suffix) on every run.
