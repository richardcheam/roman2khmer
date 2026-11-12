# Releasing to PyPI

Releases are built and published by `.github/workflows/release.yml` when a tag `vX.Y.Z` is pushed. PyPI Trusted
Publishing lets GitHub prove which workflow is publishing, so no password or token is stored anywhere.

## One-time setup (the maintainer, in a browser)
1. **Accounts:** create accounts on https://pypi.org and https://test.pypi.org (separate sites), and turn on
   two-factor authentication on both.
2. **Pending publishers** (the project does not exist yet; this lets the first upload create it):
   - PyPI: Account → Publishing → "Add a new pending publisher" →
     PyPI project name `roman2khmer`, owner `richardcheam`, repository `roman2khmer`, workflow `release.yml`,
     environment `pypi`.
   - TestPyPI: the same at https://test.pypi.org/manage/account/publishing/, with environment `testpypi`.
   A pending publisher does not reserve the name; publish soon after adding it.
3. **GitHub environments:** repository → Settings → Environments → create `testpypi` and `pypi`. For `pypi`, add
   yourself as a required reviewer, so every real release waits for your click.

## Each release
1. Set the version in `pyproject.toml`, add a `CHANGELOG.md` entry, commit, push, and wait for the `tests` workflow.
2. `git tag v0.2.0 && git push origin v0.2.0` (the tag must equal the version).
3. The workflow builds once, publishes to TestPyPI, installs from there as a check, then waits for your approval
   in the `pypi` environment and publishes the same files to PyPI, with attestations.
4. Check https://pypi.org/project/roman2khmer/ and `pip install roman2khmer` in a fresh environment.

A version number can be used only once on PyPI; a broken release is fixed with a new version.
