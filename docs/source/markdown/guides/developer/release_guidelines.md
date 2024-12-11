# {octicon}`rocket` Release Process Guide

This document outlines the release process for our Python package, including environment setup, workflow configuration, and step-by-step release instructions.

## {octicon}`checklist` Prerequisites

### {octicon}`key` Required Tokens

:::{dropdown} Test PyPI Token
:animate: fade-in-slide-down

1. Create account on [Test PyPI](https://test.pypi.org)
2. Generate API token: Account Settings → API tokens
3. Scope: Upload to project
4. Save as `TEST_PYPI_TOKEN` in GitHub Actions secrets
   :::

:::{dropdown} Production PyPI Token
:animate: fade-in-slide-down

1. Create account on [PyPI](https://pypi.org)
2. Generate API token: Account Settings → API tokens
3. Scope: Upload to project
4. Save as `PYPI_TOKEN` in GitHub Actions secrets
   :::

:::{dropdown} Codecov Token
:animate: fade-in-slide-down

1. Create account on [Codecov](https://codecov.io)
2. Generate repository upload token
3. Scope: Repository-specific access
4. Save as `CODECOV_TOKEN` in GitHub Actions secrets
   :::

### {octicon}`repo` GitHub Repository Setup

:::{dropdown} Branch Protection
:animate: fade-in-slide-down

```text
main branch:
☑️ Require pull request reviews
☑️ Require status checks to pass
☑️ Require linear history
☑️ Include administrators
```

:::

:::{dropdown} Required Status Checks
:animate: fade-in-slide-down

- Unit Tests
- Integration Tests
- Security Scans
- Quality Checks
  :::

## {octicon}`gear` Environment Setup

### {octicon}`project` 1. Create GitHub Environments

Create the following environments in your repository:

1. Go to Repository Settings → Environments → New environment

:::{dropdown} Staging Environment
:animate: fade-in-slide-down

```yaml
Name: staging
Protection rules: None (allows automated RC deployments)
```

:::

:::{dropdown} QA Environment
:animate: fade-in-slide-down

```yaml
Name: qa
Protection rules:
  - Required reviewers: [QA team members]
  - Wait timer: 0 minutes
  - Deployment branches: main
Environment secrets: None
```

:::

:::{dropdown} Production Release Environment
:animate: fade-in-slide-down

```yaml
Name: production-release
Protection rules:
  - Required reviewers: [Release managers]
  - Wait timer: 30 minutes
  - Deployment branches: main
Environment secrets: None
```

:::

:::{dropdown} Production Deploy Environment
:animate: fade-in-slide-down

```yaml
Name: production-deploy
Protection rules:
  - Required reviewers: [Senior engineers, DevOps]
  - Wait timer: 0 minutes
  - Deployment branches: main
Environment secrets:
  - PYPI_TOKEN: [Production PyPI token]
```

:::

### {octicon}`key-asterisk` 2. Configure Repository Secrets

Go to Repository Settings → Secrets and variables → Actions → New repository secret

```yaml
TEST_PYPI_TOKEN: [Test PyPI token]
PYPI_TOKEN: [Production PyPI token]
```

## {octicon}`versions` Release Types

### {octicon}`git-branch` Release Candidate (RC)

- Format: `vX.Y.Z-rcN`
- Examples: `v1.2.3-rc1`, `v1.2.3-rc2`
- Purpose: Testing and validation
- Deploys to: Test PyPI

### {octicon}`git-merge` Production Release

- Format: `vX.Y.Z`
- Examples: `v1.2.3`, `v2.0.0`
- Purpose: Production deployment
- Deploys to: Production PyPI

## {octicon}`workflow` Release Process

### {octicon}`pencil` 1. Prepare Release

1. Update version in package files:

   ```python
   # src/__init__.py or similar
   __version__ = "1.2.3"
   ```

2. Update CHANGELOG.md:

   ```markdown
   ## [1.2.3] - YYYY-MM-DD

   ### Added

   - New feature X

   ### Changed

   - Modified behavior Y

   ### Fixed

   - Bug fix Z
   ```

3. Create release branch:

   ```bash
   git checkout -b release/v1.2.3
   git add .
   git commit -m "chore: prepare release v1.2.3"
   git push origin release/v1.2.3
   ```

4. Create and merge PR to main

### {octicon}`git-branch` 2. Create Release Candidate

```bash
# Ensure you're on main and up-to-date
git checkout main
git pull origin main

# Create and push RC tag
git tag v1.2.3-rc1
git push origin v1.2.3-rc1
```

### {octicon}`checklist` 3. RC Validation Process

:::{dropdown} Automated Checks
:animate: fade-in-slide-down

- Quality checks
- Security scans
- Test suite
- Build verification
  :::

:::{dropdown} RC Deployment
:animate: fade-in-slide-down

- Automatic upload to Test PyPI
- Creates draft GitHub release
  :::

:::{dropdown} QA Validation
:animate: fade-in-slide-down

- Install from Test PyPI
- Run smoke tests
- Validate functionality
  :::

:::{dropdown} Review Approvals
:animate: fade-in-slide-down

- QA team approves in QA environment
- Release managers approve in production-release environment
  :::

### {octicon}`git-merge` 4. Production Release

After successful RC validation:

```bash
# Create and push production tag
git tag v1.2.3
git push origin v1.2.3
```

The workflow will:

1. Run all checks
2. Create GitHub release
3. Await approvals
4. Deploy to PyPI

## {octicon}`bug` Troubleshooting

### {octicon}`alert` Common Issues

:::{dropdown} RC Upload Fails
:animate: fade-in-slide-down

- Check Test PyPI token permissions
- Verify version doesn't exist on Test PyPI
- Ensure version follows PEP 440
  :::

:::{dropdown} Workflow Failures
:animate: fade-in-slide-down

```bash
# View workflow logs
gh run list
gh run view [run-id]
```

:::

:::{dropdown} Environment Approval Issues
:animate: fade-in-slide-down

- Verify reviewer permissions
- Check environment protection rules
- Ensure reviewers are in correct teams
  :::

### {octicon}`sync` Recovery Steps

:::{dropdown} Failed RC
:animate: fade-in-slide-down

```bash
# Delete failed RC tag
git tag -d v1.2.3-rc1
git push origin :refs/tags/v1.2.3-rc1

# Create new RC
git tag v1.2.3-rc2
git push origin v1.2.3-rc2
```

:::

:::{dropdown} Failed Production Release
:animate: fade-in-slide-down

- Do not delete production tags
- Create new patch version if needed
  :::

## {octicon}`light-bulb` Best Practices

### {octicon}`versions` Version Management

- Follow semantic versioning
- Use RCs for significant changes
- Include build metadata in package

### {octicon}`note` Release Notes

- Use consistent format
- Include upgrade instructions
- Document breaking changes

### {octicon}`shield-lock` Security

- Never share or commit tokens
- Review dependency updates
- Monitor security advisories

### {octicon}`megaphone` Communication

- Announce release schedule
- Document known issues
- Maintain changelog

## {octicon}`bookmark` Quick Reference

### {octicon}`terminal` Commands Cheatsheet

```bash
# Create RC
git tag v1.2.3-rc1
git push origin v1.2.3-rc1

# Create Release
git tag v1.2.3
git push origin v1.2.3

# Delete Tag (only for RCs)
git tag -d v1.2.3-rc1
git push origin :refs/tags/v1.2.3-rc1

# View Tags
git tag -l "v*"

# Check Workflow Status
gh workflow list
gh run list
```

### {octicon}`file-directory` Required Files

```text
repository/
├── .github/
│   ├── workflows/
│   │   └── release.yaml
│   └── actions/
├── src/
│   └── __init__.py
├── tests/
├── CHANGELOG.md
└── pyproject.toml
```
