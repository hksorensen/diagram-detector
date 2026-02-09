# Pre-Commit Hooks Guide

## What Are Pre-Commit Hooks?

Automated checks that run **before** you commit code. They catch simple errors automatically, so you don't have to find them later in debugging sessions.

Think of it like spell-check for code - it stops typos and simple mistakes before they become problems.

---

## Tools Installed

### 1. **Ruff** - Fast Python Linter
Catches:
- ✅ Undefined variables (`undefined_variable`)
- ✅ Unused imports (`import os` but never used)
- ✅ Syntax errors (`if x = 5:` should be `==`)
- ✅ Unused variables (`port = ...` but never read)

**Auto-fixes:**
- Removes unused imports
- Fixes import order
- Trims whitespace

### 2. **Mypy** - Type Checker
Catches:
- ✅ Wrong attributes (`config.ssh_port` when it's `config.port`)
- ✅ Wrong argument types (`Path` passed where `int` expected)
- ✅ Function signature mismatches
- ✅ Missing return types

**Requires:** Type hints in your code to work best

### 3. **Basic Checks**
- Merge conflict markers (`<<<<<<< HEAD`)
- YAML syntax errors
- Large files (>500KB)
- Trailing whitespace

---

## How To Use

### Already Installed!
The hooks are active in this repo. They run automatically on `git commit`.

### Daily Workflow

```bash
# 1. Make your changes
vim diagram_detector/server.py

# 2. Try to commit (hooks run automatically)
git add diagram_detector/server.py
git commit -m "Fix server bug"

# If errors found:
# - Auto-fixable errors are fixed automatically
# - Non-fixable errors block the commit
# - Fix the errors, then commit again
```

### Example Output

```
ruff.....................................................................Failed
PRECOMMIT_DEMO.py:11: F821 Undefined name `undefined_variable`
Found 2 errors (1 fixed, 1 remaining).
```

**What this means:**
- ❌ Commit blocked - you have errors
- ✅ 1 error auto-fixed (like unused import)
- ❌ 1 error needs manual fix
- Fix it and try committing again

### Run Manually (Without Committing)

```bash
# Check all files
pre-commit run --all-files

# Check only staged files
pre-commit run

# Check specific file
pre-commit run --files diagram_detector/server.py
```

### Skip Hooks (Emergency Only!)

```bash
# Skip all hooks (NOT recommended)
git commit --no-verify -m "Emergency fix"
```

---

## What Gets Caught?

### ✅ Would Have Caught Today's Bugs

1. **Undefined logger**
   ```python
   logger.info("message")  # Without import
   # Ruff: F821 undefined name 'logger'
   ```

2. **Wrong attribute name**
   ```python
   config.ssh_port  # Should be config.port
   # Mypy: RemoteConfig has no attribute 'ssh_port'
   ```

3. **Wrong function signature**
   ```python
   convert_pdf_to_images(pdf, output_dir, dpi=200)
   # Mypy: Too many positional arguments
   ```

4. **Unused imports**
   ```python
   import subprocess  # Never used
   # Ruff: F401 unused import (auto-removed)
   ```

### ❌ Won't Catch (Need Tests)

- Logic errors (checking wrong directory)
- Runtime errors (file not found)
- Integration issues (SSH timeouts)
- Data-dependent bugs

---

## Configuration Files

### `.pre-commit-config.yaml`
Defines which hooks run and their settings.

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]  # Auto-fix when possible
```

### `pyproject.toml` (ruff section)
```toml
[tool.ruff]
line-length = 120  # Max line length

[tool.ruff.lint]
select = ["E", "F", "I"]  # Which checks to run
ignore = ["E501"]  # Which to skip
```

### `pyproject.toml` (mypy section)
```toml
[tool.mypy]
python_version = "3.9"
ignore_missing_imports = true  # Don't complain about external packages
disallow_untyped_defs = false  # Don't require type hints everywhere
```

---

## Updating Hooks

```bash
# Update to latest versions
pre-commit autoupdate

# Reinstall after config changes
pre-commit install
```

---

## Disabling Specific Checks

### Temporarily (per line)
```python
port = config.ssh_port  # type: ignore  # Mypy: ignore this line
unused_var = 5  # noqa: F841  # Ruff: ignore this line
```

### Permanently (in config)
```toml
# pyproject.toml
[tool.ruff.lint]
ignore = ["E501", "F841"]  # Skip these checks globally
```

---

## Command-Line Usage

### Ruff Standalone
```bash
# Check for errors
ruff check diagram_detector/

# Auto-fix errors
ruff check --fix diagram_detector/

# Format code
ruff format diagram_detector/
```

### Mypy Standalone
```bash
# Check types
mypy diagram_detector/

# Check specific file
mypy diagram_detector/server.py

# Ignore missing imports
mypy --ignore-missing-imports diagram_detector/
```

---

## Troubleshooting

### Hook Fails With "No module named..."
```bash
# Reinstall hooks environment
pre-commit clean
pre-commit install --install-hooks
```

### Want To Disable A Hook Temporarily
Edit `.pre-commit-config.yaml` and comment out the hook:
```yaml
#  - repo: https://github.com/pre-commit/mirrors-mypy
#    hooks:
#      - id: mypy
```

### Mypy Too Strict
Set `disallow_untyped_defs = false` in `pyproject.toml` (already done).

### Too Many Warnings
Add specific rules to ignore in `pyproject.toml`:
```toml
[tool.ruff.lint]
ignore = ["E501", "F841", "...]
```

---

## Learning Resources

- **Ruff docs**: https://docs.astral.sh/ruff/
- **Mypy docs**: https://mypy.readthedocs.io/
- **Pre-commit**: https://pre-commit.com/

---

## Summary

**One-time setup:** ✅ Done (hooks installed)

**Daily use:**
1. Code normally
2. `git commit`
3. Hooks run automatically
4. Fix any errors it finds
5. Commit succeeds

**Result:** Fewer bugs reach the codebase!

---

**Remember:** These tools catch **simple errors** (typos, wrong types). You still need:
- Manual testing for logic
- Integration tests for complex interactions
- Code review for architecture decisions
