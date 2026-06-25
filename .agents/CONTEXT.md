# Agent Context & Security Guidelines

This document outlines structural rules, security boundaries, and validation requirements for the Irrigation Optimizer agent codebase.

## Input Validation
- Every tool and node MUST validate inputs via explicit checks and/or Pydantic schemas rather than parsing raw, unvalidated dictionaries.
- Ensure all data types are verified, values are checked for sane ranges, and required fields are explicitly asserted.

## Shell Execution Limits
- No raw shell execution is allowed anywhere in the codebase outside of the explicit subprocess call in the `irrigation-calculator` skill script execution (which runs `calc_schedule.py`).
- All other operations must use native Python functions and APIs.

## Pre-Commit Hook Compliance
- On pre-commit hook failure, the agent/developer MUST fix the underlying issue and retry.
- NEVER bypass pre-commit checks using `--no-verify` or any other git commit bypass flag.

## TDD Planning Gate
- Future implementation plans must include a "Security Boundaries & Assertions" section.
- This section must explicitly list the validation checks, security boundaries, and assertions implemented in the design to prevent input-based and runtime security vulnerabilities.
