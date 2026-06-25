# SECURITY.md - Security Model & Guidelines

This document details the security model, input validation policies, guardrail implementations, and pre-commit configurations for the Irrigation Optimizer.

## 1. Input Validation (Deterministic Backstop)

To prevent LLM hallucination and bypass attempts, all farmer profile inputs are validated deterministically in the workflow before downstream tools or LLMs are invoked.
- **Node**: `route_validation` runs the validation check `validate_profile_deterministically` independent of the LLM validator.
- **Crop Validation**: Crop types must match the supported list loaded dynamically from `crop_data.json` (e.g., wheat, maize, cotton, sugarcane, tomato, chickpea, groundnut). Unsupported crops (e.g., rice) are rejected.
- **Coordinate Validation**: Latitude must be between `-90.0` and `90.0`. Longitude must be between `-180.0` and `180.0`.
- **Field Size Validation**: Field size must be a positive number of hectares (`field_size_ha > 0.0`).
- **Planting Date Validation**: Planting date must be a valid date formatted as `YYYY-MM-DD`.

Any validation failure is intercepted by `route_validation`, setting `ctx.route = False` and routing the workflow to `clarify_profile` with detailed fields and friendly clarification questions.

## 2. Shell Execution Boundaries

Raw shell execution is strictly prohibited. The only shell execution permitted is the structured subprocess execution of the crop water balance calculator script:
- **Script**: `calc_schedule.py`
- **Execution Method**: `subprocess.run([sys.executable, script_path], input=json.dumps(payload), ...)`
- **Data Exchange**: Structured JSON passed strictly via standard input (stdin) and read from standard output (stdout). No shell parameters or arguments are constructed dynamically from user text.

## 3. Prompt Injection Protection

The workflow implements a dedicated `security_screen` node that intercepts inputs before reaching the recommendation LlmAgent.
- **Scoping**: It inspects strictly the **current turn's** latest user message, preventing early flags from locking down subsequent turns.
- **Matching**: User inputs are scanned using regular expressions for common injection patterns (e.g., "ignore all previous instructions", "system override", "you are now a...", "DAN mode").
- **Mitigation / Short-circuiting**: If an injection pattern is detected, the workflow sets `ctx.route = False` and routes to `canned_injection_response` which returns a safe canned response:
  > *\"I am sorry, but I cannot fulfill this request. I am only able to provide irrigation recommendations and answer related agricultural questions.\"*
  The injection attempt is logged as a warning (`Security Warning: Prompt injection detected`), and the request never reaches the LLM.

## 4. Output Guardrail (Daily Irrigation Ceiling)

A hard safety ceiling is enforced on the calculated irrigation depth to prevent overwatering, regardless of what the mathematical model yields:
- **Ceiling**: `MAX_DAILY_MM = 60.0` (60 mm of depth).
- **Location**: `compute_schedule` node in `agent.py`.
- **Action**: Any day's net irrigation depth exceeding 60.0 mm is clamped to 60.0 mm, and the liters required are recalculated accordingly.
- **Logging**: A warning is logged highlighting the date and the uncapped value:
  `logger.warning("Output Guardrail Triggered: Daily irrigation depth for date ... exceeded MAX_DAILY_MM ...")`

## 5. PII Storage (Consent Pattern)

Profile details (crop type, location coordinates, field size, planting date) constitute PII.
- **Consent Opt-In**: Checked via state flag `profile_saved_opt_in`. Users can opt in/out via text (e.g., "save my profile", "opt-in", "opt-out", "delete my profile") or JSON payloads.
- **Demonstration Scope**: Since the system does not have authentication/identity management, this is scoped as a demonstration of the consent pattern.
- **Persistence**:
  - If `profile_saved_opt_in` is `True`, the profile details are persisted to a local file (`saved_profiles.json`) keyed by the user/session ID.
  - If `False` or not present, the profile details are deleted/cleared from the persistent file, residing only in ephemeral session state (`ctx.state`).

## 6. Advisory Disclaimer

Every recommendation is appended with a mandatory advisory disclaimer:
- **Disclaimer**: *\"Advisory: This recommendation supplements, does not replace, local agricultural extension advice.\"*
- **Conditionality**: The disclaimer is appended to the explanation strictly on genuine recommendations. If a turn is security-blocked or routes to the canned injection response, the disclaimer is bypassed to avoid validating malicious prompts.

## 7. Pre-commit Configuration

Pre-commit validation is configured in `.pre-commit-config.yaml` to enforce standards:
- **Hooks**:
  - `end-of-file-fixer` (ensures files end with a newline).
  - `trailing-whitespace` (trims extra trailing whitespace).
  - Local `semgrep --config auto` hook (scans Python files for vulnerabilities and bad practices).
- **Installation**: Run `uv run pre-commit install` to install hooks locally.
- **Adherence**: Hooks must not be bypassed under any circumstances (no `--no-verify`).

## 8. Verification Tests

Test coverage verifies the outcomes of these security mechanisms:
- `tests/unit/test_security_guardrails.py` contains 5 dedicated outcome-based tests checking invalid crop rejection, negative/zero field size rejection, bad lat/lon rejection, daily depth cap enforcement, and prompt injection detection/disclaimer omission.
