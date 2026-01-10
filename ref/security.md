# Security Model Reference

Security boundaries and threat model for the MCP Slicer Bridge.

## What This Is

- Educational/research tool for **localhost only**
- No encryption, no authentication, no multi-user support
- `execute_python` runs arbitrary code by design
- Designed for de-identified research data in controlled environments

## What This Is NOT

- Clinical/production system
- HIPAA/GDPR compliant
- Suitable for patient data
- Safe for remote access

## Threat Model (MVP Scope)

**Assumptions:**
- Attacker has no network access (localhost only)
- User runs trusted code (no malicious prompts)
- Environment is controlled (personal machine)

**Out of Scope:**
- Network attacks (no remote access)
- Multi-user scenarios (single user)
- Clinical data protection (de-identified research data only)

**In Scope:**
- Accidental data corruption (provide undo guidance)
- Resource exhaustion (implement timeouts)
- Error handling (prevent information leakage)

## Security Controls (MVP)

| Control | Implementation |
|---------|----------------|
| Network isolation | Slicer WebServer binds to localhost only |
| Timeouts | 30-second HTTP timeout prevents hanging |
| Error sanitization | Don't expose file paths in error messages |
| Audit logging | All `execute_python` calls logged with timestamp, code hash |
| Input validation | Node IDs (ASCII, 256 chars), segment names (NFKC, 256 chars) |
| Circuit breaker | Prevents hammering unresponsive Slicer |

## Security Limitations

1. **Code Execution**: `execute_python` executes arbitrary code in Slicer
   - No sandboxing or code validation
   - Full access to Slicer's Python environment
   - Can modify or delete scene data

2. **Authentication**: No authentication mechanism
   - WebServer connection is unauthenticated
   - Assumes trusted local environment

3. **Data Privacy**: No encryption or privacy controls
   - Medical imaging data transmitted over localhost HTTP
   - Suitable only for de-identified research data

## Audit Logging

All `execute_python` calls are logged (if `SLICER_AUDIT_LOG` env var set):
- Timestamp
- Code hash (SHA-256)
- Execution result

This is for debugging and research reproducibility, not security enforcement.

## Production Roadmap

For clinical/production use, implement:
- [ ] Authentication (API keys, OAuth)
- [ ] Authorization (role-based access control)
- [ ] Code execution sandboxing (whitelist allowed operations)
- [ ] Data encryption (TLS for localhost)
- [ ] Tamper-proof audit logging
- [ ] HIPAA compliance (BAA, PHI handling)

## Recommendations

| Environment | Suitable? |
|-------------|-----------|
| Personal research machine | Yes |
| Shared lab workstation | No (no multi-user) |
| Hospital network | No (no HIPAA compliance) |
| Cloud deployment | No (no remote auth) |
| Educational demo | Yes (with de-identified data) |
