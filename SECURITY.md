# Security Policy

## Overview

The 3D Slicer MCP Bridge enables Claude Code to interact with 3D Slicer's medical imaging environment. This document outlines security considerations and best practices.

## Threat Model

### Intended Use Case
- Local development and educational environments
- Single-user workstations
- Non-clinical, non-production use

### Trust Assumptions
- The user running Claude Code is trusted
- The local network is trusted (localhost communication)
- 3D Slicer installation is legitimate and unmodified

## Security Features

### Input Validation
- **MRML Node IDs**: Validated with strict regex pattern `^[a-zA-Z][a-zA-Z0-9_]*$` (ASCII only)
- **Segment Names**: Validated with Unicode-aware pattern `^[\w\s\-]+$` (see Unicode Support below)
- **Whitespace Normalization**: Segment names are normalized to prevent edge cases

### Unicode Support
Medical terminology frequently uses international characters. Segment names support:

**Accepted Characters:**
- Greek letters: α, β, γ, δ, μ, λ, ω (e.g., "α-fetoprotein", "β-amyloid")
- Accented characters: é, ñ, ü, ö (e.g., "Müller cells", "señal region")
- Cyrillic: А-Яа-я (e.g., "Мозг region")
- CJK characters: 中文, 日本語, 한국어

**Rejected Characters (Security):**
- Emoji: ❤, 🧠, 😀 (not word characters)
- Shell metacharacters: ; ` $ | & < >
- Quotes: ' "
- Brackets: [ ] { } ( )
- Special symbols: @ # % ^ * = + \ /

**Rationale:** The `\w` regex with Unicode flag matches word characters from any language while blocking characters that could enable code injection. This balances usability for international medical collaboration with security.

### Code Injection Prevention
- All user inputs are validated before use in Python code generation
- `json.dumps()` is used for defense-in-depth string escaping
- Generated Python code uses variables instead of direct string interpolation

### Audit Logging
- All Python code executions are logged with timestamps and request IDs
- Code hash computed for identification of repeated executions
- Large code blocks are truncated with hash for reference
- Audit log path is validated against forbidden system directories

### Error Handling
- Sensitive information is not exposed in error messages
- Connection errors provide helpful suggestions without revealing internals
- Timeout errors distinguished from connection errors for proper handling

## Known Limitations

### Arbitrary Code Execution
The `execute_python` tool allows arbitrary Python code execution in Slicer's environment. This is by design for flexibility but requires trust in the code being executed.

**Mitigations:**
- Audit logging of all executed code
- Intended for local, trusted environments only
- Not recommended for production or multi-user deployments

### No Authentication
Communication between the MCP server and Slicer WebServer has no authentication.

**Mitigations:**
- Localhost-only communication by default
- Intended for single-user environments
- WebServer should not be exposed to network

## Best Practices

1. **Run locally only**: Do not expose Slicer WebServer to the network
2. **Review generated code**: Inspect Python code before execution when possible
3. **Enable audit logging**: Set `SLICER_AUDIT_LOG` environment variable
4. **Keep software updated**: Use latest versions of Slicer and this bridge
5. **Non-clinical use**: Do not use for clinical decision-making

## Reporting Security Issues

If you discover a security vulnerability, please report it by:
1. Opening a private security advisory on GitHub
2. Emailing the maintainers directly (do not open public issues for security vulnerabilities)

## Version History

| Version | Security Updates |
|---------|------------------|
| 0.1.0   | Initial security documentation |
| 0.2.0   | Added audit log path validation |
| 0.3.0   | Added Unicode support for medical terminology in segment names |
