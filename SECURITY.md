# Security Policy

## Supported versions

pydfine is pre-1.0 and follows a rolling-release model: **only the latest released version
on [PyPI](https://pypi.org/project/pydfine/) receives security fixes.** Please upgrade to the
newest release before reporting an issue.

| Version | Supported |
| ------- | --------- |
| latest `0.x` | ✅ |
| older `0.x`  | ❌ |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or
pull requests.**

Instead, use one of these private channels:

- **GitHub** — open a private advisory via
  [Security → Report a vulnerability](https://github.com/HoshiBatista/pydfine/security/advisories/new)
  (preferred), or
- **Email** — **morozyaka242005@gmail.com** with the details below.

Please include as much of the following as you can:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof of concept, affected version, and platform).
- Any known mitigations or workarounds.

## What to expect

- **Acknowledgement** within 5 business days.
- An assessment and, if confirmed, a fix plan with a target timeline.
- Coordinated disclosure: we will publish an advisory and credit you (if you wish) once a fix
  is released. Please give us reasonable time to remediate before any public disclosure.

## Scope

This policy covers the `pydfine` package (the `dfine` import package and its CLI). Note that
pydfine loads model weights (`.pth`/`.pt`) via `torch.load` and downloads checkpoints from
third-party hosts (GitHub Releases, Hugging Face) — **only load weights from sources you
trust.** Vulnerabilities in upstream dependencies (PyTorch, ONNX Runtime, etc.) should be
reported to their respective maintainers.
