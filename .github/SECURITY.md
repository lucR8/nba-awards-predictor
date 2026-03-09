# Security Policy

Thank you for your interest in the security of the **NBA Awards Predictor** project.
This document outlines the security practices of the repository and how to responsibly report vulnerabilities.

---

## Repository security

This repository relies on GitHub’s built-in security features for public projects.

### Dependabot alerts
Dependabot is enabled to:
- detect vulnerabilities in Python dependencies,
- propose secure dependency updates automatically.

### Secret scanning
GitHub automatically scans the repository for:
- API keys,
- access tokens,
- passwords accidentally committed.

### Push protection
Push protection prevents accidental commits containing:
- secrets,
- credentials,
- personal access tokens.

If a secret is detected, the push is blocked automatically.

### Code scanning (CodeQL)
When enabled, GitHub CodeQL can analyze Python code to detect:
- common vulnerabilities,
- unsafe patterns,
- logic errors.

---

## Reporting a vulnerability

Please follow **responsible disclosure** practices.

### Do not open a public issue
Security vulnerabilities must **not** be reported through public GitHub issues.

### Private contact
Report vulnerabilities privately via email:

**luc.renaud8@gmail.com**

### GitHub private vulnerability reporting
You may also use GitHub’s private reporting feature:

**Security → Private vulnerability reporting → Report a vulnerability**

This ensures a secure and traceable discussion.

---

## Scope

Included components:
- Python source code
- Data engineering and machine learning pipelines
- Evaluation and configuration logic

Excluded:
- External datasets
- Locally generated artifacts

---

## Resolution process

When a vulnerability is reported:
1. Initial assessment 
2. Reproduction and validation
3. Patch development
4. Release of a fix if required
5. Optional credit to the reporter

---

## Best practices for contributors

- Never commit:
  - API secrets
  - private keys
  - personal tokens
- Use environment variables when required
- Avoid aggressive or automated scraping of third-party websites
- Keep dependencies reasonably up to date

---

For any security-related question, please contact:
**luc.renaud8@gmail.com**
