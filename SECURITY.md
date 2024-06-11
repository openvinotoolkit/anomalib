# 🔒 Security Policy

Intel is committed to rapidly addressing security vulnerabilities affecting our
customers and providing clear guidance on the solution, impact, severity, and
mitigation.

## Security Tools and Practices

### Integrated Security Scanning with Bandit and Trivy

To ensure our codebase remains secure, we leverage GitHub Actions for continuous
security scanning with the following tools:

- **Bandit:** Automatically scans our Python code for common security issues,
  helping us identify and mitigate potential vulnerabilities proactively.
- **Trivy:** Integrated into our CI/CD pipeline via GitHub Actions, Trivy scans
  our project's dependencies and container images for known vulnerabilities,
  ensuring our external components are secure.

These integrations ensure that every commit and pull request is automatically
checked for security issues, allowing us to maintain a high security standard
across our development lifecycle.

### External Security Scanning with Checkmarx

In addition to our integrated tools, we utilize Checkmarx for static application
security testing (SAST). This comprehensive analysis tool is run externally to
scrutinize our source code for security vulnerabilities, complementing our
internal security measures with its advanced detection capabilities.

## 🚨 Reporting a Vulnerability

Please report any security vulnerabilities in this project utilizing the
guidelines [here](https://www.intel.com/content/www/us/en/security-center/vulnerability-handling-guidelines.html).

## 📢 Security Updates and Announcements

Users interested in keeping up-to-date with security announcements and updates
can:

- Follow the [GitHub repository](https://github.com/openvinotoolkit/anomalib) 🌐
- Check the [Releases](https://github.com/openvinotoolkit/anomalib/releases)
  section of our GitHub project 📦

We encourage users to report security issues and contribute to the security of
our project 🛡️. Contributions can be made in the form of code reviews, pull
requests, and constructive feedback. Refer to our
[CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

> **NOTE:** This security policy is subject to change 🔁. Users are encouraged
> to check this document periodically for updates.
