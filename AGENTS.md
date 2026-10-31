# verl agent instructions

Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing. Changes to this file or linked agent guides must follow [editing agent instructions](docs/contributing/editing-agent-instructions.md).

## Workspace and development

1. Work in a task branch and isolated worktree; in the EliAgent workspace, use `.worktree/verl-<task>/`. Preserve existing changes and services, and never force-push or delete the protected EnvLoop `v0.9.1` branch.
2. Use `uv` for Python environments and the interpreter range in `pyproject.toml`. Install development tooling and pre-commit hooks as described in the contribution guide; run checks relevant to the change.
3. Verify bot review suggestions against current code before applying them.
4. Keep AGENTS under 200 lines and domain guides under 300. Retain project-wide invariants here; link area-specific guidance and remove stale or duplicated rules.

## Contributions to verl-project/verl

1. Before proposing an upstream PR, read the issue and its comments, then search open PRs by issue number and area keywords. Do not duplicate an existing fix; explain a materially different approach in the issue.
2. Do not submit isolated trivial cleanup PRs. Bundle mechanical changes with substantive work; if the contribution is duplicate or trivial, stop and explain what is missing.
3. An accountable human must review every changed line, run relevant tests, and understand and defend the contribution. Pure code-agent submissions are not accepted.
4. Follow upstream attribution requirements: disclose AI assistance and include contribution trailers as required. The PR must explain how it avoids duplicate work and list test commands and results. EnvLoop fork delivery follows the workspace's own attribution rules.
5. Read any applicable domain guide before modifying its area. If the requested upstream change conflicts with that guide, explain the conflict before proceeding.
