# Transit Guide: Claude Code → GitHub Copilot CLI

Personal reference for running **both** Claude Code and GitHub Copilot CLI on this
repo. It maps each Claude Code concept to its Copilot CLI equivalent and lists the
one-time setup. Nothing here changes how the course is graded — it's about my
tooling.

> Verified against GitHub Docs (June 2026). Sources listed at the bottom.

## TL;DR
- The CS336 instructions in **`AGENTS.md` already work in Copilot CLI** — it reads
  root `AGENTS.md` natively. No porting needed for course rules.
- My personal preferences now live in **`.github/copilot-instructions.md`** (Copilot
  reads it alongside `AGENTS.md`) and in **Claude Code memory** (for Claude). Both
  tools are covered.
- Keep `AGENTS.md` and `CLAUDE.md` untouched (upstream-tracked) to avoid merge
  conflicts when syncing the Stanford upstream.

## Concept mapping

| Capability | Claude Code | GitHub Copilot CLI |
|---|---|---|
| Repo instructions | `CLAUDE.md`, `AGENTS.md` | `AGENTS.md` (root/cwd), `.github/copilot-instructions.md`, `.github/instructions/**.instructions.md`. Also reads `CLAUDE.md`/`GEMINI.md`. **Both** `AGENTS.md` and `copilot-instructions.md` are used together. |
| Personal/cross-session memory | `~/.claude/.../memory/*.md` + `MEMORY.md` index | No built-in memory store. Encode durable prefs as instruction files (this repo: `.github/copilot-instructions.md`). |
| Config / session home | `~/.claude/` | `~/.copilot/` (override via `COPILOT_HOME`). Holds config, sessions, logs, customizations. |
| Permissions / settings | `.claude/settings.json` (allow/deny, hooks, env) | Approval prompts + config in `~/.copilot`; no hooks system equivalent. Re-grant tool approvals interactively. |
| MCP servers | MCP config in Claude settings | `~/.copilot/mcp-config.json`; GitHub MCP preconfigured. Manage with `/mcp` (e.g. `/mcp search`, `/mcp add`). |
| Subagents (Agent tool) | Built-in agent types (Explore, Plan, …) | **Custom agents** in `~/.copilot/agents` or repo `.github/agents`; invoke via slash command or let Copilot infer. |
| Skills / slash commands | `/skill-name` skills | Built-in slash commands (`/usage`, `/mcp`, …) + agent **skills** customization. |

## One-time setup checklist
- [ ] Install the CLI (macOS/Linux install script) and authenticate with GitHub.
- [ ] Run `copilot` from the repo root so it picks up `AGENTS.md` +
      `.github/copilot-instructions.md` automatically.
- [ ] Confirm instructions loaded (ask it: "what are your instructions for this repo?").
- [ ] (Optional) Add any MCP servers I use in Claude via `/mcp add` →
      `~/.copilot/mcp-config.json`.
- [ ] (Optional) Recreate any custom subagents under `.github/agents/` so they're
      shared with the repo (and version-controlled).
- [ ] Verify it honors the no-`Co-Authored-By` rule on a throwaway commit message.

## Keeping the two tools in sync
- **Source of truth for course rules:** `AGENTS.md` (read by both; leave to upstream).
- **Source of truth for my prefs:** `.github/copilot-instructions.md` (in-repo, read
  by Copilot) mirrored by Claude Code memory. When I change a preference, update
  both places.
- **Avoid upstream conflicts:** never put personal content in `AGENTS.md` /
  `CLAUDE.md` — those are synced from `stanford-cs336/assignment1-basics`.
- **Line endings:** `.gitattributes` enforces LF for both tools.

## Sources
- [Using GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/use-copilot-cli)
- [Adding custom instructions for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-custom-instructions)
- [GitHub Copilot CLI configuration directory](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-config-dir-reference)
- [Adding MCP servers for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-mcp-servers)
- [Creating and using custom agents for GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/create-custom-agents-for-cli)
- [Copilot coding agent now supports AGENTS.md](https://github.blog/changelog/2025-08-28-copilot-coding-agent-now-supports-agents-md-custom-instructions/)
