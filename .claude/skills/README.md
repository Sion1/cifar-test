# `.claude/skills/` — Per-project Claude skills

Claude Code auto-loads any `SKILL.md` it finds under `.claude/skills/<name>/`
when you start a session in this directory. The agentic loop's `analyze` and
`propose` invocations of `claude -p` inherit the same loading rules, so a
skill defined here applies to every iteration the loop runs.

A skill is a markdown file with YAML frontmatter:

```markdown
---
name: my-skill-name
description: When this skill should trigger (one paragraph). Be specific —
  Claude uses this string to decide whether to apply the skill on each turn.
---

# Skill body

Free-form markdown. Conventions, failure modes, examples, references —
whatever the agent should keep in mind while working in your domain.
```

## What's shipped here

| Skill | Purpose | Action |
|---|---|---|
| [`experiment-analysis/`](experiment-analysis/) | The framework's **default analysis protocol**. Encodes the `hypothesis × evidence × verdict` rubric the loop's analyze step relies on. Originally written for ZSL/GZSL research but the methodology generalizes — only the worked examples are domain-specific. | **Keep as-is** unless you want to retune the verdict thresholds. |
| [`_template_task_background/`](_template_task_background/) | Empty template for **your domain knowledge** skill. The framework relies on having one of these to give the agent field-specific context (named failure modes, benchmark conventions, what counts as a sound result in your area). | **You must fill this in.** See instructions below. |

## Adding your task-background skill (required)

The agentic loop is much weaker without a domain skill. Without it, the
agent has to re-derive your field's conventions from scratch every analyze
tick — which produces shallow verdicts and reinvents vocabulary.

To add yours:

```bash
# 1. Copy the template into a permanently-named directory:
cp -r .claude/skills/_template_task_background \
      .claude/skills/<your-task-name>

# 2. Rename the template file so Claude actually loads it:
mv .claude/skills/<your-task-name>/SKILL.md.template \
   .claude/skills/<your-task-name>/SKILL.md

# 3. Edit the frontmatter (`name` + `description`) and the body.
# 4. Optionally add references/ subdirectory with PDFs of survey papers.
# 5. Commit.
```

Examples of what your skill should cover (tailor to your area):

- **Foundational definitions** — terms that get conflated in your field
  (e.g. the ZSL skill leads with "ZSL vs GZSL" because that's the #1 source
  of unsound experiments).
- **Standard splits / benchmarks / metrics** — which version of each you use
  and what each one tests.
- **Named failure modes** — the failure modes your community has labeled, so
  the agent can diagnose into existing vocabulary instead of inventing terms.
- **Methods taxonomy** — a grouped overview so the agent can place a new
  paper into the right family in one read.
- **Evaluation conventions** — multi-seed reporting, baseline requirements,
  per-class breakdowns, anything reviewers expect.
- **Reading checklist** — what 5 questions the agent should answer before
  trusting a new paper.
- **When NOT to trigger** — the boundary (without this, the skill
  over-applies).

## Naming conventions

- The directory name is the skill name. Pick something specific
  (`my-image-restoration`, `causal-inference`, `protein-folding`) rather than
  generic (`research`, `domain`).
- Claude only loads files named **exactly** `SKILL.md`. Anything else
  (`SKILL.md.template`, `notes.md`, `references/*.md`) is for humans only
  and won't activate the skill.
- A skill's `references/` subdirectory is a good place for PDFs and notes
  the agent can `Read` if it wants more context — but they don't auto-load.

## Multiple skills

You can have as many skills as you want. They activate independently based
on their `description` field. Common patterns:

- One **methodology** skill (the shipped `experiment-analysis`).
- One **domain knowledge** skill (the `_template_task_background` slot).
- Optional: one **reproduction** skill (encoding "how I run the pipeline")
  if your project has many command-line gotchas.

## Reference materials

The Claude Code documentation on skills:
<https://docs.claude.com/en/docs/claude-code/skills>
