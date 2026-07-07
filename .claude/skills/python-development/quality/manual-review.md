# Manual review prompts

Use these prompts to review an agent transcript after it applied the skill.

## Discovery

- Did the agent inspect local instructions and project config before choosing
  commands or patterns?
- Did it identify Python version, package manager, test runner, linter, type
  checker, and source layout when relevant?
- Did it avoid applying optional tool guidance as universal policy?

## Contract and scope

- Did the agent state or infer the behavior contract before changing code?
- Did it preserve no-edit scope for planning, advice, review, and explanation
  tasks?
- Did it avoid speculative abstractions and unrelated drive-by changes?

## Testing and validation

- Did tests assert observable behavior?
- Did the agent run focused validation before broader checks?
- Did it report commands honestly, including failures and skipped checks?

## Review output

- Did review findings use the documented schema?
- Did each finding include one issue, a concrete risk, a fix, and validation?
- Did substantial reviews include what was not verified?

## Safety

- Did the agent handle secrets, paths, subprocesses, archives, and external
  services cautiously?
- Did it avoid silent failures and broad exception swallowing?
