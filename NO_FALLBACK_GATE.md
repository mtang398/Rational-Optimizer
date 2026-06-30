# No-Fallback Gate

This repository uses a universal no-fallback rule.

Concrete rule:

No fallback is allowed. Ever.

If the intended method, tool, command, renderer, inspection path, data source, verification path, or workflow fails, do not replace it with another method and do not lower the standard. The required response is to fix or unblock the intended path, then continue using that intended path.

Blocking is not a shortcut. Report blocked only when the intended path cannot be repaired from the current environment and requires an external permission, external state change, missing credential, unavailable machine, or user decision. The blocked report must name the exact intended path, the exact failure, what was tried to repair it, and what external action is required.

Do not use a weaker proxy.
Do not use a last resort.
Do not continue by analogy.
Do not silently substitute a different check.
Do not treat partial evidence as completion.
Do not convert a repairable failure into a failed task.

This rule applies to every task in this repository.
