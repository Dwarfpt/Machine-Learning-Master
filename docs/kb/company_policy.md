# Company policy for preventive measures

## Approval rule (human-in-the-loop)
Any measure that costs more than 50 k$, and any release delay regardless of cost, must be approved by the
producing manager. The manager receives an e-mail with signed Approve / Reject links; nothing is executed before
the decision. Cheaper measures (playtest, critic previews) start automatically and the team is notified.

## Low-risk games
If the model's risk level is low, no money is spent: the decision is MONITOR_ONLY. This rule overrides any
proposal of the AI agent.

## Budget per game
At most 3 measures per game. Prefer measures whose expected benefit is larger than their cost (positive net
value) and that address the top risk factors from the explanation (SHAP).

## Licensed and movie tie-in games
Licensed games tied to a fixed movie or event date cannot be delayed easily. Prefer extended QA and playtests;
propose a delay only with a clear justification for the manager.

## Sequels
Yearly sequels of established series are reviewed against the previous entry. Critics punish "more of the
same": focus measures on content and polish rather than marketing.

## Data and limits
The risk model is a prioritisation tool, not an oracle (PR-AUC is reported on the model page). Effects of process
measures (QA, playtest, PR, delay) are expert assumptions, not model outputs, and must be presented as such.
