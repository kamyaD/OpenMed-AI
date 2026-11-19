# User acceptance testing scenarios
| # | Role | Scenario | Pass if |
|---|---|---|---|
| 1 | Supply manager | Open overview, filter by region and risk, open a CRITICAL row | Risk, probability, freshness and facility/medicine detail shown |
| 2 | Supply manager | Read the explanation for a prediction | Factors are understandable and match the data |
| 3 | Supply manager | Review a redistribution suggestion and approve it with a rationale | Status changes; history shows actor, time and rationale |
| 4 | Supply manager | Reject a suggestion | Rationale required; status REJECTED |
| 5 | Facility user | Log in | Sees only own facilities; cannot approve |
| 6 | Analyst | Upload an inventory CSV with bad rows | Bad rows listed as issues with raw data; good rows loaded |
| 7 | Analyst | Run monitoring and read the alerts | Alerts name an owner and a response |
| 8 | Any | Export a filtered view | CSV contains only permitted facilities |
| 9 | Admin | Promote and roll back a model | Both recorded in the audit log |
Record defects, triage them, and obtain sponsor sign-off before the pilot.