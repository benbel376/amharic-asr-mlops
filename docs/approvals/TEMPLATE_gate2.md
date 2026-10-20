# Gate 2 approval record (GOV-01)

Copy to `docs/approvals/YYYY-MM-DD_<model_version>.md`, fill in, commit. No record, no deployment.

- Model version to deploy:
- Replaces:
- Evidence: `results/<model>.json`, `results/promotion_<model>.json`, MLflow run id:
- Promotion rule result (PROMOTE / REJECT) and any exception granted, with reason:
- Per-group results reviewed (clean / phone, short / long):
- Load test result (users, p95, busy share):
- Rollout plan: canary % and duration; rollback trigger; rollback command (`CANARY_PERCENT=0`)
- Privacy check: logs contain no audio or text (test `test_logs_never_contain_transcript` passed)
- Approved by (name, role), date:
