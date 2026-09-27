# Data agreement checklist (before receiving customer data)

**Not legal advice.** This is a checklist of topics a data-sharing agreement or NDA
should cover. Have an actual agreement drafted or reviewed by a lawyer, or use the
customer's standard NDA, before you receive any files.

- [ ] **Parties and purpose.** The data is used only to build and evaluate a predictive
      maintenance model for this customer's equipment.
- [ ] **What data.** Sensor logs and maintenance and breakdown records for the named
      machines. No personal data about employees. Ask them to strip operator names from
      work orders.
- [ ] **Where it lives.** On their premises (downloadable version) or on specific named
      hardware. Never in a public repository. This repo's `.gitignore` already blocks
      `pilot/customer_data/` and `custom_models/custom_*/`.
- [ ] **Who can access it.** Named individuals only.
- [ ] **How long.** Deleted, or returned, at the end of the pilot plus N days, with
      written confirmation.
- [ ] **Ownership.** They own their data. Clarify who owns a model trained on it, and
      whether you may reuse *anonymized* learnings.
- [ ] **Confidentiality.** Results aren't shared outside the two parties without consent.
      That includes case studies, LinkedIn posts and interviews.
- [ ] **Liability.** The model gives advisory warnings only. Maintenance decisions stay
      with their team, and the pilot doesn't replace their safety procedures.
- [ ] **Security basics.** Encrypted storage and transfer, no personal cloud drives,
      breach notification.
