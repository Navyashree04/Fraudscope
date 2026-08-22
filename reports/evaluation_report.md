# FraudScope Evaluation Report

> **Dataset note:** results below are computed on the **synthetic-for-demo** dataset (`synthetic-for-demo (locally generated stand-in)`), never real bank or customer data. All numbers are computed directly from the trained model artifact on a held-out, stratified test split -- nothing here is fabricated.

- Model version: `v1`
- Test set size: 10000 rows, 17 fraud

## Headline metrics

| Metric | Value |
|---|---|
| ROC-AUC | 1.0000 |
| PR-AUC (average precision) | 0.9866 |
| Precision @ 0.5 | 1.0000 |
| Recall @ 0.5 | 0.9412 |
| F1 @ 0.5 | 0.9697 |

**Why PR-AUC matters more than ROC-AUC here:** with fraud at roughly 0.17% of transactions, a model that predicts "legitimate" for almost everything can still post a high ROC-AUC and >99.8% raw accuracy, because ROC-AUC is dominated by the abundant negative class. PR-AUC (average precision) focuses on precision/recall trade-offs on the positive (fraud) class specifically, which is what actually matters for a fraud-ops team deciding how many alerts to review.

## Precision/recall at alternative thresholds

In a real fraud-ops setting, the decision threshold is tuned against the business cost of false positives (annoyed customers, manual review load) versus false negatives (fraud losses) -- it is not simply fixed at 0.5.

| Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| 0.3 | 1.0000 | 0.9412 | 0.9697 | 16 | 0 | 1 | 9983 |
| 0.5 | 1.0000 | 0.9412 | 0.9697 | 16 | 0 | 1 | 9983 |
| 0.7 | 1.0000 | 0.9412 | 0.9697 | 16 | 0 | 1 | 9983 |

A lower threshold catches more fraud (higher recall) at the cost of more false alarms (lower precision); a higher threshold does the opposite. Which one is 'right' depends on the relative cost of a missed fraud versus an analyst wasting time on a false alarm --  a business decision, not a purely technical one.
