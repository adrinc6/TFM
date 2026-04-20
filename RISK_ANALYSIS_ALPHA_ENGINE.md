# Risk Analysis - Alpha Engine Upgrade

## 1. Leakage Risks
### Mitigations Implemented
- Purged + embargoed temporal OOF splits (module/common/purged_cv.py).
- Point-in-time fold leakage audits remain active in evaluator.
- HRP weights use pre-entry trailing returns only (no forward window in optimization input).

### Residual Risks
- Any upstream data timestamping errors in source feeds can still contaminate labels/features.
- Textual fields used for sentiment may have publication-lag ambiguity if metadata is incomplete.

## 2. Overfitting Risks
### Mitigations Implemented
- Ranking objective aligned with cross-sectional selection.
- Multi-target design reduces reliance on a single noisy label definition.
- Regime-aware weighting reduces one-regime over-specialization.
- HRP controls concentration risk at portfolio level.

### Residual Risks
- Deep momentum sequence model can overfit in low-sample sectors.
- High-dimensional seq_* features increase complexity and require strict regularization monitoring.

## 3. Model Stability Risks
### Mitigations Implemented
- Deterministic seeds preserved.
- Conservative fallbacks remain in agents and scoring paths.
- Regime model defaults to neutral when signal is weak.

### Residual Risks
- Regime transitions can induce score discontinuities and turnover spikes.
- FinBERT availability/download constraints may trigger lexical fallback behavior changes across environments.

## 4. Portfolio Construction Risks
### Mitigations Implemented
- HRP and weight caps reduce single-name and single-cluster concentration.
- Optional risk parity / robust Markowitz baselines available for stress-testing.

### Residual Risks
- Covariance estimation instability in sparse windows.
- HRP clustering sensitivity under very high correlation regimes.

## 5. Operational Risks
- Additional dependencies (torch, transformers, lightgbm) increase environment complexity.
- Inference latency may increase with deep and NLP components.

## 6. Monitoring Recommendations
1. Track rolling IC (Spearman) of predicted_alpha by regime.
2. Monitor turnover and slippage attribution after regime switches.
3. Alert on fallback-rate spikes (missing deep/NLP path usage).
4. Run rolling ablation to detect component drift and degradation.
