# Change Log - Alpha Ranking & Portfolio Optimization Upgrade

## Scope
Production refactor from binary classification stack to alpha-oriented ranking + portfolio optimization with regime awareness.

## Files Added
- module/common/target_engineering.py
- module/common/purged_cv.py
- module/common/regime.py
- module/common/portfolio_optimization.py
- module/common/finbert_features.py
- module/common/cross_sectional_features.py
- module/agents/alpha_meta_learner.py

## Files Modified
- environment.py
- requirements.txt
- pyproject.toml
- module/agents/momentum.py
- module/steps/step_02_dataset/dataset.py
- module/steps/step_02_dataset/builders/sentiment.py
- module/steps/step_03_training/oof.py
- module/steps/step_03_training/training.py
- module/steps/step_04_evaluation/evaluator.py
- module/steps/step_04_evaluation/backtester.py
- module/steps/step_04_evaluation/ablation.py

## Deprecated / Superseded Components
- Legacy stacking path based on logistic+GBM inside module/agents/meta_learner.py is superseded by module/agents/alpha_meta_learner.py in the training pipeline.
- Score-only weighting path in backtester is superseded by HRP-based allocation when pre-entry covariance data is available.

## Functional Changes
1. Target Engineering:
- Added continuous alpha target: forward_return minus benchmark.
- Added cross-sectional alpha quintiles.
- Added triple-barrier labels (+1/0/-1).

2. Validation Integrity:
- Replaced OOF TimeSeriesSplit with PurgedEmbargoKFold (purge + embargo).

3. Ranking + Meta Decision Layer:
- Introduced AlphaMetaLearner (XGBoost regressor + ranker + risk model).
- Added outputs: predicted_alpha, ranking_score, risk_score, regime_adjusted_score.

4. Regime Awareness:
- Added MarketRegimeModel (Risk-On / Neutral / Risk-Off).
- Added dynamic regime weighting over agent scores.

5. Portfolio Optimization:
- Added HRP optimizer (with risk parity / robust Markowitz alternatives).
- Integrated optimizer into backtester with pre-entry trailing covariance window.

6. Feature Engineering:
- Added sector-relative cross-sectional features and volatility-adjusted signals.
- Added financially motivated interactions.

7. NLP Sentiment:
- Added FinBERT sentiment extraction with deterministic lexical fallback.
- Integrated sentiment NLP features into dataset and sentiment builder.

8. Momentum Deep Learning:
- Augmented MomentumAgent with TFT-lite sequence model over seq_* features.
- Added sequence feature generation from 60-day OHLCV dynamics.

9. Ablation Coverage:
- Extended ablation with regime, NLP, and HRP impact estimates.
