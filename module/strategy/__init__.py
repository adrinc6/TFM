"""Strategy package: TP/SL + confidence-driven portfolio construction.

Modules
-------
signal_generation   – Translate agent scores into TP/SL percentages.
confidence_model    – Combine scores and historical calibration into confidence.
portfolio_selection – Rank and select 4–8 stocks using expected value.
backtesting_engine  – Simulate TP/SL outcomes on historical price series.
agent_weighting     – Dynamic per-agent weights based on historical hit rates.
"""
