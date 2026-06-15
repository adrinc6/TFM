import pandas as pd

from module.steps.step_03_training.agent_config import build_agents_config
from module.common.cross_sectional_features import enrich_cross_sectional_features
from module.steps.step_04_evaluation.evaluator import _prepare_fold_labels


def test_garp_agent_stack_replaces_momentum_buyer():
    cfg = build_agents_config('/tmp/agents', 42)
    assert {'quality', 'growth', 'valuation', 'fundamental_trend', 'catalyst', 'risk_bear', 'technical_guardrail'} <= set(cfg)
    assert 'momentum' not in cfg
    assert cfg['risk_bear']['invert_y'] is True


def test_cross_sectional_valuation_percentiles_are_added():
    idx = pd.MultiIndex.from_product([['AAA', 'BBB', 'CCC'], [pd.Timestamp('2024-03-31')]], names=['ticker', 'date'])
    df = pd.DataFrame({
        'sector': ['Tech', 'Tech', 'Tech'],
        'fcf_yield': [0.08, 0.02, 0.05],
        'earnings_yield': [0.07, 0.01, 0.04],
        'ev_to_ebitda': [8.0, 30.0, 15.0],
        'pe_ratio': [12.0, 60.0, 25.0],
    }, index=idx)
    out = enrich_cross_sectional_features(df)
    assert 'valuation_percentile_sector' in out
    assert 'valuation_percentile_universe' in out
    assert out.loc[('AAA', pd.Timestamp('2024-03-31')), 'valuation_percentile_sector'] > out.loc[('BBB', pd.Timestamp('2024-03-31')), 'valuation_percentile_sector']


def test_garp_composite_labels_use_train_threshold_without_tp_sl_prices():
    idx_train = pd.MultiIndex.from_product([['AAA', 'BBB', 'CCC', 'DDD'], [pd.Timestamp('2023-03-31')]], names=['ticker', 'date'])
    idx_test = pd.MultiIndex.from_product([['EEE', 'FFF'], [pd.Timestamp('2024-03-31')]], names=['ticker', 'date'])
    base_cols = {
        'forward_return': [0.30, -0.10, 0.12, 0.02],
        'sector': ['Tech', 'Tech', 'Health', 'Health'],
        'roic': [0.30, 0.05, 0.15, 0.10],
        'fcf_margin': [0.25, -0.05, 0.10, 0.05],
        'gross_margin': [0.70, 0.30, 0.55, 0.40],
        'piotroski_fscore': [8, 2, 6, 4],
        'revenue_yoy_growth': [0.20, -0.05, 0.10, 0.02],
        'fcf_yoy_growth': [0.25, -0.20, 0.08, 0.01],
        'eps_growth_trend_3y': [0.10, -0.05, 0.04, 0.00],
        'fcf_yield': [0.08, 0.01, 0.05, 0.03],
        'earnings_yield': [0.07, 0.01, 0.04, 0.02],
        'ev_to_ebitda': [8, 35, 15, 20],
        'pe_ratio': [12, 80, 25, 40],
        'roic_trend_2y': [0.05, -0.04, 0.02, 0.00],
        'net_margin_trend_2y': [0.04, -0.03, 0.01, 0.00],
        'eps_revision': [0.05, -0.02, 0.01, 0.00],
        'debt_to_ebitda': [0.5, 5.0, 1.5, 2.5],
        'volatility_60d': [0.15, 0.60, 0.25, 0.30],
        'price_vs_52w_high': [0.95, 0.40, 0.80, 0.70],
    }
    train = pd.DataFrame(base_cols, index=idx_train)
    test = pd.DataFrame({k: v[:2] for k, v in base_cols.items()}, index=idx_test)
    spy = pd.Series([100, 105, 110], index=pd.to_datetime(['2023-03-31', '2024-03-31', '2025-03-31']))
    df_train, df_test, y_train, y_test, alpha_train, alpha_test = _prepare_fold_labels(train, test, spy_prices=spy, prices_dict={})
    assert 'garp_composite_target' in df_train
    assert 'garp_composite_target' in df_test
    assert y_train.nunique() == 2
    assert len(y_test) == len(test)
    assert alpha_train.index.equals(train.index)


def test_fail_fast_rejects_forward_features_and_old_agents():
    from module.common.garp_validation import validate_garp_runtime_config, validate_no_forward_features
    import pytest

    with pytest.raises(RuntimeError):
        validate_no_forward_features(['roic', 'forward_return'], context='unit')
    with pytest.raises(RuntimeError):
        validate_garp_runtime_config({'momentum': {'invert_y': False}})


def test_opportunity_classification_and_explainability_columns():
    from module.common.garp_validation import add_ticker_explainability

    df = pd.DataFrame({
        'final_score': [0.80, 0.30],
        'quality_score': [0.70, 0.30],
        'growth_score': [0.70, 0.75],
        'valuation_score': [0.72, 0.20],
        'fundamental_trend_score': [0.65, 0.35],
        'catalyst_score': [0.55, 0.40],
        'risk_bear_score': [0.70, 0.30],
        'technical_guardrail_score': [0.55, 0.50],
    })
    out = add_ticker_explainability(df)
    assert out.loc[0, 'opportunity_class'] in {'Growth infravalorado', 'Quality Growth razonable'}
    assert out.loc[1, 'opportunity_class'] == 'Descartar'
    assert {'top_5_positive_drivers', 'top_5_risks', 'why_not_value_trap', 'why_not_expensive_growth'} <= set(out.columns)


def test_expectation_gap_moat_and_overexpectation_features_are_point_in_time():
    idx = pd.MultiIndex.from_product([
        ['CHEAP_QUALITY', 'EXPENSIVE_GROWTH'],
        [pd.Timestamp('2024-03-31')],
    ], names=['ticker', 'date'])
    df = pd.DataFrame({
        'sector': ['Tech', 'Tech'],
        'gross_margin': [0.75, 0.45],
        'operating_margin': [0.30, 0.05],
        'fcf_margin': [0.25, -0.02],
        'roic': [0.22, 0.03],
        'piotroski_fscore': [8, 3],
        'revenue_yoy_growth': [0.18, 0.40],
        'fcf_yoy_growth': [0.15, -0.10],
        'eps_growth_trend_3y': [0.08, 0.20],
        'fcf_yield': [0.07, 0.005],
        'earnings_yield': [0.06, 0.003],
        'ev_to_ebitda': [10.0, 80.0],
        'pe_ratio': [18.0, 150.0],
        'peg_ratio': [1.1, 8.0],
        'ev_to_sales': [4.0, 30.0],
        'ps_ratio': [5.0, 35.0],
    }, index=idx)
    out = enrich_cross_sectional_features(df)
    assert {'moat_proxy_score', 'expectation_gap_score', 'overexpectation_penalty'} <= set(out.columns)
    assert out.loc[('CHEAP_QUALITY', pd.Timestamp('2024-03-31')), 'expectation_gap_score'] > out.loc[('EXPENSIVE_GROWTH', pd.Timestamp('2024-03-31')), 'expectation_gap_score']
    assert out.loc[('EXPENSIVE_GROWTH', pd.Timestamp('2024-03-31')), 'overexpectation_penalty'] > out.loc[('CHEAP_QUALITY', pd.Timestamp('2024-03-31')), 'overexpectation_penalty']


def test_tp_sl_columns_are_not_in_core_agent_features_and_portfolio_range_is_5_10():
    from environment import GARP_MIN_STOCKS, GARP_MAX_STOCKS
    cfg = build_agents_config('/tmp/agents', 42)
    feature_names = []
    for c in cfg.values():
        feature_names.extend(c['kwargs'].get('include_features', []))
    assert not any('tp_sl' in f or f in {'tp_level', 'sl_level'} for f in feature_names)
    assert GARP_MIN_STOCKS == 5
    assert GARP_MAX_STOCKS == 10


def test_opportunity_explainability_exports_requested_fields():
    from module.common.garp_validation import add_ticker_explainability
    df = pd.DataFrame({
        'final_score': [0.78],
        'quality_score': [0.72],
        'growth_score': [0.68],
        'valuation_score': [0.70],
        'fundamental_trend_score': [0.65],
        'catalyst_score': [0.58],
        'risk_bear_score': [0.66],
        'technical_guardrail_score': [0.52],
        'moat_proxy_score': [0.74],
        'expectation_gap_score': [0.76],
        'overexpectation_penalty': [0.25],
    })
    out = add_ticker_explainability(df)
    required = {
        'opportunity_type', 'reason_for_classification', 'value_trap_flag',
        'expensive_growth_flag', 'moat_proxy_score', 'expectation_gap_score',
    }
    assert required <= set(out.columns)
    assert bool(out.loc[0, 'value_trap_flag']) is False
    assert bool(out.loc[0, 'expensive_growth_flag']) is False
