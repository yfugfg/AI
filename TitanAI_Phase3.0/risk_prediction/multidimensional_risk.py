import numpy as np
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

class MultidimensionalRiskPredictor:
    def __init__(self, risk_dimensions: List[str] = None):
        self.risk_dimensions = risk_dimensions or [
            'market', 'credit', 'liquidity', 'operational',
            'geopolitical', 'regulatory', 'inflation', 'interest_rate'
        ]
        self.risk_models = {}
        self.historical_risk_data = {}
    
    def predict_risk(self, market_data: Dict[str, Any], time_horizon: str = 'short-term'):
        risk_predictions = {}
        
        for dimension in self.risk_dimensions:
            risk_score = self._calculate_risk_score(dimension, market_data)
            risk_predictions[dimension] = {
                'risk_score': risk_score,
                'risk_level': self._interpret_risk_level(risk_score),
                'confidence': self._calculate_risk_confidence(dimension, market_data),
                'key_factors': self._identify_key_risk_factors(dimension, market_data)
            }
        
        aggregated_risk = self._aggregate_risk_scores(risk_predictions)
        
        return {
            'risk_predictions': risk_predictions,
            'aggregated_risk': aggregated_risk,
            'time_horizon': time_horizon,
            'timestamp': datetime.now().isoformat(),
            'recommendations': self._generate_risk_recommendations(aggregated_risk, risk_predictions)
        }
    
    def detect_systemic_risk(self, market_data: Dict[str, Any], economic_data: Dict[str, Any]):
        systemic_risk_indicators = {
            'market_correlation': self._calculate_market_correlation(market_data),
            'credit_spread_widening': self._calculate_credit_spread_change(economic_data),
            'liquidity_deterioration': self._assess_liquidity_conditions(market_data),
            'volatility_clustering': self._detect_volatility_clustering(market_data),
            'funding_stress': self._assess_funding_conditions(economic_data)
        }
        
        systemic_risk_score = self._calculate_systemic_risk_score(systemic_risk_indicators)
        
        return {
            'systemic_risk_score': systemic_risk_score,
            'systemic_risk_level': self._interpret_systemic_risk_level(systemic_risk_score),
            'risk_indicators': systemic_risk_indicators,
            'early_warning_signals': self._detect_early_warning_signals(systemic_risk_indicators),
            'confidence': self._calculate_systemic_risk_confidence(systemic_risk_indicators)
        }
    
    def simulate_extreme_market_events(self, historical_data: Dict[str, Any], scenarios: List[str] = None):
        scenarios = scenarios or [
            'market_crash', 'liquidity_crunch', 'credit_contagion',
            'inflation_spike', 'interest_rate_shock', 'geopolitical_crisis'
        ]
        
        simulation_results = {}
        for scenario in scenarios:
            impact = self._simulate_scenario_impact(scenario, historical_data)
            simulation_results[scenario] = {
                'impact': impact,
                'probability': self._estimate_scenario_probability(scenario),
                'recovery_time': self._estimate_recovery_time(scenario, impact),
                'mitigation_strategies': self._generate_mitigation_strategies(scenario)
            }
        
        return {
            'simulation_results': simulation_results,
            'highest_risk_scenario': max(simulation_results.items(), key=lambda x: x[1]['impact']['severity'])[0],
            'overall_system_resilience': self._assess_system_resilience(simulation_results),
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_risk_hedging(self, portfolio: Dict[str, Any], risk_predictions: Dict[str, Any]):
        portfolio_risk_profile = self._analyze_portfolio_risk(portfolio)
        hedging_strategy = self._generate_hedging_strategy(portfolio_risk_profile, risk_predictions)
        
        return {
            'optimal_hedging_strategy': hedging_strategy,
            'portfolio_risk_profile': portfolio_risk_profile,
            'expected_risk_reduction': hedging_strategy.get('expected_risk_reduction', 0.3),
            'implementation_cost': hedging_strategy.get('implementation_cost', 0.02),
            'risk_adjusted_return': self._calculate_risk_adjusted_return(portfolio, hedging_strategy)
        }
    
    def _calculate_risk_score(self, dimension: str, market_data: Dict[str, Any]):
        base_score = np.random.uniform(0.3, 0.7)
        
        if dimension == 'market':
            return min(1.0, base_score + self._market_volatility_factor(market_data) * 0.2)
        elif dimension == 'credit':
            return min(1.0, base_score + self._credit_quality_factor(market_data) * 0.2)
        elif dimension == 'liquidity':
            return min(1.0, base_score + self._liquidity_factor(market_data) * 0.2)
        else:
            return base_score
    
    def _interpret_risk_level(self, risk_score: float):
        if risk_score > 0.7:
            return 'high'
        elif risk_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_risk_confidence(self, dimension: str, market_data: Dict[str, Any]):
        return np.random.uniform(0.6, 0.9)
    
    def _identify_key_risk_factors(self, dimension: str, market_data: Dict[str, Any]):
        factor_mapping = {
            'market': ['volatility', 'valuation', 'sentiment'],
            'credit': ['default_rates', 'spread_widening', 'credit_quality'],
            'liquidity': ['bid_ask_spread', 'trading_volume', 'market_depth'],
            'operational': ['system_reliability', 'cybersecurity', 'human_error'],
            'geopolitical': ['conflicts', 'trade_tensions', 'regime_change'],
            'regulatory': ['policy_changes', 'compliance_costs', 'legal_risk'],
            'inflation': ['price_pressures', 'wage_growth', 'monetary_policy'],
            'interest_rate': ['rate_changes', 'yield_curve', 'central_bank_policy']
        }
        return factor_mapping.get(dimension, ['general_market_conditions'])
    
    def _aggregate_risk_scores(self, risk_predictions: Dict[str, Any]):
        weighted_scores = []
        weights = {
            'market': 0.25,
            'credit': 0.20,
            'liquidity': 0.15,
            'operational': 0.10,
            'geopolitical': 0.10,
            'regulatory': 0.05,
            'inflation': 0.05,
            'interest_rate': 0.10
        }
        
        for dimension, data in risk_predictions.items():
            weight = weights.get(dimension, 0.1)
            weighted_scores.append(data['risk_score'] * weight)
        
        aggregated_score = sum(weighted_scores)
        return {
            'total_risk_score': aggregated_score,
            'risk_level': self._interpret_risk_level(aggregated_score),
            'dominant_risk_factors': self._identify_dominant_risk_factors(risk_predictions)
        }
    
    def _generate_risk_recommendations(self, aggregated_risk: Dict[str, Any], risk_predictions: Dict[str, Any]):
        recommendations = []
        
        if aggregated_risk['risk_level'] == 'high':
            recommendations.append('Reduce portfolio risk exposure')
            recommendations.append('Increase cash reserves')
            recommendations.append('Implement hedging strategies')
        elif aggregated_risk['risk_level'] == 'medium':
            recommendations.append('Maintain diversified portfolio')
            recommendations.append('Monitor key risk factors')
            recommendations.append('Prepare contingency plans')
        else:
            recommendations.append('Opportunistic positioning')
            recommendations.append('Gradual risk taking')
        
        for dimension, data in risk_predictions.items():
            if data['risk_level'] == 'high':
                recommendations.append(f'Mitigate {dimension} risk exposure')
        
        return recommendations
    
    def _calculate_market_correlation(self, market_data: Dict[str, Any]):
        return np.random.uniform(0.6, 0.95)
    
    def _calculate_credit_spread_change(self, economic_data: Dict[str, Any]):
        return np.random.uniform(-0.1, 0.3)
    
    def _assess_liquidity_conditions(self, market_data: Dict[str, Any]):
        return np.random.uniform(0.2, 0.8)
    
    def _detect_volatility_clustering(self, market_data: Dict[str, Any]):
        return np.random.uniform(0.3, 0.7)
    
    def _assess_funding_conditions(self, economic_data: Dict[str, Any]):
        return np.random.uniform(0.2, 0.6)
    
    def _calculate_systemic_risk_score(self, indicators: Dict[str, Any]):
        weights = {
            'market_correlation': 0.25,
            'credit_spread_widening': 0.20,
            'liquidity_deterioration': 0.20,
            'volatility_clustering': 0.15,
            'funding_stress': 0.20
        }
        
        weighted_score = sum(indicators[key] * weights[key] for key in indicators)
        return min(1.0, weighted_score)
    
    def _interpret_systemic_risk_level(self, score: float):
        if score > 0.7:
            return 'severe'
        elif score > 0.5:
            return 'elevated'
        elif score > 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _detect_early_warning_signals(self, indicators: Dict[str, Any]):
        signals = []
        if indicators['market_correlation'] > 0.85:
            signals.append('High market correlation indicating systemic risk')
        if indicators['credit_spread_widening'] > 0.2:
            signals.append('Credit spread widening')
        if indicators['liquidity_deterioration'] > 0.6:
            signals.append('Liquidity conditions deteriorating')
        return signals
    
    def _calculate_systemic_risk_confidence(self, indicators: Dict[str, Any]):
        return np.random.uniform(0.6, 0.85)
    
    def _simulate_scenario_impact(self, scenario: str, historical_data: Dict[str, Any]):
        impact_severity = {
            'market_crash': np.random.uniform(0.7, 0.95),
            'liquidity_crunch': np.random.uniform(0.6, 0.85),
            'credit_contagion': np.random.uniform(0.65, 0.9),
            'inflation_spike': np.random.uniform(0.5, 0.8),
            'interest_rate_shock': np.random.uniform(0.55, 0.85),
            'geopolitical_crisis': np.random.uniform(0.6, 0.9)
        }
        
        return {
            'severity': impact_severity.get(scenario, 0.6),
            'market_impact': self._estimate_market_impact(scenario),
            'sector_impacts': self._estimate_sector_impacts(scenario),
            'duration': self._estimate_event_duration(scenario)
        }
    
    def _estimate_scenario_probability(self, scenario: str):
        probabilities = {
            'market_crash': 0.05,
            'liquidity_crunch': 0.08,
            'credit_contagion': 0.06,
            'inflation_spike': 0.1,
            'interest_rate_shock': 0.12,
            'geopolitical_crisis': 0.15
        }
        return probabilities.get(scenario, 0.1)
    
    def _estimate_recovery_time(self, scenario: str, impact: Dict[str, Any]):
        base_recovery_times = {
            'market_crash': 90,
            'liquidity_crunch': 60,
            'credit_contagion': 120,
            'inflation_spike': 180,
            'interest_rate_shock': 90,
            'geopolitical_crisis': 150
        }
        
        base_time = base_recovery_times.get(scenario, 90)
        severity_factor = impact['severity']
        return int(base_time * (1 + severity_factor))
    
    def _generate_mitigation_strategies(self, scenario: str):
        strategies = {
            'market_crash': ['Implement stop-loss orders', 'Hedge with put options', 'Reduce equity exposure'],
            'liquidity_crunch': ['Maintain cash reserves', 'Avoid illiquid assets', 'Use limit orders'],
            'credit_contagion': ['Diversify counterparty risk', 'Monitor credit ratings', 'Reduce leverage'],
            'inflation_spike': ['Invest in inflation-protected securities', 'Allocate to commodities', 'Adjust duration exposure'],
            'interest_rate_shock': ['Adjust portfolio duration', 'Consider floating-rate securities', 'Hedge with interest rate derivatives'],
            'geopolitical_crisis': ['Increase portfolio diversification', 'Maintain defensive positions', 'Prepare for volatility']
        }
        return strategies.get(scenario, ['Maintain diversified portfolio'])
    
    def _assess_system_resilience(self, simulation_results: Dict[str, Any]):
        impacts = [data['impact']['severity'] for data in simulation_results.values()]
        average_impact = np.mean(impacts)
        return max(0, 1 - average_impact)
    
    def _analyze_portfolio_risk(self, portfolio: Dict[str, Any]):
        return {
            'total_risk': np.random.uniform(0.4, 0.7),
            'sector_concentration': np.random.uniform(0.3, 0.6),
            'geographic_concentration': np.random.uniform(0.2, 0.5),
            'liquidity_profile': np.random.uniform(0.4, 0.8),
            'leverage_level': np.random.uniform(0.1, 0.4)
        }
    
    def _generate_hedging_strategy(self, portfolio_risk: Dict[str, Any], risk_predictions: Dict[str, Any]):
        return {
            'hedging_instruments': ['index_put_options', 'VIX futures', 'inverse ETFs'],
            'hedge_ratio': 0.25,
            'expected_risk_reduction': 0.3,
            'implementation_cost': 0.02,
            'optimal_hedging_duration': '3 months'
        }
    
    def _calculate_risk_adjusted_return(self, portfolio: Dict[str, Any], hedging_strategy: Dict[str, Any]):
        return np.random.uniform(0.05, 0.12)
    
    def _market_volatility_factor(self, market_data: Dict[str, Any]):
        return np.random.uniform(0, 1)
    
    def _credit_quality_factor(self, market_data: Dict[str, Any]):
        return np.random.uniform(0, 1)
    
    def _liquidity_factor(self, market_data: Dict[str, Any]):
        return np.random.uniform(0, 1)
    
    def _identify_dominant_risk_factors(self, risk_predictions: Dict[str, Any]):
        sorted_factors = sorted(risk_predictions.items(), key=lambda x: x[1]['risk_score'], reverse=True)
        return [factor for factor, _ in sorted_factors[:3]]
    
    def _estimate_market_impact(self, scenario: str):
        impacts = {
            'market_crash': -0.25,
            'liquidity_crunch': -0.15,
            'credit_contagion': -0.2,
            'inflation_spike': -0.1,
            'interest_rate_shock': -0.12,
            'geopolitical_crisis': -0.18
        }
        return impacts.get(scenario, -0.1)
    
    def _estimate_sector_impacts(self, scenario: str):
        return {
            'technology': np.random.uniform(-0.3, 0.1),
            'financials': np.random.uniform(-0.25, 0.05),
            'healthcare': np.random.uniform(-0.15, 0.05),
            'consumer_staples': np.random.uniform(-0.1, 0.05),
            'energy': np.random.uniform(-0.2, 0.15)
        }
    
    def _estimate_event_duration(self, scenario: str):
        durations = {
            'market_crash': 30,
            'liquidity_crunch': 45,
            'credit_contagion': 60,
            'inflation_spike': 180,
            'interest_rate_shock': 90,
            'geopolitical_crisis': 120
        }
        return durations.get(scenario, 60)

class ExtremeRiskEarlyWarningSystem:
    def __init__(self, predictor: MultidimensionalRiskPredictor):
        self.predictor = predictor
        self.warning_thresholds = {
            'systemic_risk': 0.7,
            'market_crash': 0.8,
            'liquidity_crisis': 0.75,
            'credit_crisis': 0.7
        }
        self.alert_history = []
    
    def monitor_market_conditions(self, market_data: Dict[str, Any], economic_data: Dict[str, Any]):
        risk_prediction = self.predictor.predict_risk(market_data)
        systemic_risk = self.predictor.detect_systemic_risk(market_data, economic_data)
        
        alerts = self._generate_alerts(risk_prediction, systemic_risk)
        
        if alerts:
            self._record_alerts(alerts)
        
        return {
            'risk_prediction': risk_prediction,
            'systemic_risk': systemic_risk,
            'alerts': alerts,
            'market_health_index': self._calculate_market_health_index(risk_prediction, systemic_risk),
            'recommendations': self._generate_early_warning_recommendations(alerts, risk_prediction)
        }
    
    def backtest_early_warning_signals(self, historical_data: Dict[str, Any], historical_events: List[Dict[str, Any]]):
        performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'average_lead_time': 0
        }
        
        lead_times = []
        for event in historical_events:
            event_date = event['date']
            event_type = event['type']
            
            historical_market_data = self._extract_historical_data(historical_data, event_date, lookback_days=90)
            historical_economic_data = self._extract_historical_economic_data(historical_data, event_date, lookback_days=90)
            
            if historical_market_data and historical_economic_data:
                monitoring_result = self.monitor_market_conditions(historical_market_data, historical_economic_data)
                
                if monitoring_result['alerts']:
                    performance_metrics['true_positives'] += 1
                    lead_times.append(30)  # Simulated lead time
                else:
                    performance_metrics['false_negatives'] += 1
        
        if lead_times:
            performance_metrics['average_lead_time'] = np.mean(lead_times)
        
        performance_metrics['precision'] = performance_metrics['true_positives'] / (
            performance_metrics['true_positives'] + performance_metrics['false_positives'] + 1e-9
        )
        performance_metrics['recall'] = performance_metrics['true_positives'] / (
            performance_metrics['true_positives'] + performance_metrics['false_negatives'] + 1e-9
        )
        
        return {
            'performance_metrics': performance_metrics,
            'backtest_summary': self._generate_backtest_summary(performance_metrics),
            'recommendations': self._generate_backtest_recommendations(performance_metrics)
        }
    
    def _generate_alerts(self, risk_prediction: Dict[str, Any], systemic_risk: Dict[str, Any]):
        alerts = []
        
        if systemic_risk['systemic_risk_score'] > self.warning_thresholds['systemic_risk']:
            alerts.append({
                'type': 'systemic_risk',
                'severity': 'high',
                'message': 'Elevated systemic risk detected',
                'confidence': systemic_risk['confidence'],
                'timestamp': datetime.now().isoformat()
            })
        
        for dimension, data in risk_prediction['risk_predictions'].items():
            if data['risk_score'] > 0.8:
                alerts.append({
                    'type': dimension + '_risk',
                    'severity': 'medium',
                    'message': f'High {dimension} risk detected',
                    'confidence': data['confidence'],
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def _record_alerts(self, alerts: List[Dict[str, Any]]):
        for alert in alerts:
            self.alert_history.append(alert)
    
    def _calculate_market_health_index(self, risk_prediction: Dict[str, Any], systemic_risk: Dict[str, Any]):
        health_score = 1 - systemic_risk['systemic_risk_score']
        return max(0, min(1, health_score))
    
    def _generate_early_warning_recommendations(self, alerts: List[Dict[str, Any]], risk_prediction: Dict[str, Any]):
        recommendations = []
        
        if alerts:
            recommendations.append('Increase monitoring frequency')
            recommendations.append('Review risk management strategies')
            recommendations.append('Prepare contingency plans')
        
        return recommendations
    
    def _extract_historical_data(self, historical_data: Dict[str, Any], event_date: str, lookback_days: int = 90):
        return historical_data
    
    def _extract_historical_economic_data(self, historical_data: Dict[str, Any], event_date: str, lookback_days: int = 90):
        return historical_data
    
    def _generate_backtest_summary(self, performance_metrics: Dict[str, Any]):
        return f"Early warning system detected {performance_metrics['true_positives']} out of {performance_metrics['true_positives'] + performance_metrics['false_negatives']} events"
    
    def _generate_backtest_recommendations(self, performance_metrics: Dict[str, Any]):
        recommendations = []
        if performance_metrics['recall'] < 0.7:
            recommendations.append('Improve signal detection sensitivity')
        if performance_metrics['precision'] < 0.5:
            recommendations.append('Reduce false positive rate')
        return recommendations