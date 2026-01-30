import numpy as np
import time
from typing import Dict, List, Optional

class BrainComputerInterface:
    def __init__(self, sampling_rate=1000, buffer_size=1000):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros((buffer_size, 8))
        self.is_recording = False
        self.recording_start_time = 0
    
    def start_recording(self):
        self.is_recording = True
        self.recording_start_time = time.time()
        print("Brain signal recording started")
    
    def stop_recording(self):
        self.is_recording = False
        print("Brain signal recording stopped")
    
    def get_brain_signals(self, duration=1.0):
        samples = int(duration * self.sampling_rate)
        signals = np.random.randn(samples, 8) * 10
        
        for i in range(samples):
            t = i / self.sampling_rate
            signals[i, 0] += np.sin(2 * np.pi * 10 * t) * 5
            signals[i, 1] += np.sin(2 * np.pi * 20 * t) * 3
            signals[i, 2] += np.sin(2 * np.pi * 30 * t) * 2
        
        return signals
    
    def detect_mental_state(self, signals):
        features = self._extract_features(signals)
        
        alpha_power = features['alpha_power']
        beta_power = features['beta_power']
        theta_power = features['theta_power']
        
        if alpha_power > 0.4 and theta_power > 0.3:
            mental_state = 'relaxed'
        elif beta_power > 0.4 and alpha_power < 0.2:
            mental_state = 'focused'
        elif theta_power > 0.5:
            mental_state = 'meditative'
        else:
            mental_state = 'neutral'
        
        return {
            'mental_state': mental_state,
            'confidence': np.random.uniform(0.7, 0.95),
            'brain_waves': {
                'alpha': alpha_power,
                'beta': beta_power,
                'theta': theta_power,
                'gamma': features['gamma_power']
            }
        }
    
    def detect_intention(self, signals, possible_intentions=['buy', 'sell', 'hold']):
        features = self._extract_features(signals)
        
        intention_scores = {}
        for intention in possible_intentions:
            intention_scores[intention] = np.random.uniform(0, 1)
        
        detected_intention = max(intention_scores, key=intention_scores.get)
        confidence = intention_scores[detected_intention]
        
        return {
            'intention': detected_intention,
            'confidence': confidence,
            'scores': intention_scores
        }
    
    def _extract_features(self, signals):
        fft_result = np.fft.fft(signals, axis=0)
        power_spectrum = np.abs(fft_result) ** 2
        frequencies = np.fft.fftfreq(len(signals), 1/self.sampling_rate)
        
        alpha_band = (8, 12)
        beta_band = (13, 30)
        theta_band = (4, 7)
        gamma_band = (31, 100)
        
        alpha_power = np.mean(power_spectrum[np.logical_and(frequencies >= alpha_band[0], frequencies <= alpha_band[1])])
        beta_power = np.mean(power_spectrum[np.logical_and(frequencies >= beta_band[0], frequencies <= beta_band[1])])
        theta_power = np.mean(power_spectrum[np.logical_and(frequencies >= theta_band[0], frequencies <= theta_band[1])])
        gamma_power = np.mean(power_spectrum[np.logical_and(frequencies >= gamma_band[0], frequencies <= gamma_band[1])])
        
        total_power = alpha_power + beta_power + theta_power + gamma_power
        
        return {
            'alpha_power': alpha_power / total_power if total_power > 0 else 0,
            'beta_power': beta_power / total_power if total_power > 0 else 0,
            'theta_power': theta_power / total_power if total_power > 0 else 0,
            'gamma_power': gamma_power / total_power if total_power > 0 else 0,
            'mean_amplitude': np.mean(np.abs(signals)),
            'std_amplitude': np.std(signals)
        }

class NeuralTradingInterface:
    def __init__(self, bci):
        self.bci = bci
        self.trading_intentions = ['buy', 'sell', 'hold', 'increase_position', 'decrease_position']
        self.calibration_data = {}
    
    def calibrate(self, user_id, calibration_time=60):
        print(f"Calibrating neural interface for user {user_id}...")
        calibration_signals = []
        expected_intentions = []
        
        for intention in self.trading_intentions:
            print(f"Please think about: {intention}")
            time.sleep(2)
            signals = self.bci.get_brain_signals(duration=5)
            calibration_signals.append(signals)
            expected_intentions.append(intention)
        
        self.calibration_data[user_id] = {
            'signals': calibration_signals,
            'intentions': expected_intentions,
            'calibration_time': time.time()
        }
        
        print("Calibration completed successfully")
        return True
    
    def get_trading_decision(self, user_id, market_data=None):
        if user_id not in self.calibration_data:
            raise ValueError("User not calibrated")
        
        signals = self.bci.get_brain_signals(duration=2)
        mental_state = self.bci.detect_mental_state(signals)
        intention = self.bci.detect_intention(signals, self.trading_intentions)
        
        if mental_state['mental_state'] == 'focused' and intention['confidence'] > 0.7:
            decision = intention['intention']
            confidence = intention['confidence'] * mental_state['confidence']
        else:
            decision = 'hold'
            confidence = 0.5
        
        return {
            'decision': decision,
            'confidence': confidence,
            'mental_state': mental_state['mental_state'],
            'brain_state_quality': mental_state['confidence'],
            'recommendation': self._generate_recommendation(decision, market_data)
        }
    
    def monitor_trading_performance(self, trading_history):
        performance_metrics = {
            'total_trades': len(trading_history),
            'winning_trades': sum(1 for trade in trading_history if trade['profit'] > 0),
            'losing_trades': sum(1 for trade in trading_history if trade['profit'] < 0),
            'win_rate': sum(1 for trade in trading_history if trade['profit'] > 0) / len(trading_history) if trading_history else 0,
            'total_profit': sum(trade['profit'] for trade in trading_history)
        }
        
        if performance_metrics['win_rate'] > 0.6:
            performance_evaluation = 'excellent'
        elif performance_metrics['win_rate'] > 0.5:
            performance_evaluation = 'good'
        else:
            performance_evaluation = 'needs_improvement'
        
        return {
            'performance_metrics': performance_metrics,
            'evaluation': performance_evaluation,
            'recommendations': self._generate_performance_recommendations(performance_metrics)
        }
    
    def optimize_trading_strategy(self, user_id, trading_history, market_data):
        if user_id not in self.calibration_data:
            raise ValueError("User not calibrated")
        
        performance = self.monitor_trading_performance(trading_history)
        
        optimal_strategy = {
            'best_mental_state': 'focused',
            'optimal_decision_threshold': 0.75,
            'recommended_trading_frequency': 'medium',
            'risk_appetite': self._assess_risk_appetite(trading_history)
        }
        
        return {
            'optimal_strategy': optimal_strategy,
            'performance_analysis': performance,
            'implementation_suggestions': self._generate_implementation_suggestions(optimal_strategy)
        }
    
    def _generate_recommendation(self, decision, market_data):
        if decision == 'buy':
            return "Consider buying with proper position sizing"
        elif decision == 'sell':
            return "Consider selling to take profits or cut losses"
        elif decision == 'hold':
            return "Maintain current position"
        elif decision == 'increase_position':
            return "Consider increasing position size"
        elif decision == 'decrease_position':
            return "Consider reducing position size"
        else:
            return "No specific recommendation"
    
    def _generate_performance_recommendations(self, performance_metrics):
        recommendations = []
        
        if performance_metrics['win_rate'] < 0.5:
            recommendations.append("Consider improving decision quality")
        if performance_metrics['total_trades'] > 50:
            recommendations.append("Evaluate trading frequency")
        if performance_metrics['total_profit'] < 0:
            recommendations.append("Review risk management strategy")
        
        return recommendations
    
    def _assess_risk_appetite(self, trading_history):
        if not trading_history:
            return 'moderate'
        
        position_sizes = [trade.get('position_size', 1) for trade in trading_history]
        avg_position_size = np.mean(position_sizes)
        
        if avg_position_size > 0.7:
            return 'aggressive'
        elif avg_position_size < 0.3:
            return 'conservative'
        else:
            return 'moderate'
    
    def _generate_implementation_suggestions(self, optimal_strategy):
        suggestions = []
        
        if optimal_strategy['best_mental_state'] == 'focused':
            suggestions.append("Ensure proper mental preparation before trading")
        if optimal_strategy['risk_appetite'] == 'aggressive':
            suggestions.append("Consider implementing stricter stop-losses")
        if optimal_strategy['recommended_trading_frequency'] == 'medium':
            suggestions.append("Balance between opportunity capture and overtrading")
        
        return suggestions

class NeuralFeedbackSystem:
    def __init__(self, bci):
        self.bci = bci
        self.feedback_history = []
    
    def provide_feedback(self, trading_decision, market_outcome):
        timestamp = time.time()
        success = market_outcome['profit'] > 0
        
        feedback = {
            'timestamp': timestamp,
            'decision': trading_decision['decision'],
            'confidence': trading_decision['confidence'],
            'mental_state': trading_decision['mental_state'],
            'outcome': 'success' if success else 'failure',
            'profit': market_outcome['profit']
        }
        
        self.feedback_history.append(feedback)
        
        return {
            'feedback': feedback,
            'neural_recommendation': self._generate_neural_recommendation(feedback),
            'suggested_improvements': self._suggest_improvements(feedback)
        }
    
    def analyze_feedback_patterns(self):
        if len(self.feedback_history) < 5:
            return {"message": "Insufficient feedback data"}
        
        success_rate_by_state = {}
        for state in ['focused', 'relaxed', 'meditative', 'neutral']:
            state_decisions = [fb for fb in self.feedback_history if fb['mental_state'] == state]
            if state_decisions:
                success_rate_by_state[state] = sum(1 for fb in state_decisions if fb['outcome'] == 'success') / len(state_decisions)
        
        best_state = max(success_rate_by_state, key=success_rate_by_state.get) if success_rate_by_state else 'focused'
        
        return {
            'success_rates_by_mental_state': success_rate_by_state,
            'best_performing_mental_state': best_state,
            'overall_success_rate': sum(1 for fb in self.feedback_history if fb['outcome'] == 'success') / len(self.feedback_history),
            'recommendations': self._generate_pattern_recommendations(success_rate_by_state)
        }
    
    def _generate_neural_recommendation(self, feedback):
        if feedback['outcome'] == 'success' and feedback['confidence'] > 0.8:
            return "Excellent decision-making. Continue in this mental state."
        elif feedback['outcome'] == 'failure' and feedback['confidence'] < 0.6:
            return "Consider improving focus before making decisions."
        else:
            return "Maintain consistent mental preparation."
    
    def _suggest_improvements(self, feedback):
        suggestions = []
        
        if feedback['mental_state'] != 'focused':
            suggestions.append("Try to achieve focused mental state before trading")
        if feedback['confidence'] < 0.7:
            suggestions.append("Wait for higher confidence before executing trades")
        
        return suggestions
    
    def _generate_pattern_recommendations(self, success_rates):
        recommendations = []
        
        for state, rate in success_rates.items():
            if rate > 0.7:
                recommendations.append(f"Maximize trading during {state} mental state")
            elif rate < 0.4:
                recommendations.append(f"Minimize trading during {state} mental state")
        
        return recommendations