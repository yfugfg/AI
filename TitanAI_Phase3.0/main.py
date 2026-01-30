#!/usr/bin/env python3
"""
TitanAI Phase 3.0 - 超级智能金融交易系统

融合量子计算、脑机接口、分子计算和超级智能AI的下一代金融交易系统
"""

import sys
import json
import argparse
from datetime import datetime
from integration.system_integrator import SystemController

def main():
    parser = argparse.ArgumentParser(description='TitanAI Phase 3.0 - Super Intelligence Financial Trading System')
    parser.add_argument('--mode', choices=['start', 'analyze', 'decision', 'status', 'stop'], 
                      default='start', help='运行模式')
    parser.add_argument('--config', type=str, default='config/system_config.json', 
                      help='配置文件路径')
    parser.add_argument('--market-data', type=str, default=None, 
                      help='市场数据文件路径')
    parser.add_argument('--economic-data', type=str, default=None, 
                      help='经济数据文件路径')
    parser.add_argument('--portfolio', type=str, default=None, 
                      help='投资组合文件路径')
    parser.add_argument('--output', type=str, default=None, 
                      help='输出文件路径')
    
    args = parser.parse_args()
    
    print("🚀 TitanAI Phase 3.0 - 超级智能金融交易系统")
    print("=" * 60)
    
    controller = SystemController()
    
    if args.mode == 'start':
        print("正在启动系统...")
        config = _load_config(args.config)
        result = controller.start_system(config)
        print(f"系统状态: {result['status']}")
        print(f"初始化模块: {len(result['initialization']['initialized_modules'])}")
        print(f"系统健康状态: {result['initialization']['system_health']['status']}")
        _save_output(result, args.output or 'startup_result.json')
        
    elif args.mode == 'analyze':
        print("运行全面市场分析...")
        market_data = _load_data(args.market_data or 'data/market_data.json')
        economic_data = _load_data(args.economic_data or 'data/economic_data.json')
        result = controller.execute_analysis(market_data, economic_data)
        print(f"分析完成，置信度: {result['confidence']:.2f}")
        print(f"市场展望: {result['integrated_analysis']['market_outlook']}")
        print(f"风险评估: {result['integrated_analysis']['risk_assessment']['level']}")
        print("投资机会:")
        for opportunity in result['integrated_analysis']['investment_opportunities']:
            print(f"  - {opportunity}")
        _save_output(result, args.output or 'analysis_result.json')
        
    elif args.mode == 'decision':
        print("生成投资决策...")
        market_data = _load_data(args.market_data or 'data/market_data.json')
        economic_data = _load_data(args.economic_data or 'data/economic_data.json')
        portfolio = _load_data(args.portfolio or 'data/portfolio.json')
        result = controller.make_investment_decision(market_data, economic_data, portfolio)
        print(f"决策: {result['decision']['action'].upper()}")
        print(f"目标行业: {', '.join(result['decision']['target_sectors'])}")
        print(f"仓位大小: {result['decision']['position_size']}")
        print(f"决策置信度: {result['confidence']:.2f}")
        print(f"风险等级: {result['risk_assessment']['risk_level']}")
        print("实施步骤:")
        for step in result['implementation_plan']['steps']:
            print(f"  - {step}")
        _save_output(result, args.output or 'decision_result.json')
        
    elif args.mode == 'status':
        print("检查系统状态...")
        result = controller.check_system_status()
        print(f"系统状态: {result['system_status']}")
        print("性能指标:")
        for metric, value in result['performance_metrics']['resource_usage'].items():
            print(f"  {metric}: {value:.1f}%")
        if result['recommendations']:
            print("优化建议:")
            for recommendation in result['recommendations']:
                print(f"  - {recommendation}")
        _save_output(result, args.output or 'status_result.json')
        
    elif args.mode == 'stop':
        print("正在关闭系统...")
        result = controller.stop_system()
        print(f"系统状态: {result['status']}")
        print(f"清理状态: {result['cleanup_status']}")
        _save_output(result, args.output or 'shutdown_result.json')
    
    print("=" * 60)
    print("操作完成!")

def _load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")
        return {}

def _load_data(data_path):
    """加载数据文件"""
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 无法加载数据文件 {data_path}: {e}")
        return {}

def _save_output(data, output_path):
    """保存输出结果"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"警告: 无法保存输出文件 {output_path}: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n系统已被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"系统错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)