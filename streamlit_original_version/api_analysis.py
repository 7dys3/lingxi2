"""
金融API能力分析脚本

本脚本用于分析YahooFinance API的功能和数据结构，为金融智能分析平台的改进提供基础。
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 创建输出目录
os.makedirs('api_test_results', exist_ok=True)

# 初始化API客户端
client = ApiClient()

def save_json(data, filename):
    """将JSON数据保存到文件"""
    with open(f'api_test_results/{filename}', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到 api_test_results/{filename}")

def test_stock_chart_api():
    """测试股票图表API"""
    print("\n===== 测试 get_stock_chart API =====")
    
    # 测试不同股票
    stocks = ['AAPL', 'MSFT', 'BABA', '600519.SS']  # 美股和A股
    
    for stock in stocks:
        print(f"\n获取 {stock} 的图表数据...")
        try:
            # 获取1年的月度数据
            chart_data = client.call_api('YahooFinance/get_stock_chart', 
                                        query={'symbol': stock, 
                                              'interval': '1mo', 
                                              'range': '1y',
                                              'includeAdjustedClose': True})
            
            # 保存原始数据
            save_json(chart_data, f'{stock}_chart_data.json')
            
            # 分析数据结构
            if 'chart' in chart_data and 'result' in chart_data['chart'] and chart_data['chart']['result']:
                result = chart_data['chart']['result'][0]
                
                # 提取元数据
                meta = result.get('meta', {})
                print(f"  货币: {meta.get('currency')}")
                print(f"  交易所: {meta.get('exchangeName')}")
                print(f"  当前价格: {meta.get('regularMarketPrice')}")
                print(f"  52周最高: {meta.get('fiftyTwoWeekHigh')}")
                print(f"  52周最低: {meta.get('fiftyTwoWeekLow')}")
                
                # 提取时间序列数据
                timestamps = result.get('timestamp', [])
                indicators = result.get('indicators', {})
                quotes = indicators.get('quote', [{}])[0]
                
                if timestamps and 'close' in quotes:
                    # 创建DataFrame
                    df = pd.DataFrame({
                        'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                        'open': quotes.get('open', []),
                        'high': quotes.get('high', []),
                        'low': quotes.get('low', []),
                        'close': quotes.get('close', []),
                        'volume': quotes.get('volume', [])
                    })
                    
                    # 保存为CSV
                    df.to_csv(f'api_test_results/{stock}_price_data.csv', index=False)
                    print(f"  获取到 {len(df)} 条价格记录")
                    
                    # 绘制收盘价走势图
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['timestamp'], df['close'])
                    plt.title(f'{stock} 收盘价走势')
                    plt.xlabel('日期')
                    plt.ylabel('价格')
                    plt.grid(True)
                    plt.savefig(f'api_test_results/{stock}_price_chart.png')
                    plt.close()
                    print(f"  价格走势图已保存")
                else:
                    print(f"  未找到 {stock} 的时间序列数据")
            else:
                print(f"  获取 {stock} 的图表数据失败，返回结构不完整")
        
        except Exception as e:
            print(f"  获取 {stock} 的图表数据时出错: {str(e)}")

def test_stock_holders_api():
    """测试股票持有者API"""
    print("\n===== 测试 get_stock_holders API =====")
    
    stocks = ['AAPL', 'MSFT', 'BABA']
    
    for stock in stocks:
        print(f"\n获取 {stock} 的持有者数据...")
        try:
            holders_data = client.call_api('YahooFinance/get_stock_holders', 
                                          query={'symbol': stock})
            
            save_json(holders_data, f'{stock}_holders_data.json')
            
            if 'quoteSummary' in holders_data and 'result' in holders_data['quoteSummary']:
                results = holders_data['quoteSummary']['result']
                if results and 'insiderHolders' in results[0]:
                    insider_holders = results[0]['insiderHolders']
                    holders = insider_holders.get('holders', [])
                    
                    print(f"  内部持有者数量: {len(holders)}")
                    
                    if holders:
                        # 创建DataFrame
                        holders_df = pd.DataFrame([{
                            'name': h.get('name', ''),
                            'relation': h.get('relation', ''),
                            'transaction': h.get('transactionDescription', ''),
                            'position': h.get('positionDirect', {}).get('fmt', '')
                        } for h in holders])
                        
                        # 保存为CSV
                        holders_df.to_csv(f'api_test_results/{stock}_insider_holders.csv', index=False)
                        print(f"  内部持有者数据已保存")
                else:
                    print(f"  未找到 {stock} 的内部持有者数据")
            else:
                print(f"  获取 {stock} 的持有者数据失败，返回结构不完整")
        
        except Exception as e:
            print(f"  获取 {stock} 的持有者数据时出错: {str(e)}")

def test_stock_insights_api():
    """测试股票洞察API"""
    print("\n===== 测试 get_stock_insights API =====")
    
    stocks = ['AAPL', 'MSFT', 'BABA']
    
    for stock in stocks:
        print(f"\n获取 {stock} 的洞察数据...")
        try:
            insights_data = client.call_api('YahooFinance/get_stock_insights', 
                                           query={'symbol': stock})
            
            save_json(insights_data, f'{stock}_insights_data.json')
            
            if 'finance' in insights_data and 'result' in insights_data['finance']:
                result = insights_data['finance']['result']
                
                # 提取技术指标
                if 'instrumentInfo' in result and 'technicalEvents' in result['instrumentInfo']:
                    tech_events = result['instrumentInfo']['technicalEvents']
                    print("  技术指标:")
                    
                    # 短期展望
                    if 'shortTermOutlook' in tech_events:
                        short_term = tech_events['shortTermOutlook']
                        print(f"    短期展望: {short_term.get('stateDescription')}")
                        print(f"    方向: {short_term.get('direction')}")
                        print(f"    评分: {short_term.get('score')}")
                    
                    # 中期展望
                    if 'intermediateTermOutlook' in tech_events:
                        mid_term = tech_events['intermediateTermOutlook']
                        print(f"    中期展望: {mid_term.get('stateDescription')}")
                        print(f"    方向: {mid_term.get('direction')}")
                        print(f"    评分: {mid_term.get('score')}")
                    
                    # 长期展望
                    if 'longTermOutlook' in tech_events:
                        long_term = tech_events['longTermOutlook']
                        print(f"    长期展望: {long_term.get('stateDescription')}")
                        print(f"    方向: {long_term.get('direction')}")
                        print(f"    评分: {long_term.get('score')}")
                
                # 提取公司快照
                if 'companySnapshot' in result:
                    snapshot = result['companySnapshot']
                    if 'company' in snapshot:
                        company = snapshot['company']
                        print("  公司指标:")
                        print(f"    创新性: {company.get('innovativeness')}")
                        print(f"    招聘: {company.get('hiring')}")
                        print(f"    可持续性: {company.get('sustainability')}")
                        print(f"    内部情绪: {company.get('insiderSentiments')}")
                
                # 提取重大发展
                if 'sigDevs' in result:
                    sig_devs = result['sigDevs']
                    print(f"  重大发展数量: {len(sig_devs)}")
                    if sig_devs:
                        # 创建DataFrame
                        sig_devs_df = pd.DataFrame([{
                            'date': dev.get('date', ''),
                            'headline': dev.get('headline', '')
                        } for dev in sig_devs])
                        
                        # 保存为CSV
                        sig_devs_df.to_csv(f'api_test_results/{stock}_significant_developments.csv', index=False)
                        print(f"  重大发展数据已保存")
            else:
                print(f"  获取 {stock} 的洞察数据失败，返回结构不完整")
        
        except Exception as e:
            print(f"  获取 {stock} 的洞察数据时出错: {str(e)}")

def test_stock_sec_filing_api():
    """测试股票SEC文件API"""
    print("\n===== 测试 get_stock_sec_filing API =====")
    
    stocks = ['AAPL', 'MSFT', 'BABA']
    
    for stock in stocks:
        print(f"\n获取 {stock} 的SEC文件数据...")
        try:
            sec_data = client.call_api('YahooFinance/get_stock_sec_filing', 
                                      query={'symbol': stock})
            
            save_json(sec_data, f'{stock}_sec_filing_data.json')
            
            if 'quoteSummary' in sec_data and 'result' in sec_data['quoteSummary']:
                results = sec_data['quoteSummary']['result']
                if results and 'secFilings' in results[0]:
                    sec_filings = results[0]['secFilings']
                    filings = sec_filings.get('filings', [])
                    
                    print(f"  SEC文件数量: {len(filings)}")
                    
                    if filings:
                        # 创建DataFrame
                        filings_df = pd.DataFrame([{
                            'date': f.get('date', ''),
                            'type': f.get('type', ''),
                            'title': f.get('title', ''),
                            'url': f.get('edgarUrl', '')
                        } for f in filings])
                        
                        # 保存为CSV
                        filings_df.to_csv(f'api_test_results/{stock}_sec_filings.csv', index=False)
                        print(f"  SEC文件数据已保存")
                else:
                    print(f"  未找到 {stock} 的SEC文件数据")
            else:
                print(f"  获取 {stock} 的SEC文件数据失败，返回结构不完整")
        
        except Exception as e:
            print(f"  获取 {stock} 的SEC文件数据时出错: {str(e)}")

def test_stock_analyst_api():
    """测试股票分析师评价API"""
    print("\n===== 测试 get_stock_what_analyst_are_saying API =====")
    
    stocks = ['AAPL', 'MSFT', 'BABA']
    
    for stock in stocks:
        print(f"\n获取 {stock} 的分析师评价数据...")
        try:
            analyst_data = client.call_api('YahooFinance/get_stock_what_analyst_are_saying', 
                                          query={'symbol': stock})
            
            save_json(analyst_data, f'{stock}_analyst_data.json')
            
            if 'result' in analyst_data:
                results = analyst_data['result']
                for result in results:
                    if 'hits' in result:
                        hits = result['hits']
                        print(f"  分析师报告数量: {len(hits)}")
                        
                        if hits:
                            # 创建DataFrame
                            reports_df = pd.DataFrame([{
                                'title': h.get('report_title', ''),
                                'author': h.get('author', ''),
                                'provider': h.get('provider', ''),
                                'date': datetime.fromtimestamp(h.get('report_date', 0)).strftime('%Y-%m-%d') if h.get('report_date') else '',
                                'abstract': h.get('abstract', '')
                            } for h in hits])
                            
                            # 保存为CSV
                            reports_df.to_csv(f'api_test_results/{stock}_analyst_reports.csv', index=False)
                            print(f"  分析师报告数据已保存")
                    else:
                        print(f"  未找到 {stock} 的分析师报告数据")
            else:
                print(f"  获取 {stock} 的分析师评价数据失败，返回结构不完整")
        
        except Exception as e:
            print(f"  获取 {stock} 的分析师评价数据时出错: {str(e)}")

def main():
    """主函数，运行所有API测试"""
    print("开始测试YahooFinance API...")
    
    # 测试各个API
    test_stock_chart_api()
    test_stock_holders_api()
    test_stock_insights_api()
    test_stock_sec_filing_api()
    test_stock_analyst_api()
    
    print("\n所有API测试完成！")

if __name__ == "__main__":
    main()
