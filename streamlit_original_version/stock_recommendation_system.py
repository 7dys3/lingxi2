import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
import datetime
from typing import List, Dict, Tuple, Any, Optional

# 添加数据API路径
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("警告: 无法导入ApiClient，将使用模拟数据")

class StockRecommendationSystem:
    """基于历史走势的股票推荐系统"""
    
    def __init__(self, api_client=None):
        """初始化推荐系统
        
        Args:
            api_client: YahooFinance API客户端
        """
        self.api_client = api_client
        self.stock_data = {}  # 存储股票历史数据
        self.technical_indicators = {}  # 存储计算的技术指标
        self.similarity_matrix = None  # 股票相似度矩阵
        self.win_rates = {}  # 存储计算的胜率
        
        # 默认股票列表（可扩展）
        self.default_stocks = [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
            'TSLA', 'NVDA', 'JPM', 'V', 'PG',
            '600519.SS', '000858.SZ', '601318.SS', '600036.SS', '000333.SZ'
        ]
        
        # 创建数据目录
        os.makedirs('data/stock_data', exist_ok=True)
        os.makedirs('data/recommendations', exist_ok=True)
        
    def fetch_stock_data(self, symbols: List[str] = None, period: str = '1y', interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """获取股票历史数据
        
        Args:
            symbols: 股票代码列表，如果为None则使用默认列表
            period: 数据周期，如'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            interval: 数据间隔，如'1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
            
        Returns:
            股票数据字典，键为股票代码，值为DataFrame
        """
        if symbols is None:
            symbols = self.default_stocks
            
        result = {}
        
        for symbol in symbols:
            try:
                # 检查是否有缓存数据
                cache_file = f'data/stock_data/{symbol}_{period}_{interval}.csv'
                if os.path.exists(cache_file):
                    # 检查缓存是否过期（超过1天）
                    if datetime.datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        result[symbol] = df
                        print(f"从缓存加载 {symbol} 数据")
                        continue
                
                if HAS_API_CLIENT and self.api_client:
                    # 使用YahooFinance API获取数据
                    data = self.api_client.call_api('YahooFinance/get_stock_chart', 
                                                   query={'symbol': symbol, 
                                                         'interval': interval, 
                                                         'range': period})
                    
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        chart_data = data['chart']['result'][0]
                        
                        # 提取时间戳和价格数据
                        timestamps = chart_data.get('timestamp', [])
                        quotes = chart_data.get('indicators', {}).get('quote', [{}])[0]
                        
                        if timestamps and 'close' in quotes:
                            # 转换时间戳为日期
                            dates = [datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
                            
                            # 创建DataFrame
                            df = pd.DataFrame({
                                'open': quotes.get('open', [None] * len(timestamps)),
                                'high': quotes.get('high', [None] * len(timestamps)),
                                'low': quotes.get('low', [None] * len(timestamps)),
                                'close': quotes.get('close', [None] * len(timestamps)),
                                'volume': quotes.get('volume', [None] * len(timestamps))
                            }, index=dates)
                            
                            # 处理缺失值
                            df = df.dropna()
                            
                            # 保存到缓存
                            df.to_csv(cache_file)
                            
                            result[symbol] = df
                            print(f"成功获取 {symbol} 数据")
                        else:
                            print(f"获取 {symbol} 数据失败：数据结构不完整")
                    else:
                        print(f"获取 {symbol} 数据失败：API返回错误")
                else:
                    # 生成模拟数据（仅用于测试）
                    print(f"使用模拟数据代替 {symbol}")
                    dates = pd.date_range(end=datetime.datetime.now(), periods=252, freq='B')
                    np.random.seed(hash(symbol) % 10000)
                    
                    # 生成随机价格走势
                    close = np.random.randn(len(dates)).cumsum() + 100
                    # 确保价格为正
                    close = np.maximum(close, 1)
                    
                    # 生成其他价格数据
                    high = close * (1 + 0.02 * np.random.rand(len(dates)))
                    low = close * (1 - 0.02 * np.random.rand(len(dates)))
                    open_price = low + np.random.rand(len(dates)) * (high - low)
                    volume = np.random.randint(100000, 10000000, size=len(dates))
                    
                    df = pd.DataFrame({
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    }, index=dates)
                    
                    # 保存到缓存
                    df.to_csv(cache_file)
                    
                    result[symbol] = df
            except Exception as e:
                print(f"获取 {symbol} 数据时出错: {str(e)}")
        
        # 存储数据以供后续使用
        self.stock_data.update(result)
        
        return result
    
    def calculate_technical_indicators(self, symbols: List[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """计算技术指标
        
        Args:
            symbols: 股票代码列表，如果为None则使用已加载的所有股票
            
        Returns:
            技术指标字典，键为股票代码，值为包含各指标的字典
        """
        if symbols is None:
            symbols = list(self.stock_data.keys())
            
        result = {}
        
        for symbol in symbols:
            if symbol not in self.stock_data:
                print(f"未找到 {symbol} 的数据，跳过计算技术指标")
                continue
                
            df = self.stock_data[symbol]
            
            # 计算移动平均线
            ma5 = df['close'].rolling(window=5).mean().values
            ma10 = df['close'].rolling(window=10).mean().values
            ma20 = df['close'].rolling(window=20).mean().values
            ma60 = df['close'].rolling(window=60).mean().values
            
            # 计算MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            # 计算RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # 计算布林带
            ma20 = df['close'].rolling(window=20).mean()
            std20 = df['close'].rolling(window=20).std()
            upper_band = ma20 + 2 * std20
            lower_band = ma20 - 2 * std20
            
            # 计算成交量变化
            volume_ma5 = df['volume'].rolling(window=5).mean()
            volume_change = df['volume'] / volume_ma5
            
            # 存储计算结果
            result[symbol] = {
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'ma60': ma60,
                'macd': macd.values,
                'macd_signal': signal.values,
                'macd_histogram': histogram.values,
                'rsi': rsi.values,
                'upper_band': upper_band.values,
                'lower_band': lower_band.values,
                'volume_change': volume_change.values
            }
            
        # 存储指标以供后续使用
        self.technical_indicators.update(result)
        
        return result
    
    def calculate_win_rate(self, symbols: List[str] = None, n_days: int = 5, target_return: float = 0.03) -> Dict[str, float]:
        """计算股票在n天内达到目标收益率的胜率
        
        Args:
            symbols: 股票代码列表，如果为None则使用已加载的所有股票
            n_days: 预测天数
            target_return: 目标收益率
            
        Returns:
            胜率字典，键为股票代码，值为胜率（0-1之间）
        """
        if symbols is None:
            symbols = list(self.stock_data.keys())
            
        result = {}
        
        for symbol in symbols:
            if symbol not in self.stock_data:
                print(f"未找到 {symbol} 的数据，跳过计算胜率")
                continue
                
            df = self.stock_data[symbol]
            prices = df['close'].values
            
            total_samples = len(prices) - n_days
            if total_samples <= 0:
                result[symbol] = 0
                continue
            
            win_count = 0
            for i in range(total_samples):
                start_price = prices[i]
                end_prices = prices[i+1:i+n_days+1]
                max_return = max([(p - start_price) / start_price for p in end_prices])
                if max_return >= target_return:
                    win_count += 1
            
            win_rate = win_count / total_samples
            result[symbol] = win_rate
            
        # 存储胜率以供后续使用
        self.win_rates.update(result)
        
        return result
    
    def calculate_stock_similarity(self, symbols: List[str] = None) -> np.ndarray:
        """计算股票之间的相似度
        
        Args:
            symbols: 股票代码列表，如果为None则使用已加载的所有股票
            
        Returns:
            相似度矩阵
        """
        if symbols is None:
            symbols = list(self.stock_data.keys())
            
        # 提取每只股票的特征
        features = []
        valid_symbols = []
        
        for symbol in symbols:
            if symbol not in self.stock_data or symbol not in self.technical_indicators:
                continue
                
            # 提取价格走势特征
            df = self.stock_data[symbol]
            returns = df['close'].pct_change().dropna().values
            
            # 提取技术指标特征
            indicators = self.technical_indicators[symbol]
            
            # 组合特征
            feature_vector = np.concatenate([
                returns[-20:],  # 最近20天的收益率
                indicators['rsi'][-20:],  # 最近20天的RSI
                indicators['macd'][-20:],  # 最近20天的MACD
                indicators['volume_change'][-20:]  # 最近20天的成交量变化
            ])
            
            # 处理缺失值
            feature_vector = np.nan_to_num(feature_vector, nan=0)
            
            features.append(feature_vector)
            valid_symbols.append(symbol)
            
        if not features:
            return np.array([])
            
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # 计算余弦相似度
        similarity_matrix = cosine_similarity(scaled_features)
        
        # 存储相似度矩阵和有效股票列表
        self.similarity_matrix = similarity_matrix
        self.valid_symbols = valid_symbols
        
        return similarity_matrix
    
    def recommend_stocks(self, top_n: int = 5, min_win_rate: float = 0.5) -> List[Dict[str, Any]]:
        """推荐股票
        
        Args:
            top_n: 推荐的股票数量
            min_win_rate: 最小胜率要求
            
        Returns:
            推荐股票列表，每个元素为包含股票信息的字典
        """
        if not self.win_rates:
            print("未计算胜率，无法推荐股票")
            return []
            
        # 筛选胜率达到要求的股票
        qualified_stocks = {symbol: win_rate for symbol, win_rate in self.win_rates.items() 
                           if win_rate >= min_win_rate}
        
        if not qualified_stocks:
            print(f"没有胜率达到 {min_win_rate} 的股票")
            return []
            
        # 按胜率排序
        sorted_stocks = sorted(qualified_stocks.items(), key=lambda x: x[1], reverse=True)
        
        # 获取技术指标信号
        recommendations = []
        
        for symbol, win_rate in sorted_stocks[:top_n]:
            if symbol not in self.stock_data or symbol not in self.technical_indicators:
                continue
                
            df = self.stock_data[symbol]
            indicators = self.technical_indicators[symbol]
            
            # 获取最新价格和技术指标
            latest_price = df['close'].iloc[-1]
            latest_ma5 = indicators['ma5'][-1]
            latest_ma20 = indicators['ma20'][-1]
            latest_rsi = indicators['rsi'][-1]
            latest_macd = indicators['macd'][-1]
            latest_signal = indicators['macd_signal'][-1]
            
            # 生成技术信号
            signals = []
            
            # MA信号
            if latest_ma5 > latest_ma20:
                signals.append("MA5上穿MA20，短期趋势向上")
            elif latest_ma5 < latest_ma20:
                signals.append("MA5下穿MA20，短期趋势向下")
                
            # RSI信号
            if latest_rsi < 30:
                signals.append("RSI低于30，可能超卖")
            elif latest_rsi > 70:
                signals.append("RSI高于70，可能超买")
                
            # MACD信号
            if latest_macd > latest_signal:
                signals.append("MACD金叉，买入信号")
            elif latest_macd < latest_signal:
                signals.append("MACD死叉，卖出信号")
                
            # 查找相似股票
            similar_stocks = []
            if hasattr(self, 'similarity_matrix') and self.similarity_matrix is not None and hasattr(self, 'valid_symbols'):
                try:
                    idx = self.valid_symbols.index(symbol)
                    similarities = self.similarity_matrix[idx]
                    similar_indices = similarities.argsort()[-4:-1][::-1]  # 排除自身，取前3个最相似的
                    similar_stocks = [self.valid_symbols[i] for i in similar_indices]
                except (ValueError, IndexError):
                    pass
            
            # 创建推荐信息
            recommendation = {
                'symbol': symbol,
                'win_rate': win_rate,
                'latest_price': latest_price,
                'signals': signals,
                'similar_stocks': similar_stocks,
                'recommendation_date': datetime.datetime.now().strftime('%Y-%m-%d')
            }
            
            recommendations.append(recommendation)
            
        # 保存推荐结果
        self._save_recommendations(recommendations)
            
        return recommendations
    
    def _save_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """保存推荐结果
        
        Args:
            recommendations: 推荐股票列表
        """
        if not recommendations:
            return
            
        # 创建保存文件名
        filename = f"data/recommendations/stock_recommendations_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        
        # 转换为可序列化的格式
        serializable_recs = []
        for rec in recommendations:
            serializable_rec = rec.copy()
            serializable_rec['latest_price'] = float(serializable_rec['latest_price'])
            serializable_recs.append(serializable_rec)
            
        # 保存为JSON
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_recs, f, ensure_ascii=False, indent=2)
            
        print(f"推荐结果已保存到 {filename}")
    
    def plot_stock_chart(self, symbol: str, with_indicators: bool = True) -> str:
        """绘制股票图表
        
        Args:
            symbol: 股票代码
            with_indicators: 是否显示技术指标
            
        Returns:
            保存的图表文件路径
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法绘制图表")
            return ""
            
        df = self.stock_data[symbol]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线图
        axes[0].plot(df.index, df['close'], label='收盘价')
        
        if with_indicators and symbol in self.technical_indicators:
            indicators = self.technical_indicators[symbol]
            
            # 绘制移动平均线
            axes[0].plot(df.index, indicators['ma5'], label='MA5', alpha=0.7)
            axes[0].plot(df.index, indicators['ma20'], label='MA20', alpha=0.7)
            
            # 绘制布林带
            axes[0].plot(df.index, indicators['upper_band'], 'r--', label='上轨', alpha=0.5)
            axes[0].plot(df.index, indicators['lower_band'], 'g--', label='下轨', alpha=0.5)
            
            # 绘制MACD
            axes[1].plot(df.index, indicators['macd'], label='MACD')
            axes[1].plot(df.index, indicators['macd_signal'], label='Signal')
            axes[1].bar(df.index, indicators['macd_histogram'], label='Histogram', alpha=0.5)
            
        # 设置图表属性
        axes[0].set_title(f"{symbol} 股票图表")
        axes[0].set_ylabel("价格")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_xlabel("日期")
        axes[1].set_ylabel("MACD")
        axes[1].legend()
        axes[1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('data/charts', exist_ok=True)
        chart_file = f"data/charts/{symbol}_chart_{datetime.datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_file)
        plt.close()
        
        return chart_file
    
    def run_recommendation_pipeline(self, symbols: List[str] = None, top_n: int = 5, 
                                   n_days: int = 5, target_return: float = 0.03) -> Tuple[List[Dict[str, Any]], List[str]]:
        """运行完整的推荐流程
        
        Args:
            symbols: 股票代码列表，如果为None则使用默认列表
            top_n: 推荐的股票数量
            n_days: 预测天数
            target_return: 目标收益率
            
        Returns:
            推荐股票列表和生成的图表文件路径列表
        """
        # 1. 获取股票数据
        self.fetch_stock_data(symbols)
        
        # 2. 计算技术指标
        self.calculate_technical_indicators()
        
        # 3. 计算胜率
        self.calculate_win_rate(n_days=n_days, target_return=target_return)
        
        # 4. 计算股票相似度
        self.calculate_stock_similarity()
        
        # 5. 生成推荐
        recommendations = self.recommend_stocks(top_n=top_n)
        
        # 6. 绘制推荐股票的图表
        chart_files = []
        for rec in recommendations:
            chart_file = self.plot_stock_chart(rec['symbol'])
            if chart_file:
                chart_files.append(chart_file)
                
        return recommendations, chart_files


# 测试代码
if __name__ == "__main__":
    # 初始化API客户端（如果可用）
    api_client = None
    if HAS_API_CLIENT:
        api_client = ApiClient()
    
    # 创建推荐系统
    recommender = StockRecommendationSystem(api_client)
    
    # 运行推荐流程
    recommendations, chart_files = recommender.run_recommendation_pipeline(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', '600519.SS', '000858.SZ'],
        top_n=3
    )
    
    # 打印推荐结果
    print("\n推荐股票:")
    for rec in recommendations:
        print(f"股票: {rec['symbol']}")
        print(f"胜率: {rec['win_rate']:.2%}")
        print(f"最新价格: {rec['latest_price']}")
        print(f"技术信号: {', '.join(rec['signals'])}")
        print(f"相似股票: {', '.join(rec['similar_stocks'])}")
        print("-" * 50)
    
    # 打印生成的图表文件
    print("\n生成的图表文件:")
    for chart_file in chart_files:
        print(chart_file)
