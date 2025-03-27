import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import datetime
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.patches as patches
from scipy.signal import argrelextrema
from scipy import stats

# 添加数据API路径
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("警告: 无法导入ApiClient，将使用模拟数据")

class ChartAnalysisSystem:
    """图表分析标识功能"""
    
    def __init__(self, api_client=None):
        """初始化图表分析系统
        
        Args:
            api_client: YahooFinance API客户端
        """
        self.api_client = api_client
        self.stock_data = {}  # 存储股票历史数据
        self.patterns = {}  # 存储识别的形态
        self.support_resistance = {}  # 存储支撑位和阻力位
        self.trend_lines = {}  # 存储趋势线
        
        # 创建数据目录
        os.makedirs('data/chart_analysis', exist_ok=True)
        
    def fetch_stock_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """获取股票历史数据
        
        Args:
            symbol: 股票代码
            period: 数据周期，如'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            interval: 数据间隔，如'1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo'
            
        Returns:
            股票数据DataFrame
        """
        try:
            # 检查是否有缓存数据
            cache_file = f'data/chart_analysis/{symbol}_{period}_{interval}.csv'
            if os.path.exists(cache_file):
                # 检查缓存是否过期（超过1天）
                if datetime.datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    self.stock_data[symbol] = df
                    print(f"从缓存加载 {symbol} 数据")
                    return df
            
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
                        
                        self.stock_data[symbol] = df
                        print(f"成功获取 {symbol} 数据")
                        return df
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
                
                self.stock_data[symbol] = df
                return df
                
        except Exception as e:
            print(f"获取 {symbol} 数据时出错: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, symbol: str) -> Dict[str, np.ndarray]:
        """计算技术指标
        
        Args:
            symbol: 股票代码
            
        Returns:
            技术指标字典
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法计算技术指标")
            return {}
            
        df = self.stock_data[symbol]
        
        # 计算移动平均线
        ma5 = df['close'].rolling(window=5).mean()
        ma10 = df['close'].rolling(window=10).mean()
        ma20 = df['close'].rolling(window=20).mean()
        ma60 = df['close'].rolling(window=60).mean()
        
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
        indicators = {
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'ma60': ma60,
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram,
            'rsi': rsi,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'volume_change': volume_change
        }
        
        return indicators
    
    def identify_support_resistance(self, symbol: str, window: int = 20, threshold: float = 0.02) -> Dict[str, List[Tuple[int, float]]]:
        """识别支撑位和阻力位
        
        Args:
            symbol: 股票代码
            window: 局部极值窗口大小
            threshold: 合并相近价格水平的阈值
            
        Returns:
            包含支撑位和阻力位的字典
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法识别支撑位和阻力位")
            return {'supports': [], 'resistances': []}
            
        df = self.stock_data[symbol]
        prices = df['close'].values
        
        supports = []
        resistances = []
        
        # 使用局部最小值识别支撑位
        for i in range(window, len(prices) - window):
            if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                supports.append((i, prices[i]))
        
        # 使用局部最大值识别阻力位
        for i in range(window, len(prices) - window):
            if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                resistances.append((i, prices[i]))
        
        # 合并相近的支撑位和阻力位
        supports = self._merge_levels(supports, threshold)
        resistances = self._merge_levels(resistances, threshold)
        
        # 存储结果
        self.support_resistance[symbol] = {
            'supports': supports,
            'resistances': resistances
        }
        
        return self.support_resistance[symbol]
    
    def _merge_levels(self, levels: List[Tuple[int, float]], threshold: float) -> List[Tuple[int, float]]:
        """合并相近的价格水平
        
        Args:
            levels: 价格水平列表，每个元素为(索引, 价格)
            threshold: 合并阈值
            
        Returns:
            合并后的价格水平列表
        """
        if not levels:
            return []
        
        # 按价格排序
        levels.sort(key=lambda x: x[1])
        merged = [levels[0]]
        
        for i in range(1, len(levels)):
            last = merged[-1]
            current = levels[i]
            
            # 如果当前水平与上一个水平相差不超过阈值，则合并
            if abs(current[1] - last[1]) / last[1] <= threshold:
                # 取平均位置和价格
                merged[-1] = ((last[0] + current[0]) // 2, (last[1] + current[1]) / 2)
            else:
                merged.append(current)
        
        return merged
    
    def identify_head_and_shoulders(self, symbol: str, window: int = 20) -> List[Dict[str, Any]]:
        """识别头肩顶/底形态
        
        Args:
            symbol: 股票代码
            window: 局部极值窗口大小
            
        Returns:
            识别到的头肩顶/底形态列表
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法识别头肩顶/底形态")
            return []
            
        df = self.stock_data[symbol]
        prices = df['close'].values
        
        # 找到局部极值点
        local_max_indices = list(argrelextrema(prices, np.greater, order=window)[0])
        local_min_indices = list(argrelextrema(prices, np.less, order=window)[0])
        
        # 识别头肩顶形态
        top_patterns = []
        for i in range(len(local_max_indices) - 2):
            # 获取三个连续的局部最大值
            left_shoulder_idx = local_max_indices[i]
            head_idx = local_max_indices[i + 1]
            right_shoulder_idx = local_max_indices[i + 2]
            
            left_shoulder = prices[left_shoulder_idx]
            head = prices[head_idx]
            right_shoulder = prices[right_shoulder_idx]
            
            # 检查头肩顶条件
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                # 找到颈线（两肩之间的低点）
                neck_indices = [idx for idx in local_min_indices 
                               if left_shoulder_idx < idx < head_idx or head_idx < idx < right_shoulder_idx]
                
                if len(neck_indices) >= 2:
                    neck_left_idx = next((idx for idx in neck_indices if left_shoulder_idx < idx < head_idx), None)
                    neck_right_idx = next((idx for idx in neck_indices if head_idx < idx < right_shoulder_idx), None)
                    
                    if neck_left_idx is not None and neck_right_idx is not None:
                        neck_left = prices[neck_left_idx]
                        neck_right = prices[neck_right_idx]
                        neckline = (neck_left + neck_right) / 2
                        
                        # 添加识别到的形态
                        pattern = {
                            'type': 'head_and_shoulders_top',
                            'left_shoulder': (left_shoulder_idx, left_shoulder),
                            'head': (head_idx, head),
                            'right_shoulder': (right_shoulder_idx, right_shoulder),
                            'neck_left': (neck_left_idx, neck_left),
                            'neck_right': (neck_right_idx, neck_right),
                            'neckline': neckline,
                            'target': neckline - (head - neckline)  # 价格目标
                        }
                        
                        top_patterns.append(pattern)
        
        # 识别头肩底形态
        bottom_patterns = []
        for i in range(len(local_min_indices) - 2):
            # 获取三个连续的局部最小值
            left_shoulder_idx = local_min_indices[i]
            head_idx = local_min_indices[i + 1]
            right_shoulder_idx = local_min_indices[i + 2]
            
            left_shoulder = prices[left_shoulder_idx]
            head = prices[head_idx]
            right_shoulder = prices[right_shoulder_idx]
            
            # 检查头肩底条件
            if (head < left_shoulder and head < right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1):
                
                # 找到颈线（两肩之间的高点）
                neck_indices = [idx for idx in local_max_indices 
                               if left_shoulder_idx < idx < head_idx or head_idx < idx < right_shoulder_idx]
                
                if len(neck_indices) >= 2:
                    neck_left_idx = next((idx for idx in neck_indices if left_shoulder_idx < idx < head_idx), None)
                    neck_right_idx = next((idx for idx in neck_indices if head_idx < idx < right_shoulder_idx), None)
                    
                    if neck_left_idx is not None and neck_right_idx is not None:
                        neck_left = prices[neck_left_idx]
                        neck_right = prices[neck_right_idx]
                        neckline = (neck_left + neck_right) / 2
                        
                        # 添加识别到的形态
                        pattern = {
                            'type': 'head_and_shoulders_bottom',
                            'left_shoulder': (left_shoulder_idx, left_shoulder),
                            'head': (head_idx, head),
                            'right_shoulder': (right_shoulder_idx, right_shoulder),
                            'neck_left': (neck_left_idx, neck_left),
                            'neck_right': (neck_right_idx, neck_right),
                            'neckline': neckline,
                            'target': neckline + (neckline - head)  # 价格目标
                        }
                        
                        bottom_patterns.append(pattern)
        
        # 合并所有形态
        patterns = top_patterns + bottom_patterns
        
        # 存储结果
        if symbol not in self.patterns:
            self.patterns[symbol] = []
        self.patterns[symbol].extend(patterns)
        
        return patterns
    
    def identify_double_top_bottom(self, symbol: str, window: int = 20, threshold: float = 0.03) -> List[Dict[str, Any]]:
        """识别双顶/双底形态
        
        Args:
            symbol: 股票代码
            window: 局部极值窗口大小
            threshold: 两个顶/底之间的最大价差阈值
            
        Returns:
            识别到的双顶/双底形态列表
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法识别双顶/双底形态")
            return []
            
        df = self.stock_data[symbol]
        prices = df['close'].values
        
        # 找到局部极值点
        local_max_indices = list(argrelextrema(prices, np.greater, order=window)[0])
        local_min_indices = list(argrelextrema(prices, np.less, order=window)[0])
        
        # 识别双顶形态
        double_top_patterns = []
        for i in range(len(local_max_indices) - 1):
            # 获取两个连续的局部最大值
            first_top_idx = local_max_indices[i]
            second_top_idx = local_max_indices[i + 1]
            
            # 确保两个顶点之间有足够的距离
            if second_top_idx - first_top_idx < window * 2:
                continue
                
            first_top = prices[first_top_idx]
            second_top = prices[second_top_idx]
            
            # 检查双顶条件
            if abs(first_top - second_top) / first_top <= threshold:
                # 找到两个顶点之间的低点
                middle_min_indices = [idx for idx in local_min_indices if first_top_idx < idx < second_top_idx]
                
                if middle_min_indices:
                    middle_min_idx = middle_min_indices[0]  # 取第一个低点
                    middle_min = prices[middle_min_idx]
                    
                    # 添加识别到的形态
                    pattern = {
                        'type': 'double_top',
                        'first_top': (first_top_idx, first_top),
                        'second_top': (second_top_idx, second_top),
                        'middle_trough': (middle_min_idx, middle_min),
                        'neckline': middle_min,
                        'target': middle_min - (first_top - middle_min)  # 价格目标
                    }
                    
                    double_top_patterns.append(pattern)
        
        # 识别双底形态
        double_bottom_patterns = []
        for i in range(len(local_min_indices) - 1):
            # 获取两个连续的局部最小值
            first_bottom_idx = local_min_indices[i]
            second_bottom_idx = local_min_indices[i + 1]
            
            # 确保两个底点之间有足够的距离
            if second_bottom_idx - first_bottom_idx < window * 2:
                continue
                
            first_bottom = prices[first_bottom_idx]
            second_bottom = prices[second_bottom_idx]
            
            # 检查双底条件
            if abs(first_bottom - second_bottom) / first_bottom <= threshold:
                # 找到两个底点之间的高点
                middle_max_indices = [idx for idx in local_max_indices if first_bottom_idx < idx < second_bottom_idx]
                
                if middle_max_indices:
                    middle_max_idx = middle_max_indices[0]  # 取第一个高点
                    middle_max = prices[middle_max_idx]
                    
                    # 添加识别到的形态
                    pattern = {
                        'type': 'double_bottom',
                        'first_bottom': (first_bottom_idx, first_bottom),
                        'second_bottom': (second_bottom_idx, second_bottom),
                        'middle_peak': (middle_max_idx, middle_max),
                        'neckline': middle_max,
                        'target': middle_max + (middle_max - first_bottom)  # 价格目标
                    }
                    
                    double_bottom_patterns.append(pattern)
        
        # 合并所有形态
        patterns = double_top_patterns + double_bottom_patterns
        
        # 存储结果
        if symbol not in self.patterns:
            self.patterns[symbol] = []
        self.patterns[symbol].extend(patterns)
        
        return patterns
    
    def identify_triangles(self, symbol: str, window: int = 10, min_points: int = 5) -> List[Dict[str, Any]]:
        """识别三角形整理形态
        
        Args:
            symbol: 股票代码
            window: 局部极值窗口大小
            min_points: 形成三角形所需的最小点数
            
        Returns:
            识别到的三角形形态列表
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法识别三角形形态")
            return []
            
        df = self.stock_data[symbol]
        prices = df['close'].values
        
        # 找到局部极值点
        local_max_indices = list(argrelextrema(prices, np.greater, order=window)[0])
        local_min_indices = list(argrelextrema(prices, np.less, order=window)[0])
        
        # 合并并排序所有极值点
        extrema_indices = sorted(local_max_indices + local_min_indices)
        
        # 识别三角形形态
        triangle_patterns = []
        
        # 至少需要5个点才能形成三角形
        if len(extrema_indices) >= min_points:
            # 尝试不同的起始点
            for start_idx in range(len(extrema_indices) - min_points + 1):
                # 获取连续的极值点
                points_indices = extrema_indices[start_idx:start_idx + min_points]
                points = [(idx, prices[idx]) for idx in points_indices]
                
                # 分离高点和低点
                highs = [(idx, price) for idx, price in points if idx in local_max_indices]
                lows = [(idx, price) for idx, price in points if idx in local_min_indices]
                
                # 需要至少2个高点和2个低点
                if len(highs) < 2 or len(lows) < 2:
                    continue
                
                # 计算高点的趋势线
                high_indices = [idx for idx, _ in highs]
                high_prices = [price for _, price in highs]
                high_slope, high_intercept, high_r, _, _ = stats.linregress(high_indices, high_prices)
                
                # 计算低点的趋势线
                low_indices = [idx for idx, _ in lows]
                low_prices = [price for _, price in lows]
                low_slope, low_intercept, low_r, _, _ = stats.linregress(low_indices, low_prices)
                
                # 检查三角形条件
                
                # 对称三角形：高点向下倾斜，低点向上倾斜
                if high_slope < -0.01 and low_slope > 0.01 and abs(high_r) > 0.7 and abs(low_r) > 0.7:
                    # 计算交点
                    if high_slope != low_slope:
                        x_intersect = (low_intercept - high_intercept) / (high_slope - low_slope)
                        y_intersect = high_slope * x_intersect + high_intercept
                        
                        # 确保交点在未来
                        if x_intersect > max(points_indices):
                            pattern = {
                                'type': 'symmetric_triangle',
                                'highs': highs,
                                'lows': lows,
                                'high_slope': high_slope,
                                'high_intercept': high_intercept,
                                'low_slope': low_slope,
                                'low_intercept': low_intercept,
                                'intersect': (x_intersect, y_intersect)
                            }
                            triangle_patterns.append(pattern)
                
                # 上升三角形：高点水平，低点向上倾斜
                elif abs(high_slope) < 0.01 and low_slope > 0.01 and abs(low_r) > 0.7:
                    # 计算突破目标
                    avg_high = sum(high_prices) / len(high_prices)
                    target = avg_high + (avg_high - low_prices[0])
                    
                    pattern = {
                        'type': 'ascending_triangle',
                        'highs': highs,
                        'lows': lows,
                        'high_slope': high_slope,
                        'high_intercept': high_intercept,
                        'low_slope': low_slope,
                        'low_intercept': low_intercept,
                        'target': target
                    }
                    triangle_patterns.append(pattern)
                
                # 下降三角形：高点向下倾斜，低点水平
                elif high_slope < -0.01 and abs(low_slope) < 0.01 and abs(high_r) > 0.7:
                    # 计算突破目标
                    avg_low = sum(low_prices) / len(low_prices)
                    target = avg_low - (high_prices[0] - avg_low)
                    
                    pattern = {
                        'type': 'descending_triangle',
                        'highs': highs,
                        'lows': lows,
                        'high_slope': high_slope,
                        'high_intercept': high_intercept,
                        'low_slope': low_slope,
                        'low_intercept': low_intercept,
                        'target': target
                    }
                    triangle_patterns.append(pattern)
        
        # 存储结果
        if symbol not in self.patterns:
            self.patterns[symbol] = []
        self.patterns[symbol].extend(triangle_patterns)
        
        return triangle_patterns
    
    def draw_trendline(self, symbol: str, window: int = 20, is_support: bool = True) -> Dict[str, Any]:
        """自动绘制趋势线
        
        Args:
            symbol: 股票代码
            window: 局部极值窗口大小
            is_support: 是否为支撑线
            
        Returns:
            趋势线信息
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法绘制趋势线")
            return {}
            
        df = self.stock_data[symbol]
        prices = df['close'].values
        x = np.array(range(len(prices)))
        
        # 找到局部极值点
        extrema = []
        if is_support:
            # 支撑线使用局部最小值
            local_min_indices = list(argrelextrema(prices, np.less, order=window)[0])
            extrema = [(idx, prices[idx]) for idx in local_min_indices]
        else:
            # 阻力线使用局部最大值
            local_max_indices = list(argrelextrema(prices, np.greater, order=window)[0])
            extrema = [(idx, prices[idx]) for idx in local_max_indices]
        
        if len(extrema) < 2:
            return {}
        
        # 使用RANSAC算法找到最佳拟合线
        from sklearn.linear_model import RANSACRegressor
        
        X = np.array([p[0] for p in extrema]).reshape(-1, 1)
        Y = np.array([p[1] for p in extrema])
        
        ransac = RANSACRegressor()
        ransac.fit(X, Y)
        
        # 获取趋势线参数
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        
        # 计算趋势线上的点
        line_x = np.array(range(len(prices)))
        line_y = slope * line_x + intercept
        
        # 存储趋势线
        trendline = {
            'slope': slope,
            'intercept': intercept,
            'points': list(zip(line_x, line_y)),
            'is_support': is_support
        }
        
        if symbol not in self.trend_lines:
            self.trend_lines[symbol] = []
        self.trend_lines[symbol].append(trendline)
        
        return trendline
    
    def plot_chart_with_analysis(self, symbol: str) -> str:
        """绘制带有分析标识的股票图表
        
        Args:
            symbol: 股票代码
            
        Returns:
            保存的图表文件路径
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法绘制图表")
            return ""
            
        df = self.stock_data[symbol]
        
        # 计算技术指标
        indicators = self.calculate_technical_indicators(symbol)
        
        # 识别支撑位和阻力位
        self.identify_support_resistance(symbol)
        
        # 识别形态
        self.identify_head_and_shoulders(symbol)
        self.identify_double_top_bottom(symbol)
        self.identify_triangles(symbol)
        
        # 绘制趋势线
        self.draw_trendline(symbol, is_support=True)
        self.draw_trendline(symbol, is_support=False)
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # 绘制K线图
        axes[0].plot(df.index, df['close'], label='收盘价')
        
        # 绘制移动平均线
        if 'ma5' in indicators:
            axes[0].plot(df.index, indicators['ma5'], label='MA5', alpha=0.7)
        if 'ma20' in indicators:
            axes[0].plot(df.index, indicators['ma20'], label='MA20', alpha=0.7)
        if 'ma60' in indicators:
            axes[0].plot(df.index, indicators['ma60'], label='MA60', alpha=0.7)
        
        # 绘制布林带
        if 'upper_band' in indicators and 'lower_band' in indicators:
            axes[0].plot(df.index, indicators['upper_band'], 'r--', label='上轨', alpha=0.5)
            axes[0].plot(df.index, indicators['lower_band'], 'g--', label='下轨', alpha=0.5)
        
        # 绘制支撑位和阻力位
        if symbol in self.support_resistance:
            sr = self.support_resistance[symbol]
            
            for idx, level in sr.get('supports', []):
                axes[0].axhline(y=level, color='g', linestyle='-', alpha=0.5)
                axes[0].text(df.index[min(idx + 5, len(df.index) - 1)], level, f'支撑: {level:.2f}', 
                           color='g', alpha=0.8)
            
            for idx, level in sr.get('resistances', []):
                axes[0].axhline(y=level, color='r', linestyle='-', alpha=0.5)
                axes[0].text(df.index[min(idx + 5, len(df.index) - 1)], level, f'阻力: {level:.2f}', 
                           color='r', alpha=0.8)
        
        # 绘制趋势线
        if symbol in self.trend_lines:
            for trendline in self.trend_lines[symbol]:
                line_x = [p[0] for p in trendline['points']]
                line_y = [p[1] for p in trendline['points']]
                
                color = 'g' if trendline['is_support'] else 'r'
                label = '支撑趋势线' if trendline['is_support'] else '阻力趋势线'
                
                axes[0].plot(df.index[line_x], line_y, color=color, linestyle='--', alpha=0.7, label=label)
        
        # 标注形态
        if symbol in self.patterns:
            for pattern in self.patterns[symbol]:
                pattern_type = pattern['type']
                
                if pattern_type == 'head_and_shoulders_top':
                    # 绘制头肩顶
                    left_shoulder = pattern['left_shoulder']
                    head = pattern['head']
                    right_shoulder = pattern['right_shoulder']
                    
                    # 连接三个点
                    x_points = [df.index[left_shoulder[0]], df.index[head[0]], df.index[right_shoulder[0]]]
                    y_points = [left_shoulder[1], head[1], right_shoulder[1]]
                    
                    axes[0].plot(x_points, y_points, 'ro-', alpha=0.7)
                    axes[0].text(df.index[head[0]], head[1] * 1.05, '头肩顶', color='r')
                    
                    # 绘制颈线
                    neckline = pattern['neckline']
                    axes[0].axhline(y=neckline, color='r', linestyle='--', alpha=0.5)
                    
                    # 标注目标价格
                    target = pattern['target']
                    axes[0].axhline(y=target, color='r', linestyle=':', alpha=0.5)
                    axes[0].text(df.index[-1], target, f'目标: {target:.2f}', color='r')
                
                elif pattern_type == 'head_and_shoulders_bottom':
                    # 绘制头肩底
                    left_shoulder = pattern['left_shoulder']
                    head = pattern['head']
                    right_shoulder = pattern['right_shoulder']
                    
                    # 连接三个点
                    x_points = [df.index[left_shoulder[0]], df.index[head[0]], df.index[right_shoulder[0]]]
                    y_points = [left_shoulder[1], head[1], right_shoulder[1]]
                    
                    axes[0].plot(x_points, y_points, 'go-', alpha=0.7)
                    axes[0].text(df.index[head[0]], head[1] * 0.95, '头肩底', color='g')
                    
                    # 绘制颈线
                    neckline = pattern['neckline']
                    axes[0].axhline(y=neckline, color='g', linestyle='--', alpha=0.5)
                    
                    # 标注目标价格
                    target = pattern['target']
                    axes[0].axhline(y=target, color='g', linestyle=':', alpha=0.5)
                    axes[0].text(df.index[-1], target, f'目标: {target:.2f}', color='g')
                
                elif pattern_type == 'double_top':
                    # 绘制双顶
                    first_top = pattern['first_top']
                    second_top = pattern['second_top']
                    
                    # 连接两个顶点
                    x_points = [df.index[first_top[0]], df.index[second_top[0]]]
                    y_points = [first_top[1], second_top[1]]
                    
                    axes[0].plot(x_points, y_points, 'ro-', alpha=0.7)
                    axes[0].text(df.index[second_top[0]], second_top[1] * 1.05, '双顶', color='r')
                    
                    # 绘制颈线
                    neckline = pattern['neckline']
                    axes[0].axhline(y=neckline, color='r', linestyle='--', alpha=0.5)
                    
                    # 标注目标价格
                    target = pattern['target']
                    axes[0].axhline(y=target, color='r', linestyle=':', alpha=0.5)
                    axes[0].text(df.index[-1], target, f'目标: {target:.2f}', color='r')
                
                elif pattern_type == 'double_bottom':
                    # 绘制双底
                    first_bottom = pattern['first_bottom']
                    second_bottom = pattern['second_bottom']
                    
                    # 连接两个底点
                    x_points = [df.index[first_bottom[0]], df.index[second_bottom[0]]]
                    y_points = [first_bottom[1], second_bottom[1]]
                    
                    axes[0].plot(x_points, y_points, 'go-', alpha=0.7)
                    axes[0].text(df.index[second_bottom[0]], second_bottom[1] * 0.95, '双底', color='g')
                    
                    # 绘制颈线
                    neckline = pattern['neckline']
                    axes[0].axhline(y=neckline, color='g', linestyle='--', alpha=0.5)
                    
                    # 标注目标价格
                    target = pattern['target']
                    axes[0].axhline(y=target, color='g', linestyle=':', alpha=0.5)
                    axes[0].text(df.index[-1], target, f'目标: {target:.2f}', color='g')
                
                elif 'triangle' in pattern_type:
                    # 绘制三角形
                    highs = pattern['highs']
                    lows = pattern['lows']
                    
                    # 绘制高点趋势线
                    high_x = [df.index[idx] for idx, _ in highs]
                    high_y = [price for _, price in highs]
                    
                    # 绘制低点趋势线
                    low_x = [df.index[idx] for idx, _ in lows]
                    low_y = [price for _, price in lows]
                    
                    # 使用趋势线方程绘制延长线
                    x_range = np.array(range(min(highs[0][0], lows[0][0]), len(df)))
                    high_line = pattern['high_slope'] * x_range + pattern['high_intercept']
                    low_line = pattern['low_slope'] * x_range + pattern['low_intercept']
                    
                    axes[0].plot(df.index[x_range], high_line, 'r--', alpha=0.7)
                    axes[0].plot(df.index[x_range], low_line, 'g--', alpha=0.7)
                    
                    # 标注三角形类型
                    triangle_type_map = {
                        'symmetric_triangle': '对称三角形',
                        'ascending_triangle': '上升三角形',
                        'descending_triangle': '下降三角形'
                    }
                    
                    triangle_type = triangle_type_map.get(pattern_type, '三角形')
                    mid_idx = (highs[0][0] + lows[-1][0]) // 2
                    mid_price = (highs[0][1] + lows[-1][1]) / 2
                    
                    axes[0].text(df.index[mid_idx], mid_price, triangle_type, 
                               color='b', bbox=dict(facecolor='white', alpha=0.7))
                    
                    # 标注目标价格
                    if 'target' in pattern:
                        target = pattern['target']
                        axes[0].axhline(y=target, color='b', linestyle=':', alpha=0.5)
                        axes[0].text(df.index[-1], target, f'目标: {target:.2f}', color='b')
        
        # 绘制MACD
        if all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
            axes[1].plot(df.index, indicators['macd'], label='MACD')
            axes[1].plot(df.index, indicators['macd_signal'], label='Signal')
            axes[1].bar(df.index, indicators['macd_histogram'], label='Histogram', alpha=0.5)
            axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.2)
        
        # 设置图表属性
        axes[0].set_title(f"{symbol} 技术分析图表")
        axes[0].set_ylabel("价格")
        axes[0].legend(loc='upper left')
        axes[0].grid(True)
        
        axes[1].set_xlabel("日期")
        axes[1].set_ylabel("MACD")
        axes[1].legend(loc='upper left')
        axes[1].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('data/chart_analysis', exist_ok=True)
        chart_file = f"data/chart_analysis/{symbol}_analysis_{datetime.datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_file)
        plt.close()
        
        return chart_file
    
    def generate_analysis_report(self, symbol: str) -> Dict[str, Any]:
        """生成技术分析报告
        
        Args:
            symbol: 股票代码
            
        Returns:
            分析报告字典
        """
        if symbol not in self.stock_data:
            print(f"未找到 {symbol} 的数据，无法生成分析报告")
            return {}
            
        df = self.stock_data[symbol]
        
        # 确保已经进行了所有分析
        indicators = self.calculate_technical_indicators(symbol)
        self.identify_support_resistance(symbol)
        self.identify_head_and_shoulders(symbol)
        self.identify_double_top_bottom(symbol)
        self.identify_triangles(symbol)
        self.draw_trendline(symbol, is_support=True)
        self.draw_trendline(symbol, is_support=False)
        
        # 获取最新价格和技术指标
        latest_price = df['close'].iloc[-1]
        latest_ma5 = indicators['ma5'].iloc[-1] if 'ma5' in indicators else None
        latest_ma20 = indicators['ma20'].iloc[-1] if 'ma20' in indicators else None
        latest_ma60 = indicators['ma60'].iloc[-1] if 'ma60' in indicators else None
        latest_rsi = indicators['rsi'].iloc[-1] if 'rsi' in indicators else None
        latest_macd = indicators['macd'].iloc[-1] if 'macd' in indicators else None
        latest_signal = indicators['macd_signal'].iloc[-1] if 'macd_signal' in indicators else None
        
        # 生成技术信号
        signals = []
        
        # MA信号
        if latest_ma5 is not None and latest_ma20 is not None:
            if latest_ma5 > latest_ma20:
                signals.append({"type": "bullish", "description": "MA5上穿MA20，短期趋势向上"})
            elif latest_ma5 < latest_ma20:
                signals.append({"type": "bearish", "description": "MA5下穿MA20，短期趋势向下"})
        
        if latest_ma20 is not None and latest_ma60 is not None:
            if latest_ma20 > latest_ma60:
                signals.append({"type": "bullish", "description": "MA20上穿MA60，中期趋势向上"})
            elif latest_ma20 < latest_ma60:
                signals.append({"type": "bearish", "description": "MA20下穿MA60，中期趋势向下"})
        
        # RSI信号
        if latest_rsi is not None:
            if latest_rsi < 30:
                signals.append({"type": "bullish", "description": "RSI低于30，可能超卖"})
            elif latest_rsi > 70:
                signals.append({"type": "bearish", "description": "RSI高于70，可能超买"})
        
        # MACD信号
        if latest_macd is not None and latest_signal is not None:
            if latest_macd > latest_signal:
                signals.append({"type": "bullish", "description": "MACD金叉，买入信号"})
            elif latest_macd < latest_signal:
                signals.append({"type": "bearish", "description": "MACD死叉，卖出信号"})
        
        # 支撑位和阻力位分析
        support_resistance_analysis = []
        if symbol in self.support_resistance:
            sr = self.support_resistance[symbol]
            
            # 找到最近的支撑位
            supports = sorted([(level, abs(level - latest_price)) for _, level in sr.get('supports', [])], 
                             key=lambda x: x[1])
            
            # 找到最近的阻力位
            resistances = sorted([(level, abs(level - latest_price)) for _, level in sr.get('resistances', [])], 
                                key=lambda x: x[1])
            
            if supports:
                nearest_support = supports[0][0]
                support_distance = (latest_price - nearest_support) / latest_price * 100
                support_resistance_analysis.append({
                    "type": "support",
                    "level": nearest_support,
                    "distance": f"{support_distance:.2f}%",
                    "description": f"最近支撑位: {nearest_support:.2f}，距当前价格 {support_distance:.2f}%"
                })
            
            if resistances:
                nearest_resistance = resistances[0][0]
                resistance_distance = (nearest_resistance - latest_price) / latest_price * 100
                support_resistance_analysis.append({
                    "type": "resistance",
                    "level": nearest_resistance,
                    "distance": f"{resistance_distance:.2f}%",
                    "description": f"最近阻力位: {nearest_resistance:.2f}，距当前价格 {resistance_distance:.2f}%"
                })
        
        # 形态分析
        pattern_analysis = []
        if symbol in self.patterns:
            for pattern in self.patterns[symbol]:
                pattern_type = pattern['type']
                
                # 形态类型映射
                pattern_type_map = {
                    'head_and_shoulders_top': '头肩顶',
                    'head_and_shoulders_bottom': '头肩底',
                    'double_top': '双顶',
                    'double_bottom': '双底',
                    'symmetric_triangle': '对称三角形',
                    'ascending_triangle': '上升三角形',
                    'descending_triangle': '下降三角形'
                }
                
                # 形态方向映射
                pattern_direction_map = {
                    'head_and_shoulders_top': 'bearish',
                    'head_and_shoulders_bottom': 'bullish',
                    'double_top': 'bearish',
                    'double_bottom': 'bullish',
                    'symmetric_triangle': 'neutral',
                    'ascending_triangle': 'bullish',
                    'descending_triangle': 'bearish'
                }
                
                chinese_type = pattern_type_map.get(pattern_type, pattern_type)
                direction = pattern_direction_map.get(pattern_type, 'neutral')
                
                # 获取目标价格
                target = pattern.get('target')
                target_distance = ((target - latest_price) / latest_price * 100) if target else None
                
                pattern_analysis.append({
                    "type": chinese_type,
                    "direction": direction,
                    "target": target,
                    "target_distance": f"{target_distance:.2f}%" if target_distance else None,
                    "description": f"识别到{chinese_type}形态，{'看涨' if direction == 'bullish' else '看跌' if direction == 'bearish' else '中性'}"
                                  + (f"，目标价格: {target:.2f}，距当前价格 {target_distance:.2f}%" if target else "")
                })
        
        # 趋势线分析
        trendline_analysis = []
        if symbol in self.trend_lines:
            for trendline in self.trend_lines[symbol]:
                is_support = trendline['is_support']
                slope = trendline['slope']
                
                # 计算当前趋势线价格
                current_idx = len(df) - 1
                current_trendline_price = slope * current_idx + trendline['intercept']
                
                # 计算与当前价格的距离
                distance = (latest_price - current_trendline_price) / latest_price * 100
                
                trendline_analysis.append({
                    "type": "support" if is_support else "resistance",
                    "slope": slope,
                    "current_level": current_trendline_price,
                    "distance": f"{distance:.2f}%",
                    "description": f"{'支撑' if is_support else '阻力'}趋势线，当前水平: {current_trendline_price:.2f}，"
                                  + f"距当前价格 {abs(distance):.2f}%，趋势{'向上' if slope > 0 else '向下' if slope < 0 else '水平'}"
                })
        
        # 综合分析
        bullish_signals = len([s for s in signals if s['type'] == 'bullish'])
        bearish_signals = len([s for s in signals if s['type'] == 'bearish'])
        
        if bullish_signals > bearish_signals:
            overall_bias = "bullish"
            overall_description = "总体偏向看涨，多头信号占优"
        elif bearish_signals > bullish_signals:
            overall_bias = "bearish"
            overall_description = "总体偏向看跌，空头信号占优"
        else:
            overall_bias = "neutral"
            overall_description = "多空信号均衡，建议观望"
        
        # 生成完整报告
        report = {
            "symbol": symbol,
            "analysis_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "latest_price": latest_price,
            "technical_indicators": {
                "ma5": latest_ma5,
                "ma20": latest_ma20,
                "ma60": latest_ma60,
                "rsi": latest_rsi,
                "macd": latest_macd,
                "macd_signal": latest_signal
            },
            "signals": signals,
            "support_resistance": support_resistance_analysis,
            "patterns": pattern_analysis,
            "trendlines": trendline_analysis,
            "overall": {
                "bias": overall_bias,
                "description": overall_description,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals
            }
        }
        
        # 保存报告
        report_file = f"data/chart_analysis/{symbol}_report_{datetime.datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def run_analysis_pipeline(self, symbol: str) -> Tuple[Dict[str, Any], str]:
        """运行完整的分析流程
        
        Args:
            symbol: 股票代码
            
        Returns:
            分析报告和图表文件路径
        """
        # 1. 获取股票数据
        self.fetch_stock_data(symbol)
        
        # 2. 绘制分析图表
        chart_file = self.plot_chart_with_analysis(symbol)
        
        # 3. 生成分析报告
        report = self.generate_analysis_report(symbol)
        
        return report, chart_file


# 测试代码
if __name__ == "__main__":
    # 初始化API客户端（如果可用）
    api_client = None
    if HAS_API_CLIENT:
        api_client = ApiClient()
    
    # 创建图表分析系统
    analyzer = ChartAnalysisSystem(api_client)
    
    # 运行分析流程
    symbol = 'AAPL'
    report, chart_file = analyzer.run_analysis_pipeline(symbol)
    
    # 打印分析结果
    print(f"\n{symbol} 技术分析报告:")
    print(f"分析日期: {report['analysis_date']}")
    print(f"最新价格: {report['latest_price']}")
    
    print("\n技术信号:")
    for signal in report['signals']:
        print(f"- {signal['description']}")
    
    print("\n支撑位和阻力位:")
    for sr in report['support_resistance']:
        print(f"- {sr['description']}")
    
    print("\n形态分析:")
    for pattern in report['patterns']:
        print(f"- {pattern['description']}")
    
    print("\n趋势线分析:")
    for trendline in report['trendlines']:
        print(f"- {trendline['description']}")
    
    print("\n综合分析:")
    print(f"- {report['overall']['description']}")
    print(f"- 看涨信号: {report['overall']['bullish_signals']}, 看跌信号: {report['overall']['bearish_signals']}")
    
    print(f"\n图表已保存到: {chart_file}")
