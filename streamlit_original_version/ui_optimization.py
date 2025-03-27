import os
import sys
import pandas as pd
import numpy as np
import json
import datetime
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import asyncio
import base64
from PIL import Image, ImageDraw, ImageFont
import io

# 导入自定义模块
sys.path.append('/home/ubuntu/financial_platform_improved')
try:
    from stock_recommendation_system import StockRecommendationSystem
    from chart_analysis_system import ChartAnalysisSystem
    from news_and_market_review_system import NewsAndMarketReviewSystem
    from enhanced_multi_model_service import EnhancedMultiModelService
except ImportError as e:
    print(f"导入自定义模块时出错: {str(e)}")

# 添加数据API路径
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("警告: 无法导入ApiClient，将使用模拟数据")

class UIOptimization:
    """UI界面优化"""
    
    def __init__(self):
        """初始化UI优化系统"""
        # 创建数据目录
        os.makedirs('data/ui', exist_ok=True)
        
        # 初始化API客户端
        self.api_client = None
        if HAS_API_CLIENT:
            self.api_client = ApiClient()
        
        # 初始化各个子系统
        self.stock_recommendation = StockRecommendationSystem(self.api_client)
        self.chart_analysis = ChartAnalysisSystem(self.api_client)
        self.news_system = NewsAndMarketReviewSystem(self.api_client)
        self.multi_model_service = EnhancedMultiModelService()
        
        # 设置主题颜色
        self.theme = {
            'primary': '#1E88E5',    # 主色调（蓝色）
            'secondary': '#26A69A',  # 次要色调（青色）
            'accent': '#FF8F00',     # 强调色（橙色）
            'warning': '#E53935',    # 警告色（红色）
            'success': '#43A047',    # 成功色（绿色）
            'background': '#F5F5F5', # 背景色（浅灰色）
            'text': '#212121',       # 文本色（深灰色）
            'text_secondary': '#757575', # 次要文本色（中灰色）
            'border': '#BDBDBD',     # 边框色（灰色）
            'chart_colors': ['#1E88E5', '#26A69A', '#FF8F00', '#E53935', '#43A047', 
                            '#7E57C2', '#D81B60', '#FFC107', '#5D4037', '#00ACC1']
        }
        
        # 加载中文字体
        self.font_path = self._get_chinese_font_path()
    
    def _get_chinese_font_path(self) -> str:
        """获取中文字体路径
        
        Returns:
            字体路径
        """
        # 尝试加载中文字体
        font_path = None
        for font in ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font))
                if font_path:
                    break
            except:
                continue
        
        if not font_path:
            print("警告: 未找到中文字体，可能无法正确显示中文")
            font_path = None
        
        return font_path
    
    def generate_logo(self, text: str = "金融智能分析平台", size: Tuple[int, int] = (200, 200)) -> str:
        """生成平台Logo
        
        Args:
            text: Logo文字
            size: Logo尺寸
            
        Returns:
            Base64编码的Logo图像
        """
        # 创建一个白色背景的图像
        img = Image.new('RGB', size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # 绘制圆形背景
        center = (size[0] // 2, size[1] // 2)
        radius = min(size) // 2 - 10
        draw.ellipse(
            [(center[0] - radius, center[1] - radius), 
             (center[0] + radius, center[1] + radius)], 
            fill=self.theme['primary']
        )
        
        # 尝试加载字体
        try:
            if self.font_path:
                font = ImageFont.truetype(self.font_path, size=radius // 2)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            print(f"加载字体时出错: {str(e)}")
            font = ImageFont.load_default()
        
        # 计算文本位置
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (radius, radius // 2)
        text_position = (center[0] - text_width // 2, center[1] - text_height // 2)
        
        # 绘制文本
        draw.text(text_position, text, fill=(255, 255, 255), font=font)
        
        # 转换为Base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
    
    def generate_css(self) -> str:
        """生成CSS样式
        
        Returns:
            CSS样式字符串
        """
        css = f"""
        <style>
            /* 全局样式 */
            body {{
                font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                color: {self.theme['text']};
                background-color: {self.theme['background']};
            }}
            
            /* 标题样式 */
            h1, h2, h3, h4, h5, h6 {{
                color: {self.theme['primary']};
                font-weight: bold;
            }}
            
            /* 卡片样式 */
            .card {{
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            
            /* 主要按钮 */
            .button-primary {{
                background-color: {self.theme['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .button-primary:hover {{
                background-color: #1976D2;
            }}
            
            /* 次要按钮 */
            .button-secondary {{
                background-color: {self.theme['secondary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .button-secondary:hover {{
                background-color: #00897B;
            }}
            
            /* 强调按钮 */
            .button-accent {{
                background-color: {self.theme['accent']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                cursor: pointer;
                transition: background-color 0.3s;
            }}
            .button-accent:hover {{
                background-color: #EF6C00;
            }}
            
            /* 标签 */
            .tag {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-right: 5px;
                margin-bottom: 5px;
            }}
            .tag-primary {{
                background-color: {self.theme['primary']};
                color: white;
            }}
            .tag-secondary {{
                background-color: {self.theme['secondary']};
                color: white;
            }}
            .tag-accent {{
                background-color: {self.theme['accent']};
                color: white;
            }}
            .tag-success {{
                background-color: {self.theme['success']};
                color: white;
            }}
            .tag-warning {{
                background-color: {self.theme['warning']};
                color: white;
            }}
            
            /* 表格样式 */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th {{
                background-color: {self.theme['primary']};
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid {self.theme['border']};
            }}
            tr:nth-child(even) {{
                background-color: rgba(0, 0, 0, 0.05);
            }}
            
            /* 股票涨跌样式 */
            .stock-up {{
                color: {self.theme['warning']};
                font-weight: bold;
            }}
            .stock-down {{
                color: {self.theme['success']};
                font-weight: bold;
            }}
            
            /* 导航栏 */
            .navbar {{
                background-color: {self.theme['primary']};
                padding: 15px;
                color: white;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                border-radius: 8px;
            }}
            .navbar-brand {{
                font-size: 24px;
                font-weight: bold;
            }}
            .navbar-links {{
                display: flex;
            }}
            .navbar-link {{
                color: white;
                margin-left: 20px;
                text-decoration: none;
                font-weight: bold;
            }}
            .navbar-link:hover {{
                text-decoration: underline;
            }}
            
            /* 页脚 */
            .footer {{
                background-color: {self.theme['text']};
                color: white;
                padding: 20px;
                text-align: center;
                margin-top: 40px;
                border-radius: 8px;
            }}
            
            /* 搜索框 */
            .search-box {{
                width: 100%;
                padding: 10px;
                border: 1px solid {self.theme['border']};
                border-radius: 4px;
                margin-bottom: 20px;
            }}
            
            /* 股票卡片 */
            .stock-card {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                border: 1px solid {self.theme['border']};
                border-radius: 8px;
                margin-bottom: 10px;
                transition: transform 0.3s;
            }}
            .stock-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }}
            .stock-info {{
                flex: 1;
            }}
            .stock-name {{
                font-weight: bold;
                font-size: 18px;
            }}
            .stock-price {{
                font-size: 24px;
                font-weight: bold;
                margin-left: 20px;
            }}
            
            /* 新闻卡片 */
            .news-card {{
                padding: 15px;
                border: 1px solid {self.theme['border']};
                border-radius: 8px;
                margin-bottom: 15px;
                transition: transform 0.3s;
            }}
            .news-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            }}
            .news-title {{
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 10px;
            }}
            .news-meta {{
                color: {self.theme['text_secondary']};
                font-size: 14px;
                margin-bottom: 10px;
            }}
            .news-content {{
                margin-bottom: 10px;
            }}
            
            /* 图表容器 */
            .chart-container {{
                width: 100%;
                height: 400px;
                margin-bottom: 20px;
            }}
            
            /* 统计卡片 */
            .stat-card {{
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .stat-value {{
                font-size: 36px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .stat-label {{
                color: {self.theme['text_secondary']};
                font-size: 14px;
            }}
            
            /* 热点标签云 */
            .hot-topic {{
                display: inline-block;
                padding: 8px 16px;
                margin: 5px;
                background-color: {self.theme['primary']};
                color: white;
                border-radius: 20px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.3s;
            }}
            .hot-topic:hover {{
                transform: scale(1.1);
            }}
            
            /* 自定义滚动条 */
            ::-webkit-scrollbar {{
                width: 8px;
            }}
            ::-webkit-scrollbar-track {{
                background: #f1f1f1;
            }}
            ::-webkit-scrollbar-thumb {{
                background: {self.theme['primary']};
                border-radius: 4px;
            }}
            ::-webkit-scrollbar-thumb:hover {{
                background: #1976D2;
            }}
            
            /* 响应式布局 */
            @media (max-width: 768px) {{
                .navbar {{
                    flex-direction: column;
                    align-items: flex-start;
                }}
                .navbar-links {{
                    margin-top: 10px;
                    flex-direction: column;
                }}
                .navbar-link {{
                    margin-left: 0;
                    margin-top: 10px;
                }}
                .stock-card {{
                    flex-direction: column;
                    align-items: flex-start;
                }}
                .stock-price {{
                    margin-left: 0;
                    margin-top: 10px;
                }}
            }}
        </style>
        """
        
        return css
    
    def create_streamlit_app(self):
        """创建Streamlit应用"""
        # 设置页面配置
        st.set_page_config(
            page_title="金融智能分析平台",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 注入CSS
        st.markdown(self.generate_css(), unsafe_allow_html=True)
        
        # 生成Logo
        logo_base64 = self.generate_logo()
        
        # 创建导航栏
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(f"data:image/png;base64,{logo_base64}", width=100)
        with col2:
            st.title("金融智能分析平台")
            st.markdown("集成AI驱动的股票分析、市场资讯和智能投顾服务")
        
        # 创建侧边栏导航
        st.sidebar.title("功能导航")
        page = st.sidebar.radio(
            "选择功能",
            ["首页", "股票推荐", "图表分析", "热点资讯", "智能投顾", "人工客服"]
        )
        
        # 根据选择的页面显示不同内容
        if page == "首页":
            self.render_home_page()
        elif page == "股票推荐":
            self.render_stock_recommendation_page()
        elif page == "图表分析":
            self.render_chart_analysis_page()
        elif page == "热点资讯":
            self.render_news_page()
        elif page == "智能投顾":
            self.render_advisor_page()
        elif page == "人工客服":
            self.render_customer_service_page()
        
        # 添加页脚
        st.markdown("""
        <div class="footer">
            <p>© 2025 金融智能分析平台 | 版权所有</p>
            <p>免责声明：本平台提供的信息仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_home_page(self):
        """渲染首页"""
        # 欢迎信息
        st.header("欢迎使用金融智能分析平台")
        st.markdown("""
        <div class="card">
            <h3>平台亮点</h3>
            <p>金融智能分析平台集成了多种先进的AI技术，为您提供全方位的金融市场分析和投资决策支持。</p>
            <ul>
                <li>基于历史走势的股票推荐系统</li>
                <li>智能图表分析与技术形态识别</li>
                <li>实时热点资讯与市场复盘</li>
                <li>个性化智能投顾服务</li>
                <li>专业人工客服支持</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 市场概览
        st.subheader("今日市场概览")
        
        # 获取市场复盘数据
        market_review = self.news_system.generate_market_review()
        
        # 显示主要指数
        col1, col2, col3 = st.columns(3)
        
        index_map = {
            '^GSPC': {'name': '标普500', 'col': col1},
            '^DJI': {'name': '道琼斯', 'col': col1},
            '000001.SS': {'name': '上证指数', 'col': col2},
            '399001.SZ': {'name': '深证成指', 'col': col2},
            '^IXIC': {'name': '纳斯达克', 'col': col3},
            '399006.SZ': {'name': '创业板指', 'col': col3}
        }
        
        for symbol, info in index_map.items():
            if symbol in market_review.get('index_changes', {}):
                index_data = market_review['index_changes'][symbol]
                direction_class = 'stock-up' if index_data['direction'] == 'up' else 'stock-down'
                change_sign = '+' if index_data['direction'] == 'up' else ''
                
                with info['col']:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">{info['name']}</div>
                        <div class="stat-value">{index_data['latest_close']:.2f}</div>
                        <div class="{direction_class}">{change_sign}{index_data['change_pct']:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 市场概览图表
        st.subheader("主要指数走势")
        market_chart = self.news_system.plot_market_overview()
        if market_chart and os.path.exists(market_chart):
            st.image(market_chart)
        
        # 行业板块表现
        st.subheader("行业板块表现")
        sector_chart = self.news_system.plot_sector_performance()
        if sector_chart and os.path.exists(sector_chart):
            st.image(sector_chart)
        
        # 热点话题
        st.subheader("今日热点话题")
        
        # 获取热点话题
        self.news_system.fetch_financial_news()
        hot_topics = self.news_system.analyze_hot_topics()
        
        # 显示热点话题
        hot_topics_html = '<div style="margin-bottom: 20px;">'
        for topic in hot_topics[:10]:
            size = min(24, max(14, 14 + topic['count']))
            hot_topics_html += f'<span class="hot-topic" style="font-size: {size}px;">{topic["keyword"]} ({topic["count"]})</span>'
        hot_topics_html += '</div>'
        
        st.markdown(hot_topics_html, unsafe_allow_html=True)
        
        # 生成词云
        wordcloud_file = self.news_system.generate_word_cloud()
        if wordcloud_file and os.path.exists(wordcloud_file):
            st.image(wordcloud_file)
        
        # 推荐股票
        st.subheader("今日推荐股票")
        
        # 获取推荐股票
        recommended_stocks = self.stock_recommendation.recommend_stocks(top_n=5)
        
        # 显示推荐股票
        for stock in recommended_stocks:
            direction_class = 'stock-up' if stock['change_percent'] > 0 else 'stock-down'
            change_sign = '+' if stock['change_percent'] > 0 else ''
            
            st.markdown(f"""
            <div class="stock-card">
                <div class="stock-info">
                    <div class="stock-name">{stock['name']} ({stock['symbol']})</div>
                    <div>推荐理由: {stock['recommendation_reason']}</div>
                    <div>胜率: {stock['win_rate']:.2f}%</div>
                </div>
                <div>
                    <div class="stock-price">{stock['current_price']:.2f}</div>
                    <div class="{direction_class}">{change_sign}{stock['change_percent']:.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 最新资讯
        st.subheader("最新财经资讯")
        
        # 获取最新新闻
        news_data = self.news_system.news_data[:5]
        
        # 显示最新新闻
        for news in news_data:
            st.markdown(f"""
            <div class="news-card">
                <div class="news-title">{news['title']}</div>
                <div class="news-meta">{news['source']} | {news['date']}</div>
                <div class="news-content">{news['content'][:200]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_stock_recommendation_page(self):
        """渲染股票推荐页面"""
        st.header("股票推荐系统")
        st.markdown("""
        <div class="card">
            <p>基于历史走势和技术指标分析，为您推荐具有较高胜率的股票。系统会综合考虑多种因素，包括价格趋势、成交量、技术指标和市场情绪等。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 筛选选项
        col1, col2, col3 = st.columns(3)
        with col1:
            market = st.selectbox("选择市场", ["全部", "A股", "港股", "美股"])
        with col2:
            industry = st.selectbox("选择行业", ["全部", "科技", "金融", "医药", "消费", "能源", "工业", "原材料"])
        with col3:
            sort_by = st.selectbox("排序方式", ["胜率", "上涨空间", "最新价格", "成交量"])
        
        # 高级筛选
        with st.expander("高级筛选"):
            col1, col2 = st.columns(2)
            with col1:
                min_win_rate = st.slider("最低胜率", 0, 100, 60)
                min_price = st.number_input("最低价格", 0.0, 10000.0, 0.0)
            with col2:
                time_frame = st.selectbox("时间周期", ["日线", "周线", "月线"])
                max_price = st.number_input("最高价格", 0.0, 10000.0, 10000.0)
        
        # 获取推荐股票
        recommended_stocks = self.stock_recommendation.recommend_stocks(
            market=market if market != "全部" else None,
            industry=industry if industry != "全部" else None,
            min_win_rate=min_win_rate,
            min_price=min_price,
            max_price=max_price,
            time_frame=time_frame,
            top_n=20
        )
        
        # 排序
        if sort_by == "胜率":
            recommended_stocks.sort(key=lambda x: x['win_rate'], reverse=True)
        elif sort_by == "上涨空间":
            recommended_stocks.sort(key=lambda x: x['potential_upside'], reverse=True)
        elif sort_by == "最新价格":
            recommended_stocks.sort(key=lambda x: x['current_price'], reverse=True)
        elif sort_by == "成交量":
            recommended_stocks.sort(key=lambda x: x['volume'], reverse=True)
        
        # 显示推荐股票
        st.subheader(f"推荐股票列表 (共 {len(recommended_stocks)} 只)")
        
        # 创建表格
        table_html = """
        <table>
            <tr>
                <th>股票名称</th>
                <th>代码</th>
                <th>最新价格</th>
                <th>涨跌幅</th>
                <th>胜率</th>
                <th>上涨空间</th>
                <th>推荐理由</th>
            </tr>
        """
        
        for stock in recommended_stocks:
            direction_class = 'stock-up' if stock['change_percent'] > 0 else 'stock-down'
            change_sign = '+' if stock['change_percent'] > 0 else ''
            
            table_html += f"""
            <tr>
                <td>{stock['name']}</td>
                <td>{stock['symbol']}</td>
                <td>{stock['current_price']:.2f}</td>
                <td class="{direction_class}">{change_sign}{stock['change_percent']:.2f}%</td>
                <td>{stock['win_rate']:.2f}%</td>
                <td>{stock['potential_upside']:.2f}%</td>
                <td>{stock['recommendation_reason']}</td>
            </tr>
            """
        
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)
        
        # 股票详情
        st.subheader("股票详细分析")
        selected_stock = st.selectbox("选择股票查看详细分析", 
                                     [f"{stock['name']} ({stock['symbol']})" for stock in recommended_stocks])
        
        if selected_stock:
            # 提取股票代码
            symbol = selected_stock.split('(')[1].split(')')[0]
            
            # 查找选中的股票
            stock_detail = None
            for stock in recommended_stocks:
                if stock['symbol'] == symbol:
                    stock_detail = stock
                    break
            
            if stock_detail:
                # 显示股票详情
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 显示股票走势图
                    st.markdown("### 股票走势图")
                    chart_img = self.stock_recommendation.plot_stock_chart(symbol)
                    if chart_img and os.path.exists(chart_img):
                        st.image(chart_img)
                    
                    # 显示技术指标
                    st.markdown("### 技术指标分析")
                    indicators_img = self.stock_recommendation.plot_technical_indicators(symbol)
                    if indicators_img and os.path.exists(indicators_img):
                        st.image(indicators_img)
                
                with col2:
                    # 显示股票基本信息
                    st.markdown("### 股票基本信息")
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>股票名称:</strong> {stock_detail['name']}</p>
                        <p><strong>股票代码:</strong> {stock_detail['symbol']}</p>
                        <p><strong>最新价格:</strong> {stock_detail['current_price']:.2f}</p>
                        <p><strong>涨跌幅:</strong> <span class="{direction_class}">{change_sign}{stock_detail['change_percent']:.2f}%</span></p>
                        <p><strong>成交量:</strong> {stock_detail['volume']:,}</p>
                        <p><strong>市值:</strong> {stock_detail.get('market_cap', 'N/A')}</p>
                        <p><strong>所属行业:</strong> {stock_detail.get('industry', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 显示推荐指标
                    st.markdown("### 推荐指标")
                    st.markdown(f"""
                    <div class="card">
                        <p><strong>胜率:</strong> {stock_detail['win_rate']:.2f}%</p>
                        <p><strong>上涨空间:</strong> {stock_detail['potential_upside']:.2f}%</p>
                        <p><strong>风险评级:</strong> {stock_detail.get('risk_rating', 'N/A')}</p>
                        <p><strong>推荐理由:</strong> {stock_detail['recommendation_reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 显示相关股票
                    st.markdown("### 相关股票")
                    related_stocks = stock_detail.get('related_stocks', [])
                    if related_stocks:
                        for related in related_stocks:
                            st.markdown(f"- {related['name']} ({related['symbol']})")
                    else:
                        st.markdown("暂无相关股票信息")
    
    def render_chart_analysis_page(self):
        """渲染图表分析页面"""
        st.header("图表分析系统")
        st.markdown("""
        <div class="card">
            <p>智能图表分析系统可以自动识别股票图表中的关键技术形态、支撑位/阻力位和趋势线，帮助您更好地理解市场走势和做出投资决策。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 股票选择
        col1, col2 = st.columns([3, 1])
        with col1:
            stock_input = st.text_input("输入股票代码或名称", "AAPL")
        with col2:
            time_frame = st.selectbox("选择时间周期", ["日线", "周线", "月线"])
        
        # 分析按钮
        analyze_button = st.button("分析图表")
        
        if analyze_button or stock_input:
            # 获取股票数据
            stock_data = self.chart_analysis.get_stock_data(stock_input)
            
            if stock_data is not None:
                # 显示股票基本信息
                st.subheader(f"{stock_data.get('name', stock_input)} ({stock_input}) 图表分析")
                
                # 识别技术形态
                patterns = self.chart_analysis.identify_patterns(stock_input)
                
                # 识别支撑位和阻力位
                support_resistance = self.chart_analysis.identify_support_resistance(stock_input)
                
                # 识别趋势线
                trend_lines = self.chart_analysis.identify_trend_lines(stock_input)
                
                # 显示带有标记的图表
                st.markdown("### 技术形态识别")
                chart_img = self.chart_analysis.plot_with_patterns(stock_input, patterns)
                if chart_img and os.path.exists(chart_img):
                    st.image(chart_img)
                
                # 显示支撑位和阻力位
                st.markdown("### 支撑位和阻力位")
                sr_img = self.chart_analysis.plot_with_support_resistance(stock_input, support_resistance)
                if sr_img and os.path.exists(sr_img):
                    st.image(sr_img)
                
                # 显示趋势线
                st.markdown("### 趋势线分析")
                trend_img = self.chart_analysis.plot_with_trend_lines(stock_input, trend_lines)
                if trend_img and os.path.exists(trend_img):
                    st.image(trend_img)
                
                # 显示分析结果
                st.subheader("分析结果")
                
                # 技术形态
                st.markdown("### 识别到的技术形态")
                if patterns:
                    for pattern in patterns:
                        confidence = pattern.get('confidence', 0) * 100
                        confidence_class = 'tag-success' if confidence >= 70 else 'tag-warning' if confidence >= 50 else 'tag-accent'
                        
                        st.markdown(f"""
                        <div class="card">
                            <h4>{pattern['name']}</h4>
                            <p>{pattern['description']}</p>
                            <p><strong>位置:</strong> {pattern['position']}</p>
                            <p><strong>信号类型:</strong> {pattern['signal_type']}</p>
                            <p><strong>置信度:</strong> <span class="{confidence_class}">{confidence:.1f}%</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("未识别到明显的技术形态")
                
                # 支撑位和阻力位
                st.markdown("### 支撑位和阻力位")
                if support_resistance:
                    # 支撑位
                    st.markdown("#### 支撑位")
                    if 'support' in support_resistance and support_resistance['support']:
                        for level in support_resistance['support']:
                            st.markdown(f"- 价格: {level['price']:.2f}, 强度: {level['strength']:.2f}")
                    else:
                        st.markdown("未识别到明显的支撑位")
                    
                    # 阻力位
                    st.markdown("#### 阻力位")
                    if 'resistance' in support_resistance and support_resistance['resistance']:
                        for level in support_resistance['resistance']:
                            st.markdown(f"- 价格: {level['price']:.2f}, 强度: {level['strength']:.2f}")
                    else:
                        st.markdown("未识别到明显的阻力位")
                else:
                    st.markdown("未识别到明显的支撑位和阻力位")
                
                # 趋势线
                st.markdown("### 趋势线分析")
                if trend_lines:
                    for trend in trend_lines:
                        st.markdown(f"""
                        <div class="card">
                            <h4>{trend['type']}趋势线</h4>
                            <p><strong>起始日期:</strong> {trend['start_date']}</p>
                            <p><strong>结束日期:</strong> {trend['end_date']}</p>
                            <p><strong>斜率:</strong> {trend['slope']:.4f}</p>
                            <p><strong>强度:</strong> {trend['strength']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("未识别到明显的趋势线")
                
                # 综合分析
                st.subheader("综合分析")
                analysis = self.chart_analysis.generate_analysis_summary(stock_input, patterns, support_resistance, trend_lines)
                
                st.markdown(f"""
                <div class="card">
                    <h4>市场状态: {analysis['market_state']}</h4>
                    <p><strong>短期趋势:</strong> {analysis['short_term_trend']}</p>
                    <p><strong>中期趋势:</strong> {analysis['medium_term_trend']}</p>
                    <p><strong>长期趋势:</strong> {analysis['long_term_trend']}</p>
                    <p><strong>波动性:</strong> {analysis['volatility']}</p>
                    <p><strong>成交量分析:</strong> {analysis['volume_analysis']}</p>
                    <p><strong>技术指标综合:</strong> {analysis['technical_summary']}</p>
                    <p><strong>关键价位:</strong> {analysis['key_price_levels']}</p>
                    <p><strong>综合建议:</strong> {analysis['recommendation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 风险提示
                st.markdown("""
                <div class="card" style="background-color: #FFF3E0; border-left: 5px solid #FF8F00;">
                    <h4>风险提示</h4>
                    <p>以上分析仅供参考，不构成投资建议。技术分析存在局限性，无法预测突发事件和基本面变化带来的影响。投资有风险，入市需谨慎。</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"无法获取股票 {stock_input} 的数据，请检查股票代码是否正确。")
    
    def render_news_page(self):
        """渲染热点资讯页面"""
        st.header("热点资讯与市场复盘")
        st.markdown("""
        <div class="card">
            <p>实时获取最新财经资讯，分析市场热点话题，提供每日市场复盘，帮助您把握市场脉搏和投资机会。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["热点资讯", "市场复盘", "热点分析"])
        
        # 热点资讯标签页
        with tab1:
            # 获取新闻数据
            self.news_system.fetch_financial_news()
            news_data = self.news_system.news_data
            
            # 筛选选项
            col1, col2 = st.columns(2)
            with col1:
                news_source = st.selectbox("新闻来源", ["全部", "东方财富网", "新浪财经", "其他"])
            with col2:
                news_category = st.selectbox("新闻分类", ["全部", "宏观经济", "股市", "债市", "外汇", "商品", "公司新闻"])
            
            # 搜索框
            search_query = st.text_input("搜索关键词")
            
            # 筛选新闻
            filtered_news = []
            for news in news_data:
                # 来源筛选
                if news_source != "全部" and news['source'] != news_source:
                    continue
                
                # 关键词搜索
                if search_query and search_query.lower() not in news['title'].lower() and search_query.lower() not in news['content'].lower():
                    continue
                
                filtered_news.append(news)
            
            # 显示新闻
            st.subheader(f"最新资讯 (共 {len(filtered_news)} 条)")
            
            for news in filtered_news:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">{news['title']}</div>
                    <div class="news-meta">{news['source']} | {news['date']}</div>
                    <div class="news-content">{news['content'][:300]}...</div>
                    <div>
                        <span class="tag tag-primary">财经</span>
                        {' '.join([f'<span class="tag tag-secondary">{keyword}</span>' for keyword in news.get('keywords', [])])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # 市场复盘标签页
        with tab2:
            # 获取市场复盘数据
            market_review = self.news_system.generate_market_review()
            
            # 日期选择
            selected_date = st.date_input("选择日期", datetime.datetime.now())
            
            # 市场趋势
            st.subheader("市场趋势")
            st.markdown(f"""
            <div class="card">
                <h3>今日市场: {market_review.get('market_trend', '未知')}</h3>
                <p>日期: {market_review.get('date', selected_date.strftime('%Y-%m-%d'))}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 主要指数表现
            st.subheader("主要指数表现")
            
            # 创建表格
            table_html = """
            <table>
                <tr>
                    <th>指数名称</th>
                    <th>最新收盘</th>
                    <th>涨跌幅</th>
                </tr>
            """
            
            index_map = {
                '^GSPC': '标普500',
                '^DJI': '道琼斯',
                '^IXIC': '纳斯达克',
                '^HSI': '恒生指数',
                '000001.SS': '上证指数',
                '399001.SZ': '深证成指',
                '399006.SZ': '创业板指'
            }
            
            for symbol, name in index_map.items():
                if symbol in market_review.get('index_changes', {}):
                    index_data = market_review['index_changes'][symbol]
                    direction_class = 'stock-up' if index_data['direction'] == 'up' else 'stock-down'
                    change_sign = '+' if index_data['direction'] == 'up' else ''
                    
                    table_html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{index_data['latest_close']:.2f}</td>
                        <td class="{direction_class}">{change_sign}{index_data['change_pct']:.2f}%</td>
                    </tr>
                    """
            
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            # 市场概览图表
            st.subheader("主要指数走势")
            market_chart = self.news_system.plot_market_overview()
            if market_chart and os.path.exists(market_chart):
                st.image(market_chart)
            
            # 行业板块表现
            st.subheader("行业板块表现")
            sector_chart = self.news_system.plot_sector_performance()
            if sector_chart and os.path.exists(sector_chart):
                st.image(sector_chart)
            
            # 市场亮点
            st.subheader("市场亮点")
            
            # 获取排名前三的行业
            top_sectors = market_review.get('top_sectors', [])
            bottom_sectors = market_review.get('bottom_sectors', [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 表现最佳行业")
                for sector in top_sectors:
                    st.markdown(f"- {sector}")
            
            with col2:
                st.markdown("#### 表现最差行业")
                for sector in bottom_sectors:
                    st.markdown(f"- {sector}")
            
            # 市场总结
            st.subheader("市场总结")
            
            # 生成市场总结（实际应用中应该有更详细的分析）
            market_summary = f"""
            今日市场整体呈{market_review.get('market_trend', '未知')}态势。主要指数中，
            {'多数上涨' if sum(1 for data in market_review.get('index_changes', {}).values() if data['direction'] == 'up') > len(market_review.get('index_changes', {})) / 2 else '多数下跌'}。
            行业板块方面，{', '.join(top_sectors[:2])}等行业表现较好，而{', '.join(bottom_sectors[:2])}等行业表现较弱。
            市场成交量{random.choice(['有所放大', '较为平稳', '有所萎缩'])}，
            资金面{random.choice(['相对宽松', '中性', '略显紧张'])}。
            短期来看，市场可能继续{random.choice(['震荡整理', '保持强势', '承压回调'])}，
            建议投资者{random.choice(['关注政策变化', '把握结构性机会', '控制仓位', '适度防御'])}。
            """
            
            st.markdown(f"""
            <div class="card">
                {market_summary}
            </div>
            """, unsafe_allow_html=True)
        
        # 热点分析标签页
        with tab3:
            # 获取热点话题
            hot_topics = self.news_system.analyze_hot_topics()
            
            # 热点词云
            st.subheader("热点词云")
            wordcloud_file = self.news_system.generate_word_cloud()
            if wordcloud_file and os.path.exists(wordcloud_file):
                st.image(wordcloud_file)
            
            # 热点话题列表
            st.subheader("热点话题分析")
            
            for topic in hot_topics[:10]:
                with st.expander(f"{topic['keyword']} (提及次数: {topic['count']})"):
                    # 相关新闻
                    st.markdown("#### 相关新闻")
                    for news in topic.get('related_news', []):
                        st.markdown(f"- {news['title']} - {news['source']} ({news['date']})")
                    
                    # 话题分析（实际应用中应该有更详细的分析）
                    st.markdown("#### 话题分析")
                    
                    # 生成随机分析（实际应用中应该基于真实数据）
                    topic_analysis = f"""
                    "{topic['keyword']}"相关话题近期受到市场广泛关注，提及次数达到{topic['count']}次。
                    从相关新闻来看，该话题主要与{random.choice(['政策变化', '行业动态', '公司业绩', '市场情绪', '技术突破'])}有关。
                    短期内，该话题可能{random.choice(['持续发酵', '逐渐降温', '引发市场波动', '带动相关板块表现'])}。
                    建议投资者{random.choice(['密切关注后续发展', '理性看待相关信息', '关注政策导向', '留意市场反应'])}。
                    """
                    
                    st.markdown(topic_analysis)
            
            # 热点板块
            st.subheader("热点板块追踪")
            
            # 模拟热点板块数据（实际应用中应该基于真实数据）
            hot_sectors = [
                {"name": "人工智能", "change_percent": 2.35, "hot_stocks": ["科大讯飞", "寒武纪", "中科曙光"]},
                {"name": "新能源车", "change_percent": 1.87, "hot_stocks": ["比亚迪", "宁德时代", "亿纬锂能"]},
                {"name": "半导体", "change_percent": 1.42, "hot_stocks": ["中芯国际", "韦尔股份", "北方华创"]},
                {"name": "医药生物", "change_percent": 0.95, "hot_stocks": ["恒瑞医药", "药明康德", "迈瑞医疗"]},
                {"name": "元宇宙", "change_percent": 0.78, "hot_stocks": ["腾讯控股", "网易", "完美世界"]}
            ]
            
            for sector in hot_sectors:
                direction_class = 'stock-up' if sector['change_percent'] > 0 else 'stock-down'
                change_sign = '+' if sector['change_percent'] > 0 else ''
                
                st.markdown(f"""
                <div class="card">
                    <h4>{sector['name']} <span class="{direction_class}">{change_sign}{sector['change_percent']}%</span></h4>
                    <p><strong>热门个股:</strong> {', '.join(sector['hot_stocks'])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_advisor_page(self):
        """渲染智能投顾页面"""
        st.header("智能投顾与财富管理")
        st.markdown("""
        <div class="card">
            <p>基于您的风险偏好、投资目标和财务状况，提供个性化的投资建议和资产配置方案，帮助您实现财富增值。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建标签页
        tab1, tab2, tab3 = st.tabs(["风险评估", "资产配置", "投资组合"])
        
        # 风险评估标签页
        with tab1:
            st.subheader("投资者风险评估")
            
            # 个人信息
            st.markdown("### 基本信息")
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("年龄", 18, 80, 35)
                income = st.selectbox("年收入(万元)", ["10以下", "10-30", "30-50", "50-100", "100以上"])
            with col2:
                investment_horizon = st.selectbox("投资期限", ["1年以内", "1-3年", "3-5年", "5-10年", "10年以上"])
                financial_assets = st.selectbox("金融资产(万元)", ["10以下", "10-50", "50-100", "100-500", "500以上"])
            
            # 风险偏好问卷
            st.markdown("### 风险偏好评估")
            
            q1 = st.radio(
                "1. 您的投资经验如何？",
                ["无经验", "1年以内", "1-3年", "3-5年", "5年以上"]
            )
            
            q2 = st.radio(
                "2. 您能接受的最大投资损失是多少？",
                ["不能接受任何损失", "5%以内", "10%以内", "20%以内", "30%以上"]
            )
            
            q3 = st.radio(
                "3. 如果您的投资在短期内下跌20%，您会怎么做？",
                ["立即全部卖出", "卖出一部分", "继续持有", "买入更多"]
            )
            
            q4 = st.radio(
                "4. 您更倾向于哪种类型的投资？",
                ["保本保息类产品", "固定收益类产品", "混合型产品", "股票型产品", "高风险高收益产品"]
            )
            
            q5 = st.radio(
                "5. 您的投资目标是什么？",
                ["保本保息", "稳健增值", "平衡增长", "积极增长", "追求最大收益"]
            )
            
            # 评估按钮
            if st.button("提交评估"):
                # 计算风险得分（实际应用中应该有更复杂的算法）
                risk_score = 0
                
                # 年龄得分
                risk_score += max(0, (60 - age) / 10)
                
                # 收入得分
                income_scores = {"10以下": 1, "10-30": 2, "30-50": 3, "50-100": 4, "100以上": 5}
                risk_score += income_scores.get(income, 0)
                
                # 投资期限得分
                horizon_scores = {"1年以内": 1, "1-3年": 2, "3-5年": 3, "5-10年": 4, "10年以上": 5}
                risk_score += horizon_scores.get(investment_horizon, 0)
                
                # 金融资产得分
                asset_scores = {"10以下": 1, "10-50": 2, "50-100": 3, "100-500": 4, "500以上": 5}
                risk_score += asset_scores.get(financial_assets, 0)
                
                # 问卷得分
                q1_scores = {"无经验": 1, "1年以内": 2, "1-3年": 3, "3-5年": 4, "5年以上": 5}
                risk_score += q1_scores.get(q1, 0)
                
                q2_scores = {"不能接受任何损失": 1, "5%以内": 2, "10%以内": 3, "20%以内": 4, "30%以上": 5}
                risk_score += q2_scores.get(q2, 0)
                
                q3_scores = {"立即全部卖出": 1, "卖出一部分": 2, "继续持有": 3, "买入更多": 5}
                risk_score += q3_scores.get(q3, 0)
                
                q4_scores = {"保本保息类产品": 1, "固定收益类产品": 2, "混合型产品": 3, "股票型产品": 4, "高风险高收益产品": 5}
                risk_score += q4_scores.get(q4, 0)
                
                q5_scores = {"保本保息": 1, "稳健增值": 2, "平衡增长": 3, "积极增长": 4, "追求最大收益": 5}
                risk_score += q5_scores.get(q5, 0)
                
                # 归一化得分（0-100）
                normalized_score = min(100, max(0, risk_score * 4))
                
                # 确定风险类型
                risk_type = "保守型"
                if normalized_score >= 80:
                    risk_type = "激进型"
                elif normalized_score >= 60:
                    risk_type = "进取型"
                elif normalized_score >= 40:
                    risk_type = "平衡型"
                elif normalized_score >= 20:
                    risk_type = "稳健型"
                
                # 显示评估结果
                st.markdown("### 评估结果")
                
                # 创建仪表盘
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = normalized_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "风险承受能力得分"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self.theme['primary']},
                        'steps': [
                            {'range': [0, 20], 'color': "#E8F5E9"},
                            {'range': [20, 40], 'color': "#C8E6C9"},
                            {'range': [40, 60], 'color': "#A5D6A7"},
                            {'range': [60, 80], 'color': "#81C784"},
                            {'range': [80, 100], 'color': "#66BB6A"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': normalized_score
                        }
                    }
                ))
                
                st.plotly_chart(fig)
                
                st.markdown(f"""
                <div class="card">
                    <h3>您的风险类型: {risk_type}</h3>
                    <p><strong>风险得分:</strong> {normalized_score:.1f}/100</p>
                    <p><strong>风险特征:</strong></p>
                    <ul>
                """, unsafe_allow_html=True)
                
                # 根据风险类型显示特征
                if risk_type == "保守型":
                    st.markdown("""
                    <li>追求资金安全性和流动性</li>
                    <li>不愿意承担投资风险</li>
                    <li>期望获得稳定的收益</li>
                    <li>适合投资货币市场基金、短期债券等低风险产品</li>
                    """, unsafe_allow_html=True)
                elif risk_type == "稳健型":
                    st.markdown("""
                    <li>追求资金安全性的同时获得一定收益</li>
                    <li>能够承受小幅度的投资波动</li>
                    <li>期望获得高于存款的收益</li>
                    <li>适合投资债券基金、平衡型基金等中低风险产品</li>
                    """, unsafe_allow_html=True)
                elif risk_type == "平衡型":
                    st.markdown("""
                    <li>追求资金的长期增值</li>
                    <li>能够承受一定程度的投资波动</li>
                    <li>期望获得较为平衡的风险和收益</li>
                    <li>适合投资混合型基金、蓝筹股等中等风险产品</li>
                    """, unsafe_allow_html=True)
                elif risk_type == "进取型":
                    st.markdown("""
                    <li>追求资金的较高增值</li>
                    <li>能够承受较大幅度的投资波动</li>
                    <li>期望获得较高的长期回报</li>
                    <li>适合投资股票型基金、成长股等中高风险产品</li>
                    """, unsafe_allow_html=True)
                elif risk_type == "激进型":
                    st.markdown("""
                    <li>追求资金的最大增值</li>
                    <li>能够承受较大的投资风险</li>
                    <li>期望获得显著高于市场平均水平的回报</li>
                    <li>适合投资高成长股、期权、杠杆产品等高风险产品</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # 资产配置标签页
        with tab2:
            st.subheader("个性化资产配置")
            
            # 配置参数
            st.markdown("### 配置参数")
            col1, col2 = st.columns(2)
            with col1:
                risk_preference = st.select_slider(
                    "风险偏好",
                    options=["保守型", "稳健型", "平衡型", "进取型", "激进型"]
                )
                investment_amount = st.number_input("投资金额(万元)", min_value=1, max_value=10000, value=100)
            with col2:
                investment_period = st.slider("投资期限(年)", 1, 30, 5)
                special_needs = st.multiselect(
                    "特殊需求",
                    ["流动性需求", "定期收益", "税收优化", "养老规划", "子女教育"]
                )
            
            # 生成配置按钮
            if st.button("生成资产配置方案"):
                # 根据风险偏好生成资产配置比例
                if risk_preference == "保守型":
                    allocation = {
                        "现金及等价物": 20,
                        "债券": 50,
                        "股票": 15,
                        "房地产": 10,
                        "另类投资": 5
                    }
                elif risk_preference == "稳健型":
                    allocation = {
                        "现金及等价物": 15,
                        "债券": 40,
                        "股票": 30,
                        "房地产": 10,
                        "另类投资": 5
                    }
                elif risk_preference == "平衡型":
                    allocation = {
                        "现金及等价物": 10,
                        "债券": 30,
                        "股票": 40,
                        "房地产": 15,
                        "另类投资": 5
                    }
                elif risk_preference == "进取型":
                    allocation = {
                        "现金及等价物": 5,
                        "债券": 20,
                        "股票": 55,
                        "房地产": 15,
                        "另类投资": 5
                    }
                else:  # 激进型
                    allocation = {
                        "现金及等价物": 5,
                        "债券": 10,
                        "股票": 65,
                        "房地产": 10,
                        "另类投资": 10
                    }
                
                # 根据特殊需求调整配置
                if "流动性需求" in special_needs:
                    allocation["现金及等价物"] += 10
                    allocation["股票"] -= 5
                    allocation["另类投资"] -= 5
                
                if "定期收益" in special_needs:
                    allocation["债券"] += 10
                    allocation["股票"] -= 10
                
                if "养老规划" in special_needs:
                    allocation["债券"] += 5
                    allocation["房地产"] += 5
                    allocation["股票"] -= 10
                
                # 确保所有比例之和为100%
                total = sum(allocation.values())
                allocation = {k: round(v / total * 100) for k, v in allocation.items()}
                
                # 调整以确保总和为100
                diff = 100 - sum(allocation.values())
                if diff != 0:
                    keys = list(allocation.keys())
                    allocation[keys[0]] += diff
                
                # 显示资产配置方案
                st.markdown("### 资产配置方案")
                
                # 创建饼图
                fig = px.pie(
                    values=list(allocation.values()),
                    names=list(allocation.keys()),
                    title=f"{risk_preference}投资者的资产配置建议",
                    color_discrete_sequence=self.theme['chart_colors']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig)
                
                # 显示具体配置金额
                st.markdown("### 具体配置金额")
                
                table_html = """
                <table>
                    <tr>
                        <th>资产类别</th>
                        <th>配置比例</th>
                        <th>配置金额(万元)</th>
                    </tr>
                """
                
                for asset, percentage in allocation.items():
                    amount = investment_amount * percentage / 100
                    table_html += f"""
                    <tr>
                        <td>{asset}</td>
                        <td>{percentage}%</td>
                        <td>{amount:.2f}</td>
                    </tr>
                    """
                
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)
                
                # 显示预期收益和风险
                st.markdown("### 预期收益和风险")
                
                # 模拟不同资产类别的预期年化收益率和波动率
                expected_returns = {
                    "现金及等价物": 0.02,
                    "债券": 0.04,
                    "股票": 0.08,
                    "房地产": 0.06,
                    "另类投资": 0.10
                }
                
                volatilities = {
                    "现金及等价物": 0.01,
                    "债券": 0.05,
                    "股票": 0.18,
                    "房地产": 0.12,
                    "另类投资": 0.20
                }
                
                # 计算组合预期收益率和波动率
                portfolio_return = sum(allocation[asset] / 100 * expected_returns[asset] for asset in allocation)
                portfolio_volatility = sum(allocation[asset] / 100 * volatilities[asset] for asset in allocation)
                
                # 计算投资期末预期总值
                final_value = investment_amount * (1 + portfolio_return) ** investment_period
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">预期年化收益率</div>
                        <div class="stat-value">{portfolio_return:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">预期年化波动率</div>
                        <div class="stat-value">{portfolio_volatility:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">{investment_period}年后预期总值</div>
                    <div class="stat-value">{final_value:.2f}万元</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 模拟投资增长曲线
                years = list(range(investment_period + 1))
                values = [investment_amount * (1 + portfolio_return) ** year for year in years]
                
                fig = px.line(
                    x=years,
                    y=values,
                    labels={"x": "投资年限", "y": "投资价值(万元)"},
                    title="投资价值增长预测"
                )
                fig.update_traces(line_color=self.theme['primary'])
                
                st.plotly_chart(fig)
                
                # 投资建议
                st.markdown("### 投资建议")
                
                st.markdown(f"""
                <div class="card">
                    <h4>根据您的风险偏好和投资目标，我们建议：</h4>
                    <ul>
                """, unsafe_allow_html=True)
                
                # 根据风险偏好给出建议
                if risk_preference == "保守型":
                    st.markdown("""
                    <li>将大部分资金配置在货币市场基金、国债等低风险产品</li>
                    <li>少量配置优质蓝筹股或指数基金，分散风险</li>
                    <li>考虑配置一些通胀保值型资产，如国债通胀保值债券</li>
                    <li>保持充足的流动性，以应对突发需求</li>
                    """, unsafe_allow_html=True)
                elif risk_preference == "稳健型":
                    st.markdown("""
                    <li>配置一定比例的债券基金和优质债券，获取稳定收益</li>
                    <li>适当配置大盘蓝筹股和指数基金，分享经济增长</li>
                    <li>考虑配置一些高股息股票，获取稳定现金流</li>
                    <li>少量配置REITs等房地产投资工具，分散风险</li>
                    """, unsafe_allow_html=True)
                elif risk_preference == "平衡型":
                    st.markdown("""
                    <li>均衡配置股票和债券，兼顾收益和风险</li>
                    <li>股票部分可考虑配置行业龙头和成长股</li>
                    <li>债券部分可考虑配置一些中高等级信用债，提高收益</li>
                    <li>适当配置REITs和黄金等另类资产，分散风险</li>
                    """, unsafe_allow_html=True)
                elif risk_preference == "进取型":
                    st.markdown("""
                    <li>较高比例配置股票，包括成长股和价值股</li>
                    <li>可考虑配置一些行业ETF，把握行业轮动机会</li>
                    <li>适当配置高收益债券，提高整体收益</li>
                    <li>少量配置商品期货等另类资产，增加组合多样性</li>
                    """, unsafe_allow_html=True)
                else:  # 激进型
                    st.markdown("""
                    <li>高比例配置股票，包括高成长股和主题投资</li>
                    <li>可考虑配置一些新兴市场股票，把握全球机会</li>
                    <li>适当配置杠杆产品和期权，提高潜在收益</li>
                    <li>少量配置私募股权和风险投资，寻求超额收益</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                    </ul>
                    <p><strong>风险提示：</strong>以上建议仅供参考，不构成投资建议。实际投资应根据市场情况和个人需求进行调整。投资有风险，入市需谨慎。</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 投资组合标签页
        with tab3:
            st.subheader("投资组合跟踪")
            
            # 模拟投资组合数据
            portfolio = {
                "总资产": 1000000,
                "持仓": [
                    {"name": "茅台", "symbol": "600519.SS", "shares": 10, "cost": 1800, "current": 1900, "weight": 0.19},
                    {"name": "腾讯控股", "symbol": "0700.HK", "shares": 100, "cost": 350, "current": 380, "weight": 0.38},
                    {"name": "苹果", "symbol": "AAPL", "shares": 50, "cost": 150, "current": 170, "weight": 0.085},
                    {"name": "微软", "symbol": "MSFT", "shares": 30, "cost": 280, "current": 300, "weight": 0.09},
                    {"name": "招商银行", "symbol": "600036.SS", "shares": 200, "cost": 40, "current": 38, "weight": 0.076},
                    {"name": "南方中证500ETF", "symbol": "510500.SS", "shares": 1000, "cost": 7.5, "current": 7.8, "weight": 0.078},
                    {"name": "易方达中债1-3年国开债ETF", "symbol": "511010.SS", "shares": 1000, "cost": 100, "current": 101, "weight": 0.101}
                ]
            }
            
            # 计算投资组合表现
            total_value = sum(stock["shares"] * stock["current"] for stock in portfolio["持仓"])
            total_cost = sum(stock["shares"] * stock["cost"] for stock in portfolio["持仓"])
            total_profit = total_value - total_cost
            total_return = total_profit / total_cost * 100
            
            # 显示投资组合概览
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">总资产</div>
                    <div class="stat-value">{total_value:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                profit_class = 'stock-up' if total_profit > 0 else 'stock-down'
                profit_sign = '+' if total_profit > 0 else ''
                
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">总收益</div>
                    <div class="stat-value {profit_class}">{profit_sign}{total_profit:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                return_class = 'stock-up' if total_return > 0 else 'stock-down'
                return_sign = '+' if total_return > 0 else ''
                
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">收益率</div>
                    <div class="stat-value {return_class}">{return_sign}{total_return:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 资产配置饼图
            st.subheader("资产配置")
            
            # 按资产类别分组
            asset_classes = {
                "股票": ["600519.SS", "0700.HK", "AAPL", "MSFT", "600036.SS"],
                "ETF": ["510500.SS"],
                "债券": ["511010.SS"],
                "现金": []
            }
            
            # 计算各资产类别的价值
            class_values = {}
            for class_name, symbols in asset_classes.items():
                class_values[class_name] = sum(
                    stock["shares"] * stock["current"] 
                    for stock in portfolio["持仓"] 
                    if stock["symbol"] in symbols
                )
            
            # 添加现金
            cash_value = portfolio["总资产"] - total_value
            class_values["现金"] = cash_value
            
            # 创建饼图
            fig = px.pie(
                values=list(class_values.values()),
                names=list(class_values.keys()),
                title="资产配置比例",
                color_discrete_sequence=self.theme['chart_colors']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig)
            
            # 持仓明细
            st.subheader("持仓明细")
            
            # 创建表格
            table_html = """
            <table>
                <tr>
                    <th>股票名称</th>
                    <th>代码</th>
                    <th>持仓数量</th>
                    <th>成本价</th>
                    <th>现价</th>
                    <th>市值</th>
                    <th>收益</th>
                    <th>收益率</th>
                </tr>
            """
            
            for stock in portfolio["持仓"]:
                stock_value = stock["shares"] * stock["current"]
                stock_cost = stock["shares"] * stock["cost"]
                stock_profit = stock_value - stock_cost
                stock_return = stock_profit / stock_cost * 100
                
                profit_class = 'stock-up' if stock_profit > 0 else 'stock-down'
                profit_sign = '+' if stock_profit > 0 else ''
                
                return_class = 'stock-up' if stock_return > 0 else 'stock-down'
                return_sign = '+' if stock_return > 0 else ''
                
                table_html += f"""
                <tr>
                    <td>{stock["name"]}</td>
                    <td>{stock["symbol"]}</td>
                    <td>{stock["shares"]}</td>
                    <td>{stock["cost"]:.2f}</td>
                    <td>{stock["current"]:.2f}</td>
                    <td>{stock_value:,.2f}</td>
                    <td class="{profit_class}">{profit_sign}{stock_profit:,.2f}</td>
                    <td class="{return_class}">{return_sign}{stock_return:.2f}%</td>
                </tr>
                """
            
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)
            
            # 投资组合分析
            st.subheader("投资组合分析")
            
            # 模拟投资组合分析数据
            portfolio_analysis = {
                "年化收益率": 12.5,
                "年化波动率": 15.8,
                "夏普比率": 0.65,
                "最大回撤": 18.2,
                "贝塔系数": 0.92,
                "阿尔法": 2.3
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">年化收益率</div>
                    <div class="stat-value">{portfolio_analysis["年化收益率"]:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">年化波动率</div>
                    <div class="stat-value">{portfolio_analysis["年化波动率"]:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">夏普比率</div>
                    <div class="stat-value">{portfolio_analysis["夏普比率"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">最大回撤</div>
                    <div class="stat-value">{portfolio_analysis["最大回撤"]:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">贝塔系数</div>
                    <div class="stat-value">{portfolio_analysis["贝塔系数"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">阿尔法</div>
                    <div class="stat-value">{portfolio_analysis["阿尔法"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # 投资组合优化建议
            st.subheader("投资组合优化建议")
            
            st.markdown(f"""
            <div class="card">
                <h4>基于当前市场环境和您的投资组合，我们建议：</h4>
                <ul>
                    <li>您的投资组合整体表现良好，年化收益率{portfolio_analysis["年化收益率"]:.2f}%高于市场平均水平</li>
                    <li>投资组合的波动率略高，可以考虑增加一些低相关性资产来降低整体波动</li>
                    <li>当前股票配置比例较高，可以适当增加债券配置，提高组合稳定性</li>
                    <li>科技股占比较大，可以考虑适当分散到其他行业，降低行业集中风险</li>
                    <li>可以考虑增加一些国际市场资产，进一步分散地域风险</li>
                </ul>
                <p><strong>风险提示：</strong>以上建议仅供参考，不构成投资建议。实际投资应根据市场情况和个人需求进行调整。投资有风险，入市需谨慎。</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_customer_service_page(self):
        """渲染人工客服页面"""
        st.header("智能客服服务")
        st.markdown("""
        <div class="card">
            <p>智能客服系统结合了多种AI模型，为您提供专业、高效的金融咨询服务。无论是产品咨询、操作指导还是投资建议，我们都能为您提供及时的帮助。</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建聊天界面
        st.subheader("智能客服对话")
        
        # 初始化会话状态
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "您好，我是金融智能分析平台的AI客服助手。请问有什么可以帮助您的吗？"}
            ]
        
        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 接收用户输入
        if prompt := st.chat_input("请输入您的问题"):
            # 添加用户消息到历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 生成回复
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # 调用多模型服务获取回答
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 构建上下文
                context = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]  # 不包括最新的用户消息
                ]
                
                # 获取回答
                response = loop.run_until_complete(
                    self.multi_model_service.get_answer(prompt, context)
                )
                
                # 显示回答
                answer = response.get('answer', '抱歉，我无法回答您的问题，请稍后再试。')
                message_placeholder.markdown(answer)
                
                # 添加助手消息到历史
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 显示模型来源
                st.caption(f"回答来源: {response.get('model_name', 'Unknown')}")
        
        # 常见问题
        st.subheader("常见问题")
        
        faq_items = [
            {"question": "如何开始使用股票推荐功能？", "answer": "您可以在左侧导航栏选择\"股票推荐\"功能，然后根据自己的偏好设置筛选条件，系统会为您推荐符合条件的股票。"},
            {"question": "图表分析系统支持哪些技术形态识别？", "answer": "我们的图表分析系统支持多种技术形态识别，包括头肩顶/底、双顶/双底、三角形整理、旗形、楔形等常见形态，以及支撑位/阻力位和趋势线的自动识别。"},
            {"question": "如何查看最新的市场热点？", "answer": "您可以在\"热点资讯\"页面查看最新的市场热点话题和相关新闻，我们会每日更新市场热点分析和词云图，帮助您把握市场脉搏。"},
            {"question": "智能投顾功能如何使用？", "answer": "首先在\"智能投顾\"页面完成风险评估问卷，系统会根据您的风险偏好和投资目标，为您生成个性化的资产配置方案和投资建议。"},
            {"question": "平台的数据来源是什么？", "answer": "我们的数据来源包括雅虎财经、东方财富、新浪财经等多个权威金融数据提供商，确保数据的准确性和及时性。"}
        ]
        
        for i, faq in enumerate(faq_items):
            with st.expander(faq["question"]):
                st.markdown(faq["answer"])
        
        # 联系方式
        st.subheader("联系我们")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>在线客服</h4>
                <p>工作时间: 周一至周五 9:00-18:00</p>
                <p>电话: 400-123-4567</p>
                <p>邮箱: support@financial-ai-platform.com</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>紧急联系</h4>
                <p>非工作时间紧急问题，请拨打:</p>
                <p>电话: 138-1234-5678</p>
                <p>我们会有专业人员为您提供帮助</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 反馈表单
        st.subheader("意见反馈")
        
        with st.form("feedback_form"):
            feedback_type = st.selectbox("反馈类型", ["产品建议", "功能问题", "使用咨询", "其他"])
            feedback_content = st.text_area("反馈内容")
            contact_info = st.text_input("联系方式 (选填)")
            
            submitted = st.form_submit_button("提交反馈")
            if submitted:
                st.success("感谢您的反馈！我们会认真考虑您的建议，不断改进我们的产品和服务。")


# 测试代码
if __name__ == "__main__":
    # 创建UI优化系统
    ui = UIOptimization()
    
    # 启动Streamlit应用
    ui.create_streamlit_app()
