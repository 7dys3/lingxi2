import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import datetime
import requests
from typing import List, Dict, Tuple, Any, Optional
from bs4 import BeautifulSoup
import re
import jieba
import jieba.analyse
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# 添加数据API路径
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("警告: 无法导入ApiClient，将使用模拟数据")

class NewsAndMarketReviewSystem:
    """热点资讯和今日复盘功能"""
    
    def __init__(self, api_client=None):
        """初始化热点资讯和今日复盘系统
        
        Args:
            api_client: YahooFinance API客户端
        """
        self.api_client = api_client
        self.news_data = []  # 存储新闻数据
        self.market_review = {}  # 存储市场复盘数据
        self.hot_topics = []  # 存储热点话题
        
        # 创建数据目录
        os.makedirs('data/news', exist_ok=True)
        os.makedirs('data/market_review', exist_ok=True)
        
        # 加载中文停用词
        self.stopwords = self._load_stopwords()
        
        # 初始化jieba分词
        jieba.initialize()
        
        # 添加金融领域词汇
        self._add_financial_terms()
    
    def _load_stopwords(self) -> set:
        """加载中文停用词
        
        Returns:
            停用词集合
        """
        # 常见停用词
        stopwords = set([
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
            '或', '一个', '没有', '我们', '你们', '他们', '她们', '它们',
            '有', '这个', '那个', '这些', '那些', '不', '在', '人', '我',
            '中', '上', '下', '由', '对', '到', '为', '等', '以', '于'
        ])
        
        # 尝试从文件加载更多停用词
        try:
            stopwords_file = 'data/news/stopwords.txt'
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        stopwords.add(line.strip())
        except Exception as e:
            print(f"加载停用词文件时出错: {str(e)}")
        
        return stopwords
    
    def _add_financial_terms(self):
        """添加金融领域词汇到jieba分词词典"""
        financial_terms = [
            '股票', '基金', '债券', '期货', '期权', '外汇', '证券', '指数',
            '牛市', '熊市', '涨停', '跌停', '大盘', '个股', '主力', '散户',
            '融资', '融券', '做多', '做空', '多头', '空头', '趋势', '反转',
            '支撑位', '阻力位', '成交量', '换手率', '市盈率', '市净率', '股息率',
            '分红', '配股', '增发', '减持', '回购', '解禁', '限售股', '流通股',
            '总市值', '流通市值', '业绩', '财报', '利润', '营收', '负债', '资产',
            '现金流', '毛利率', '净利率', 'ROE', 'ROA', 'EPS', 'PE', 'PB',
            '技术分析', '基本面', '量化', '波动率', '风险', '收益', '投资组合',
            '资产配置', '分散投资', '价值投资', '成长投资', '指标', '策略', '交易',
            '持仓', '仓位', '止损', '止盈', '追涨', '杀跌', '抄底', '逃顶',
            '盘整', '突破', '回调', '反弹', '见顶', '筑底', '金叉', '死叉',
            '头肩顶', '双顶', '三重顶', '楔形', '旗形', '三角形', '缺口', '跳空',
            '均线', 'MACD', 'KDJ', 'RSI', 'BOLL', 'OBV', 'ADX', 'CCI',
            '沪指', '深指', '创业板', '科创板', '上证', '深证', '纳斯达克', '道琼斯',
            '标普', '恒生', '日经', '富时', 'A股', 'H股', '美股', '港股',
            '央行', '降息', '降准', '加息', '通胀', '通缩', '货币政策', '财政政策',
            '经济增长', 'GDP', 'CPI', 'PPI', '失业率', '贸易战', '汇率', '贬值',
            '升值', '外储', '国债', '地方债', '企业债', '可转债', 'ETF', 'LOF',
            '公募基金', '私募基金', '对冲基金', '养老金', '社保基金', '保险资金', '券商',
            '银行', '信托', '资管', '理财', '存款', '贷款', '抵押', '质押',
            '杠杆', '去杠杆', '风控', '监管', '证监会', '银保监会', '交易所', '结算所'
        ]
        
        for term in financial_terms:
            jieba.add_word(term)
    
    def fetch_financial_news(self, keywords: List[str] = None, max_pages: int = 3) -> List[Dict[str, Any]]:
        """获取金融新闻
        
        Args:
            keywords: 关键词列表，用于过滤新闻
            max_pages: 最大抓取页数
            
        Returns:
            新闻列表
        """
        # 检查是否有缓存数据
        cache_file = f'data/news/financial_news_{datetime.datetime.now().strftime("%Y%m%d")}.json'
        if os.path.exists(cache_file):
            # 检查缓存是否过期（超过4小时）
            if datetime.datetime.now().timestamp() - os.path.getmtime(cache_file) < 14400:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        self.news_data = json.load(f)
                        print(f"从缓存加载新闻数据，共 {len(self.news_data)} 条")
                        return self.news_data
                except Exception as e:
                    print(f"加载缓存新闻数据时出错: {str(e)}")
        
        # 如果没有缓存或缓存过期，则抓取新闻
        news_list = []
        
        # 定义新闻源列表
        news_sources = [
            {
                'name': '东方财富网',
                'url': 'https://finance.eastmoney.com/a/cywjh_{}.html',
                'article_selector': '.articleList .title a',
                'date_selector': '.time',
                'content_selector': '.article-content'
            },
            {
                'name': '新浪财经',
                'url': 'https://finance.sina.com.cn/roll/index.d.html?cid=56592&page={}',
                'article_selector': '.list_009 li a',
                'date_selector': '.time-source',
                'content_selector': '.article p'
            }
        ]
        
        try:
            for source in news_sources:
                for page in range(1, max_pages + 1):
                    url = source['url'].format(page)
                    
                    # 模拟数据（实际实现中应该使用requests获取真实数据）
                    if True:  # 替换为实际的网络请求条件
                        # 生成模拟新闻数据
                        simulated_news = self._generate_simulated_news(source['name'], 10)
                        news_list.extend(simulated_news)
                    else:
                        # 实际的网络请求代码
                        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            articles = soup.select(source['article_selector'])
                            
                            for article in articles:
                                article_url = article['href']
                                title = article.text.strip()
                                
                                # 过滤关键词
                                if keywords and not any(keyword in title for keyword in keywords):
                                    continue
                                
                                # 获取文章内容
                                article_response = requests.get(article_url, headers={'User-Agent': 'Mozilla/5.0'})
                                if article_response.status_code == 200:
                                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                                    
                                    # 提取日期
                                    date_element = article_soup.select_one(source['date_selector'])
                                    date = date_element.text.strip() if date_element else '未知日期'
                                    
                                    # 提取内容
                                    content_elements = article_soup.select(source['content_selector'])
                                    content = '\n'.join([p.text.strip() for p in content_elements])
                                    
                                    # 添加到新闻列表
                                    news_list.append({
                                        'title': title,
                                        'url': article_url,
                                        'date': date,
                                        'source': source['name'],
                                        'content': content,
                                        'keywords': self.extract_keywords(title + ' ' + content)
                                    })
        except Exception as e:
            print(f"获取新闻时出错: {str(e)}")
        
        # 按日期排序
        news_list.sort(key=lambda x: x['date'], reverse=True)
        
        # 保存到缓存
        self.news_data = news_list
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(news_list, f, ensure_ascii=False, indent=2)
        
        print(f"获取新闻数据完成，共 {len(news_list)} 条")
        return news_list
    
    def _generate_simulated_news(self, source: str, count: int) -> List[Dict[str, Any]]:
        """生成模拟新闻数据（仅用于测试）
        
        Args:
            source: 新闻源名称
            count: 生成数量
            
        Returns:
            模拟新闻列表
        """
        news_list = []
        
        # 模拟新闻标题模板
        title_templates = [
            "{}股市{}：{}板块领涨，{}概念受关注",
            "央行{}政策出台，{}市场{}反应",
            "{}季度GDP增长{}%，经济{}预期",
            "{}公司发布{}财报，净利润同比{}{}%",
            "{}指数{}点，{}板块{}表现",
            "{}行业迎来政策利好，{}概念股{}",
            "{}会议召开，{}成为市场焦点",
            "{}数据公布，{}市场{}波动",
            "{}股{}创新高，{}引发市场关注",
            "{}报告：{}行业{}趋势明显"
        ]
        
        # 模拟内容段落模板
        content_templates = [
            "今日{}市场{}，{}指数报收于{}点，较前一交易日{}{}%。{}板块表现{}，其中{}、{}、{}等个股{}。分析人士认为，{}因素是导致市场{}的主要原因。",
            
            "据{}消息，{}近期{}，这一消息引发市场广泛关注。业内专家{}表示，此举将对{}产生{}影响，尤其是{}、{}等细分领域。投资者需关注{}可能带来的{}机会。",
            
            "{}发布的最新研报指出，{}行业正处于{}阶段，{}、{}等公司具有较强的{}优势。报告预计，未来{}年内，该行业将保持{}%的年均增长，其中{}领域增速可能达到{}%。",
            
            "近期，{}事件持续发酵，对{}市场产生{}影响。{}表示，{}是理解此次事件的关键。从长期来看，{}可能成为{}的转折点，投资者应{}应对可能的{}变化。",
            
            "{}数据显示，{}月{}指标{}预期，环比{}{}%，同比{}{}%。{}分析师指出，这一数据{}了市场对{}的担忧，短期内{}可能会{}。建议投资者关注{}的变化。"
        ]
        
        # 模拟市场方向
        directions = ["上涨", "下跌", "震荡", "反弹", "回调", "走强", "走弱", "盘整"]
        
        # 模拟板块
        sectors = ["科技", "金融", "医药", "消费", "新能源", "半导体", "人工智能", "元宇宙", "大数据", "云计算", 
                  "生物医药", "新材料", "军工", "汽车", "房地产", "基建", "农业", "煤炭", "石油", "电力"]
        
        # 模拟概念
        concepts = ["ChatGPT", "AIGC", "光伏", "储能", "氢能", "碳中和", "数字经济", "智能制造", "国产替代", 
                   "消费升级", "元宇宙", "区块链", "虚拟现实", "生物识别", "智慧城市", "卫星互联网"]
        
        # 模拟公司
        companies = ["阿里巴巴", "腾讯", "百度", "京东", "美团", "拼多多", "比亚迪", "宁德时代", "中国平安", 
                    "贵州茅台", "恒瑞医药", "药明康德", "中芯国际", "华为", "小米", "格力电器", "海尔智家"]
        
        # 模拟指数
        indices = ["上证指数", "深证成指", "创业板指", "科创50", "沪深300", "中证500", "中证1000", 
                  "道琼斯指数", "纳斯达克指数", "标普500指数", "恒生指数", "日经225指数"]
        
        # 模拟分析师和机构
        analysts = ["中金公司", "国泰君安", "华泰证券", "海通证券", "招商证券", "广发证券", "中信证券", 
                   "摩根士丹利", "高盛", "摩根大通", "瑞银", "野村证券"]
        
        # 生成模拟新闻
        for i in range(count):
            # 生成标题
            title_template = np.random.choice(title_templates)
            title = title_template.format(
                np.random.choice(["今日", "昨日", "本周", "本月", "Q1", "Q2", "Q3", "Q4", "上半年", "下半年"]),
                np.random.choice(directions),
                np.random.choice(sectors),
                np.random.choice(concepts)
            )
            
            # 生成内容
            content_paragraphs = []
            for _ in range(np.random.randint(3, 6)):
                template = np.random.choice(content_templates)
                paragraph = template.format(
                    np.random.choice(analysts),
                    np.random.choice(sectors + concepts + companies),
                    np.random.choice(["上涨", "下跌", "波动", "调整", "反弹", "突破", "回落"]),
                    np.random.randint(2000, 4000),
                    np.random.choice(["上涨", "下跌"]),
                    np.random.uniform(0.5, 5).round(2),
                    np.random.choice(sectors),
                    np.random.choice(["活跃", "低迷", "分化", "强势", "弱势"]),
                    np.random.choice(companies),
                    np.random.choice(companies),
                    np.random.choice(companies),
                    np.random.choice(["领涨", "领跌", "表现突出", "大幅波动"]),
                    np.random.choice(["政策", "资金", "情绪", "基本面", "技术面", "外盘", "内盘"]),
                    np.random.choice(directions)
                )
                content_paragraphs.append(paragraph)
            
            content = "\n\n".join(content_paragraphs)
            
            # 生成日期（最近7天内的随机日期）
            days_ago = np.random.randint(0, 7)
            date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")
            
            # 提取关键词
            keywords = self.extract_keywords(title + ' ' + content)
            
            # 添加到新闻列表
            news_list.append({
                'title': title,
                'url': f"https://example.com/news/{i}",
                'date': date,
                'source': source,
                'content': content,
                'keywords': keywords
            })
        
        return news_list
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """从文本中提取关键词
        
        Args:
            text: 文本内容
            top_n: 提取的关键词数量
            
        Returns:
            关键词列表
        """
        # 使用jieba提取关键词
        keywords = jieba.analyse.extract_tags(text, topK=top_n, withWeight=False)
        
        # 过滤停用词
        keywords = [word for word in keywords if word not in self.stopwords]
        
        return keywords
    
    def analyze_hot_topics(self, min_count: int = 2) -> List[Dict[str, Any]]:
        """分析热点话题
        
        Args:
            min_count: 最小出现次数
            
        Returns:
            热点话题列表
        """
        if not self.news_data:
            print("没有新闻数据，无法分析热点话题")
            return []
        
        # 收集所有关键词
        all_keywords = []
        for news in self.news_data:
            all_keywords.extend(news.get('keywords', []))
        
        # 统计关键词频率
        keyword_counter = Counter(all_keywords)
        
        # 过滤低频关键词
        hot_keywords = {word: count for word, count in keyword_counter.items() if count >= min_count}
        
        # 按频率排序
        sorted_keywords = sorted(hot_keywords.items(), key=lambda x: x[1], reverse=True)
        
        # 构建热点话题
        hot_topics = []
        for keyword, count in sorted_keywords:
            # 查找包含该关键词的新闻
            related_news = []
            for news in self.news_data:
                if keyword in news.get('keywords', []) or keyword in news.get('title', ''):
                    related_news.append({
                        'title': news['title'],
                        'url': news['url'],
                        'date': news['date'],
                        'source': news['source']
                    })
            
            hot_topics.append({
                'keyword': keyword,
                'count': count,
                'related_news': related_news[:5]  # 最多包含5条相关新闻
            })
        
        # 存储热点话题
        self.hot_topics = hot_topics
        
        # 保存到文件
        hot_topics_file = f'data/news/hot_topics_{datetime.datetime.now().strftime("%Y%m%d")}.json'
        with open(hot_topics_file, 'w', encoding='utf-8') as f:
            json.dump(hot_topics, f, ensure_ascii=False, indent=2)
        
        return hot_topics
    
    def generate_word_cloud(self, width: int = 800, height: int = 400) -> str:
        """生成热点词云图
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            保存的图像文件路径
        """
        if not self.news_data:
            print("没有新闻数据，无法生成词云")
            return ""
        
        # 收集所有关键词及其频率
        all_keywords = []
        for news in self.news_data:
            all_keywords.extend(news.get('keywords', []))
        
        keyword_counter = Counter(all_keywords)
        
        # 检查是否有关键词
        if not keyword_counter:
            print("没有提取到关键词，无法生成词云")
            return ""
        
        try:
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
                print("警告: 未找到中文字体，词云可能无法正确显示中文")
                font_path = None
            
            # 生成词云
            wordcloud = WordCloud(
                width=width,
                height=height,
                background_color='white',
                max_words=100,
                max_font_size=100,
                random_state=42,
                font_path=font_path
            )
            
            # 生成词云图像
            wordcloud.generate_from_frequencies(keyword_counter)
            
            # 创建图表
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # 保存图像
            os.makedirs('data/news', exist_ok=True)
            image_file = f'data/news/wordcloud_{datetime.datetime.now().strftime("%Y%m%d")}.png'
            plt.savefig(image_file)
            plt.close()
            
            return image_file
        except Exception as e:
            print(f"生成词云时出错: {str(e)}")
            return ""
    
    def fetch_market_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """获取市场数据
        
        Args:
            symbols: 股票代码列表，如果为None则使用默认列表
            
        Returns:
            市场数据字典
        """
        if symbols is None:
            # 默认股票列表
            symbols = [
                # 指数
                '^GSPC',  # 标普500
                '^DJI',   # 道琼斯工业平均指数
                '^IXIC',  # 纳斯达克综合指数
                '^HSI',   # 恒生指数
                '000001.SS',  # 上证指数
                '399001.SZ',  # 深证成指
                '399006.SZ',  # 创业板指
                
                # 行业ETF
                'XLF',    # 金融业ETF
                'XLK',    # 科技业ETF
                'XLV',    # 医疗保健ETF
                'XLE',    # 能源业ETF
                'XLI',    # 工业ETF
                'XLP',    # 必需消费品ETF
                'XLY',    # 非必需消费品ETF
                
                # 大型股票
                'AAPL',   # 苹果
                'MSFT',   # 微软
                'GOOGL',  # Alphabet
                'AMZN',   # 亚马逊
                'META',   # Meta
                'TSLA',   # 特斯拉
                'NVDA',   # 英伟达
                'JPM',    # 摩根大通
                
                # A股代表性股票
                '600519.SS',  # 贵州茅台
                '601318.SS',  # 中国平安
                '600036.SS',  # 招商银行
                '000858.SZ',  # 五粮液
                '000333.SZ',  # 美的集团
                '002594.SZ',  # 比亚迪
                '300750.SZ',  # 宁德时代
                '688981.SS'   # 中芯国际
            ]
        
        market_data = {}
        
        for symbol in symbols:
            try:
                # 检查是否有缓存数据
                cache_file = f'data/market_review/{symbol}_daily.csv'
                if os.path.exists(cache_file):
                    # 检查缓存是否过期（超过1天）
                    if datetime.datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        market_data[symbol] = df
                        print(f"从缓存加载 {symbol} 数据")
                        continue
                
                if HAS_API_CLIENT and self.api_client:
                    # 使用YahooFinance API获取数据
                    data = self.api_client.call_api('YahooFinance/get_stock_chart', 
                                                   query={'symbol': symbol, 
                                                         'interval': '1d', 
                                                         'range': '1mo'})
                    
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
                            
                            market_data[symbol] = df
                            print(f"成功获取 {symbol} 数据")
                        else:
                            print(f"获取 {symbol} 数据失败：数据结构不完整")
                    else:
                        print(f"获取 {symbol} 数据失败：API返回错误")
                else:
                    # 生成模拟数据（仅用于测试）
                    print(f"使用模拟数据代替 {symbol}")
                    dates = pd.date_range(end=datetime.datetime.now(), periods=30, freq='B')
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
                    
                    market_data[symbol] = df
            except Exception as e:
                print(f"获取 {symbol} 数据时出错: {str(e)}")
        
        return market_data
    
    def generate_market_review(self) -> Dict[str, Any]:
        """生成市场复盘
        
        Returns:
            市场复盘数据
        """
        # 获取市场数据
        market_data = self.fetch_market_data()
        
        if not market_data:
            print("没有市场数据，无法生成市场复盘")
            return {}
        
        # 今日日期
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # 计算各指数涨跌幅
        index_changes = {}
        for symbol, df in market_data.items():
            if not df.empty and len(df) >= 2:
                latest_close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2]
                change_pct = (latest_close - prev_close) / prev_close * 100
                
                index_changes[symbol] = {
                    'latest_close': latest_close,
                    'change_pct': change_pct,
                    'direction': 'up' if change_pct > 0 else 'down'
                }
        
        # 计算行业板块表现
        sector_etfs = {
            'XLF': '金融业',
            'XLK': '科技业',
            'XLV': '医疗保健',
            'XLE': '能源业',
            'XLI': '工业',
            'XLP': '必需消费品',
            'XLY': '非必需消费品'
        }
        
        sector_performance = {}
        for symbol, sector_name in sector_etfs.items():
            if symbol in index_changes:
                sector_performance[sector_name] = index_changes[symbol]
        
        # 按涨跌幅排序
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1]['change_pct'], reverse=True)
        
        # 计算大盘资金流向
        market_trend = "震荡"
        if len(index_changes) > 0:
            # 计算主要指数的平均涨跌幅
            major_indices = ['^GSPC', '^DJI', '^IXIC', '000001.SS', '399001.SZ']
            valid_indices = [idx for idx in major_indices if idx in index_changes]
            
            if valid_indices:
                avg_change = sum(index_changes[idx]['change_pct'] for idx in valid_indices) / len(valid_indices)
                
                if avg_change > 1.5:
                    market_trend = "强势上涨"
                elif avg_change > 0.5:
                    market_trend = "上涨"
                elif avg_change > -0.5:
                    market_trend = "震荡"
                elif avg_change > -1.5:
                    market_trend = "下跌"
                else:
                    market_trend = "大幅下跌"
        
        # 生成市场复盘数据
        market_review = {
            'date': today,
            'market_trend': market_trend,
            'index_changes': index_changes,
            'sector_performance': sorted_sectors,
            'top_sectors': [sector for sector, _ in sorted_sectors[:3]] if sorted_sectors else [],
            'bottom_sectors': [sector for sector, _ in sorted_sectors[-3:]] if sorted_sectors else []
        }
        
        # 存储市场复盘数据
        self.market_review = market_review
        
        # 保存到文件
        review_file = f'data/market_review/market_review_{today}.json'
        with open(review_file, 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式
            serializable_review = json.loads(
                json.dumps(market_review, default=lambda o: float(o) if isinstance(o, np.float64) else o)
            )
            json.dump(serializable_review, f, ensure_ascii=False, indent=2)
        
        return market_review
    
    def plot_market_overview(self) -> str:
        """绘制市场概览图表
        
        Returns:
            保存的图表文件路径
        """
        # 获取市场数据
        market_data = self.fetch_market_data()
        
        if not market_data:
            print("没有市场数据，无法绘制市场概览图表")
            return ""
        
        # 选择主要指数
        main_indices = [
            ('^GSPC', '标普500'),
            ('^DJI', '道琼斯'),
            ('^IXIC', '纳斯达克'),
            ('000001.SS', '上证指数'),
            ('399001.SZ', '深证成指'),
            ('399006.SZ', '创业板指')
        ]
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (symbol, name) in enumerate(main_indices):
            if symbol in market_data and i < len(axes):
                df = market_data[symbol]
                
                if not df.empty:
                    # 计算归一化价格（相对于第一天的涨跌幅）
                    normalized = df['close'] / df['close'].iloc[0] * 100 - 100
                    
                    # 绘制价格走势
                    color = 'red' if normalized.iloc[-1] > 0 else 'green'
                    axes[i].plot(df.index, normalized, color=color, linewidth=2)
                    
                    # 添加最新涨跌幅标签
                    latest_change = normalized.iloc[-1]
                    axes[i].text(0.02, 0.95, f"{latest_change:.2f}%", 
                               transform=axes[i].transAxes, 
                               color=color, 
                               fontsize=12, 
                               fontweight='bold')
                    
                    # 设置图表属性
                    axes[i].set_title(name)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.2)
                    
                    # 设置x轴标签
                    if len(df) > 10:
                        step = len(df) // 5
                        axes[i].set_xticks(df.index[::step])
                        axes[i].set_xticklabels([d.strftime('%m-%d') for d in pd.to_datetime(df.index[::step])])
                    
                    # 旋转x轴标签
                    plt.setp(axes[i].get_xticklabels(), rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('data/market_review', exist_ok=True)
        chart_file = f"data/market_review/market_overview_{datetime.datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_file)
        plt.close()
        
        return chart_file
    
    def plot_sector_performance(self) -> str:
        """绘制行业板块表现图表
        
        Returns:
            保存的图表文件路径
        """
        if not self.market_review or 'sector_performance' not in self.market_review:
            # 如果没有市场复盘数据，先生成
            self.generate_market_review()
            
            if not self.market_review or 'sector_performance' not in self.market_review:
                print("无法获取行业板块表现数据")
                return ""
        
        # 提取行业板块表现数据
        sector_performance = self.market_review['sector_performance']
        
        if not sector_performance:
            print("行业板块表现数据为空")
            return ""
        
        # 提取行业名称和涨跌幅
        sectors = [sector for sector, _ in sector_performance]
        changes = [data['change_pct'] for _, data in sector_performance]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 设置颜色
        colors = ['red' if change > 0 else 'green' for change in changes]
        
        # 绘制水平条形图
        bars = ax.barh(sectors, changes, color=colors)
        
        # 添加数据标签
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else width - 0.3
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                   va='center', ha='left' if width > 0 else 'right',
                   color='black', fontweight='bold')
        
        # 设置图表属性
        ax.set_title('行业板块表现')
        ax.set_xlabel('涨跌幅 (%)')
        ax.grid(True, axis='x', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.2)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        os.makedirs('data/market_review', exist_ok=True)
        chart_file = f"data/market_review/sector_performance_{datetime.datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(chart_file)
        plt.close()
        
        return chart_file
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """生成每日市场报告
        
        Returns:
            报告数据和图表文件路径
        """
        # 1. 获取新闻数据
        self.fetch_financial_news()
        
        # 2. 分析热点话题
        hot_topics = self.analyze_hot_topics()
        
        # 3. 生成词云
        wordcloud_file = self.generate_word_cloud()
        
        # 4. 生成市场复盘
        market_review = self.generate_market_review()
        
        # 5. 绘制市场概览图表
        market_overview_chart = self.plot_market_overview()
        
        # 6. 绘制行业板块表现图表
        sector_chart = self.plot_sector_performance()
        
        # 7. 整合报告数据
        report = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'market_review': market_review,
            'hot_topics': hot_topics[:10],  # 最多显示10个热点话题
            'latest_news': self.news_data[:15],  # 最多显示15条最新新闻
            'charts': {
                'wordcloud': wordcloud_file,
                'market_overview': market_overview_chart,
                'sector_performance': sector_chart
            }
        }
        
        # 8. 保存报告
        report_file = f'data/market_review/daily_report_{datetime.datetime.now().strftime("%Y%m%d")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式
            serializable_report = json.loads(
                json.dumps(report, default=lambda o: float(o) if isinstance(o, np.float64) else o)
            )
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def generate_html_report(self) -> str:
        """生成HTML格式的每日市场报告
        
        Returns:
            HTML文件路径
        """
        # 生成报告数据
        report = self.generate_daily_report()
        
        if not report:
            print("无法生成报告数据")
            return ""
        
        # 构建HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>每日市场报告 - {report['date']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #0066cc;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 40px;
                }}
                .chart-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .market-summary {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }}
                .market-card {{
                    width: 30%;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.05);
                }}
                .up {{
                    color: #d33;
                }}
                .down {{
                    color: #393;
                }}
                .topic-list {{
                    list-style-type: none;
                    padding: 0;
                }}
                .topic-item {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 15px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.05);
                }}
                .news-list {{
                    list-style-type: none;
                    padding: 0;
                }}
                .news-item {{
                    border-bottom: 1px solid #eee;
                    padding: 10px 0;
                }}
                .news-title {{
                    font-weight: bold;
                }}
                .news-meta {{
                    font-size: 0.8em;
                    color: #666;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    font-size: 0.8em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>每日市场报告</h1>
                <p>日期: {report['date']}</p>
            </div>
            
            <div class="section">
                <h2>市场概览</h2>
                <p>今日市场整体呈<strong>{report.get('market_review', {}).get('market_trend', '未知')}</strong>态势。</p>
                
                <div class="chart-container">
                    <img src="{report.get('charts', {}).get('market_overview', '')}" alt="市场概览图表">
                </div>
                
                <h3>主要指数表现</h3>
                <div class="market-summary">
        """
        
        # 添加主要指数表现
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
            if symbol in report.get('market_review', {}).get('index_changes', {}):
                index_data = report['market_review']['index_changes'][symbol]
                direction_class = 'up' if index_data['direction'] == 'up' else 'down'
                change_sign = '+' if index_data['direction'] == 'up' else ''
                
                html_content += f"""
                <div class="market-card">
                    <h4>{name}</h4>
                    <p>收盘: {index_data['latest_close']:.2f}</p>
                    <p>涨跌幅: <span class="{direction_class}">{change_sign}{index_data['change_pct']:.2f}%</span></p>
                </div>
                """
        
        html_content += """
                </div>
                
                <h3>行业板块表现</h3>
                <div class="chart-container">
                    <img src="{}" alt="行业板块表现图表">
                </div>
            </div>
        """.format(report.get('charts', {}).get('sector_performance', ''))
        
        # 添加热点话题
        html_content += """
            <div class="section">
                <h2>热点话题</h2>
                <div class="chart-container">
                    <img src="{}" alt="热点词云">
                </div>
                
                <ul class="topic-list">
        """.format(report.get('charts', {}).get('wordcloud', ''))
        
        for topic in report.get('hot_topics', []):
            html_content += f"""
                <li class="topic-item">
                    <h3>{topic['keyword']} (提及次数: {topic['count']})</h3>
                    <p>相关新闻:</p>
                    <ul>
            """
            
            for news in topic.get('related_news', []):
                html_content += f"""
                        <li><a href="{news['url']}" target="_blank">{news['title']}</a> - {news['source']} ({news['date']})</li>
                """
            
            html_content += """
                    </ul>
                </li>
            """
        
        html_content += """
                </ul>
            </div>
        """
        
        # 添加最新新闻
        html_content += """
            <div class="section">
                <h2>最新财经新闻</h2>
                <ul class="news-list">
        """
        
        for news in report.get('latest_news', []):
            html_content += f"""
                <li class="news-item">
                    <div class="news-title"><a href="{news['url']}" target="_blank">{news['title']}</a></div>
                    <div class="news-meta">{news['source']} | {news['date']}</div>
                </li>
            """
        
        html_content += """
                </ul>
            </div>
            
            <div class="footer">
                <p>本报告由金融智能分析平台自动生成</p>
                <p>© 2025 金融智能分析平台</p>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML文件
        os.makedirs('data/market_review', exist_ok=True)
        html_file = f'data/market_review/daily_report_{datetime.datetime.now().strftime("%Y%m%d")}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file


# 测试代码
if __name__ == "__main__":
    # 初始化API客户端（如果可用）
    api_client = None
    if HAS_API_CLIENT:
        api_client = ApiClient()
    
    # 创建热点资讯和今日复盘系统
    news_system = NewsAndMarketReviewSystem(api_client)
    
    # 获取新闻数据
    news_system.fetch_financial_news()
    
    # 分析热点话题
    hot_topics = news_system.analyze_hot_topics()
    print(f"\n热点话题 (共 {len(hot_topics)} 个):")
    for i, topic in enumerate(hot_topics[:5]):
        print(f"{i+1}. {topic['keyword']} (提及次数: {topic['count']})")
        print("   相关新闻:")
        for news in topic['related_news'][:2]:
            print(f"   - {news['title']}")
    
    # 生成词云
    wordcloud_file = news_system.generate_word_cloud()
    print(f"\n词云已保存到: {wordcloud_file}")
    
    # 生成市场复盘
    market_review = news_system.generate_market_review()
    print(f"\n市场复盘:")
    print(f"日期: {market_review.get('date')}")
    print(f"市场趋势: {market_review.get('market_trend')}")
    print("主要指数表现:")
    for symbol, data in list(market_review.get('index_changes', {}).items())[:5]:
        print(f"  {symbol}: {data['change_pct']:.2f}%")
    
    # 绘制市场概览图表
    market_chart = news_system.plot_market_overview()
    print(f"\n市场概览图表已保存到: {market_chart}")
    
    # 绘制行业板块表现图表
    sector_chart = news_system.plot_sector_performance()
    print(f"\n行业板块表现图表已保存到: {sector_chart}")
    
    # 生成HTML报告
    html_report = news_system.generate_html_report()
    print(f"\nHTML报告已保存到: {html_report}")
