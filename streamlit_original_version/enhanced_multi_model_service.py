import os
import sys
import pandas as pd
import numpy as np
import json
import datetime
from typing import List, Dict, Tuple, Any, Optional
import asyncio
import random

# 添加数据API路径
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    HAS_API_CLIENT = True
except ImportError:
    HAS_API_CLIENT = False
    print("警告: 无法导入ApiClient，将使用模拟数据")

class EnhancedMultiModelService:
    """增强的多模型服务"""
    
    def __init__(self):
        """初始化多模型服务"""
        # 模型配置
        self.models = {
            'deepseek': {
                'name': 'DeepSeek',
                'weight': 0.25,
                'api_key': os.environ.get('DEEPSEEK_API_KEY', ''),
                'enabled': True,
                'strengths': ['金融分析', '市场预测', '技术分析']
            },
            'zhipu': {
                'name': '智谱AI',
                'weight': 0.20,
                'api_key': os.environ.get('ZHIPU_API_KEY', ''),
                'enabled': True,
                'strengths': ['中文处理', '政策解读', '行业研究']
            },
            'baidu': {
                'name': '文心一言',
                'weight': 0.20,
                'api_key': os.environ.get('BAIDU_API_KEY', ''),
                'enabled': True,
                'strengths': ['数据分析', '图表解读', '财报分析']
            },
            'xunfei': {
                'name': '讯飞星火',
                'weight': 0.15,
                'api_key': os.environ.get('XUNFEI_API_KEY', ''),
                'enabled': True,
                'strengths': ['语音交互', '实时翻译', '自然语言理解']
            },
            'alibaba': {
                'name': '通义千问',
                'weight': 0.20,
                'api_key': os.environ.get('ALIBABA_API_KEY', ''),
                'enabled': True,
                'strengths': ['电商数据', '消费趋势', '供应链分析']
            }
        }
        
        # 创建数据目录
        os.makedirs('data/multi_model', exist_ok=True)
        
        # 加载金融术语库
        self.financial_terms = self._load_financial_terms()
        
        # 加载合规规则
        self.compliance_rules = self._load_compliance_rules()
        
        # 历史查询记录
        self.query_history = []
        
        # 模型性能统计
        self.model_stats = {model_id: {'queries': 0, 'selected': 0, 'avg_response_time': 0} 
                           for model_id in self.models.keys()}
    
    def _load_financial_terms(self) -> Dict[str, str]:
        """加载金融术语库
        
        Returns:
            金融术语字典，键为术语，值为解释
        """
        # 示例金融术语
        terms = {
            "股票": "股份公司发行的所有权凭证，是股份公司为筹集资金而发行给各个股东作为持股凭证并借以取得股息和红利的一种有价证券",
            "债券": "债券是一种有价证券，是社会各类经济主体为筹集资金而向债券投资者出具的、承诺按一定利率支付利息并按约定条件偿还本金的债权债务凭证",
            "基金": "基金是指通过发行基金份额，将众多投资者的资金集中起来，形成独立财产，由基金托管人托管，基金管理人管理，以投资组合的方式进行证券投资的一种利益共享、风险共担的集合投资方式",
            "期货": "期货是一种在期货交易所交易的标准化合约，它规定了在未来某一特定时间和地点交割一定数量标的物的交易",
            "期权": "期权是一种合约，赋予持有人在某一特定日期或该日之前的任何时间以固定价格购买或出售一种资产的权利",
            "ETF": "交易型开放式指数基金，通常又被称为交易所交易基金(Exchange Traded Funds)，是一种在交易所上市交易的、基金份额可变的一种开放式基金",
            "LOF": "上市型开放式基金(Listed Open-ended Fund)，是一种既可以在场外市场进行基金份额申购、赎回，又可以在交易所（场内市场）进行基金份额交易的开放式基金",
            "QDII": "合格境内机构投资者(Qualified Domestic Institutional Investor)，是指在人民币资本项目下不可兑换、资本市场未开放条件下，在一国境内设立，经该国有关部门批准从事境外证券市场的股票、债券等有价证券业务的证券投资基金管理公司、保险公司、银行等金融机构",
            "QFII": "合格境外机构投资者(Qualified Foreign Institutional Investor)，是指符合条件的境外机构投资者，经中国证监会批准投资于中国证券市场的境外法人",
            "IPO": "首次公开募股(Initial Public Offerings)，是指一家企业或公司第一次将它的股份向公众出售",
            "PE": "市盈率(Price Earnings Ratio)，是指股票价格除以每股收益的比率",
            "PB": "市净率(Price Book Ratio)，是指股票价格除以每股净资产的比率",
            "ROE": "净资产收益率(Return On Equity)，是指净利润除以平均股东权益的比率，反映股东权益的收益水平",
            "MACD": "指数平滑异同移动平均线(Moving Average Convergence Divergence)，是一种趋势跟踪的股票分析工具",
            "KDJ": "随机指标(Random Index)，是一种超买超卖指标，在计算中考虑了最高价、最低价和收盘价，并与价格与成交量指标相比，KDJ指标更具有先行性",
            "RSI": "相对强弱指标(Relative Strength Index)，是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力",
            "BOLL": "布林线指标(Bollinger Bands)，是根据统计学中的标准差原理设计出来的一种非常实用的技术分析指标",
            "牛市": "行情普遍看涨的市场，指整个股票市场处于大幅上涨的状态",
            "熊市": "行情普遍看跌的市场，指整个股票市场处于大幅下跌的状态",
            "多头": "预期市场价格将要上涨的投资者",
            "空头": "预期市场价格将要下跌的投资者",
            "做多": "买入证券或期货合约，期望价格上涨获利的交易行为",
            "做空": "卖出证券或期货合约，期望价格下跌获利的交易行为",
            "止损": "当投资出现一定程度的亏损时，主动斩仓出局的操作",
            "止盈": "当投资出现一定程度的盈利时，主动斩仓出局的操作",
            "趋势线": "连接两个或两个以上的峰或谷的直线，用来确定价格的运行趋势",
            "支撑位": "股价下跌时的支撑水平，是指股价下跌到某个价位附近时会停止下跌的现象",
            "阻力位": "股价上涨时的阻力水平，是指股价上涨到某个价位附近时会停止上涨的现象",
            "头肩顶": "技术分析中的一种反转形态，由三个高点组成，中间的高点(头部)高于两侧的高点(肩部)，预示着上升趋势即将结束",
            "头肩底": "技术分析中的一种反转形态，由三个低点组成，中间的低点(头部)低于两侧的低点(肩部)，预示着下降趋势即将结束",
            "双顶": "技术分析中的一种顶部反转形态，由两个相近的高点组成，预示着上升趋势即将结束",
            "双底": "技术分析中的一种底部反转形态，由两个相近的低点组成，预示着下降趋势即将结束",
            "三角形整理": "技术分析中的一种整理形态，价格波动幅度逐渐缩小，形成三角形，包括对称三角形、上升三角形和下降三角形",
            "旗形": "技术分析中的一种整理形态，由一个小的平行四边形构成，通常出现在强劲的上升或下降趋势中",
            "楔形": "技术分析中的一种整理形态，由两条收敛的趋势线构成，可能是上升楔形或下降楔形",
            "缺口": "技术分析中的一种现象，指两个交易日的K线之间出现没有成交的价格区域",
            "量价背离": "成交量与价格变动方向不一致的现象，可能预示着价格趋势即将改变",
            "金叉": "技术指标中的一种买入信号，指短期均线从下向上穿越长期均线",
            "死叉": "技术指标中的一种卖出信号，指短期均线从上向下穿越长期均线",
            "超买": "技术指标显示市场过度买入的状态，可能预示着价格即将回落",
            "超卖": "技术指标显示市场过度卖出的状态，可能预示着价格即将反弹"
        }
        
        return terms
    
    def _load_compliance_rules(self) -> List[Dict[str, Any]]:
        """加载合规规则
        
        Returns:
            合规规则列表
        """
        # 示例合规规则
        rules = [
            {
                'id': 'no_investment_advice',
                'description': '不得提供具体投资建议',
                'keywords': ['建议买入', '建议卖出', '应该购买', '应该出售', '必涨', '必跌', '稳赚', '包赚', '绝对收益'],
                'replacement': '根据分析，该股票/市场可能会[上涨/下跌]，但具体投资决策请结合自身风险承受能力和投资目标，并咨询专业投资顾问。'
            },
            {
                'id': 'no_guaranteed_returns',
                'description': '不得承诺确定收益',
                'keywords': ['保证收益', '确保盈利', '稳赚不赔', '零风险', '无风险', '必定盈利'],
                'replacement': '投资有风险，任何投资策略都无法保证收益，请谨慎决策。'
            },
            {
                'id': 'risk_disclosure',
                'description': '必须披露风险',
                'check_function': lambda text: '风险' in text or '投资有风险' in text,
                'append_text': '投资有风险，入市需谨慎。过往业绩不代表未来表现。'
            },
            {
                'id': 'no_market_manipulation',
                'description': '不得操纵市场',
                'keywords': ['抱团', '一起买入', '联合购买', '共同拉升', '割韭菜'],
                'replacement': '市场参与者应当独立做出投资决策，任何形式的市场操纵都是违法的。'
            }
        ]
        
        return rules
    
    async def query_model(self, model_id: str, query: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """查询单个模型
        
        Args:
            model_id: 模型ID
            query: 查询内容
            context: 上下文对话历史
            
        Returns:
            模型响应
        """
        if model_id not in self.models:
            return {
                'model_id': model_id,
                'success': False,
                'error': f"未知模型: {model_id}",
                'response': None,
                'response_time': 0
            }
        
        model = self.models[model_id]
        
        if not model['enabled']:
            return {
                'model_id': model_id,
                'success': False,
                'error': f"模型已禁用: {model_id}",
                'response': None,
                'response_time': 0
            }
        
        # 记录开始时间
        start_time = datetime.datetime.now()
        
        try:
            # 实际实现中应该调用相应的API
            # 这里使用模拟响应
            await asyncio.sleep(random.uniform(0.5, 2.0))  # 模拟API调用延迟
            
            # 生成模拟响应
            response = self._generate_simulated_response(model_id, query, context)
            
            # 记录结束时间
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # 更新模型统计
            self.model_stats[model_id]['queries'] += 1
            self.model_stats[model_id]['avg_response_time'] = (
                (self.model_stats[model_id]['avg_response_time'] * (self.model_stats[model_id]['queries'] - 1) + response_time) / 
                self.model_stats[model_id]['queries']
            )
            
            return {
                'model_id': model_id,
                'success': True,
                'response': response,
                'response_time': response_time
            }
        except Exception as e:
            # 记录结束时间
            end_time = datetime.datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            return {
                'model_id': model_id,
                'success': False,
                'error': str(e),
                'response': None,
                'response_time': response_time
            }
    
    def _generate_simulated_response(self, model_id: str, query: str, context: List[Dict[str, str]] = None) -> str:
        """生成模拟响应（仅用于测试）
        
        Args:
            model_id: 模型ID
            query: 查询内容
            context: 上下文对话历史
            
        Returns:
            模拟响应
        """
        model = self.models[model_id]
        model_name = model['name']
        
        # 检查是否是金融术语解释
        for term, explanation in self.financial_terms.items():
            if term in query and ("什么是" in query or "解释" in query or "含义" in query):
                return f"{term}是指{explanation}。这是{model_name}的解释，希望对您有所帮助。"
        
        # 检查是否是市场分析
        if "市场" in query and ("分析" in query or "走势" in query or "预测" in query):
            return f"""根据最近的市场数据分析，当前市场呈现{random.choice(['震荡', '上涨', '下跌', '盘整'])}态势。
主要指数表现{random.choice(['良好', '一般', '较差'])}，其中科技、医药和消费板块{random.choice(['表现活跃', '领涨', '领跌', '分化明显'])}。
从技术面看，市场{random.choice(['超买', '超卖', '处于中性'])}，建议投资者{random.choice(['保持谨慎', '关注机会', '控制仓位'])}。
需要注意的是，投资有风险，以上分析仅供参考，不构成投资建议。

这是{model_name}的市场分析，希望对您有所帮助。"""
        
        # 检查是否是股票分析
        stock_pattern = r'[0-9]{6}|[A-Z]{1,5}'
        if ("股票" in query or "个股" in query) and ("分析" in query or "评估" in query):
            return f"""对于您提到的股票，基于最新的财务数据和技术指标分析：

基本面分析：
- 公司最近一季度营收同比{random.choice(['增长', '下降'])}{random.randint(5, 30)}%
- 净利润同比{random.choice(['增长', '下降'])}{random.randint(3, 35)}%
- 市盈率(PE)为{random.randint(10, 40)}，{random.choice(['高于', '低于'])}行业平均水平
- 市净率(PB)为{random.uniform(1, 5):.2f}

技术面分析：
- 股价处于{random.choice(['上升', '下降', '横盘整理'])}通道
- MACD指标显示{random.choice(['金叉', '死叉', '多头排列', '空头排列'])}
- RSI指标为{random.randint(30, 70)}，处于{random.choice(['超买', '超卖', '中性'])}区域
- 成交量{random.choice(['放大', '萎缩', '平稳'])}

风险提示：投资有风险，以上分析仅供参考，不构成投资建议。

这是{model_name}的分析，希望对您有所帮助。"""
        
        # 检查是否是政策解读
        if "政策" in query and ("解读" in query or "分析" in query or "影响" in query):
            return f"""关于您提到的政策，{model_name}的解读如下：

政策要点：
1. 该政策主要针对{random.choice(['金融市场', '实体经济', '科技创新', '消费领域', '房地产市场'])}
2. 核心措施包括{random.choice(['减税降费', '提供补贴', '放宽准入', '加强监管', '优化流程'])}
3. 实施时间预计从{random.choice(['即日起', '下个月', '下个季度', '明年初'])}开始

可能的影响：
- 对市场的影响：可能导致相关板块{random.choice(['短期波动', '中长期向好', '结构性机会'])}
- 对企业的影响：有望{random.choice(['降低成本', '提升效率', '扩大市场', '增加负担'])}
- 对投资者的影响：建议关注{random.choice(['政策受益板块', '龙头企业', '创新型企业'])}

需要注意的是，政策执行还需观察实际落地情况，市场反应可能存在滞后性。

这是{model_name}的政策解读，希望对您有所帮助。"""
        
        # 检查是否是财报分析
        if "财报" in query and ("分析" in query or "解读" in query):
            return f"""关于您提到的财报分析，{model_name}的解读如下：

财务表现：
- 营业收入：{random.randint(10, 100)}亿元，同比{random.choice(['增长', '下降'])}{random.randint(5, 30)}%
- 净利润：{random.randint(1, 20)}亿元，同比{random.choice(['增长', '下降'])}{random.randint(3, 35)}%
- 毛利率：{random.randint(20, 60)}%，{random.choice(['提升', '下降'])}{random.randint(1, 5)}个百分点
- 经营现金流：{random.randint(5, 50)}亿元，{random.choice(['同比增长', '同比下降', '保持稳定'])}

业务亮点：
1. {random.choice(['核心业务稳健增长', '新业务快速发展', '海外市场表现亮眼', '研发投入大幅增加'])}
2. {random.choice(['产品结构持续优化', '客户基础不断扩大', '成本控制效果显著', '渠道建设成效明显'])}

风险因素：
1. {random.choice(['行业竞争加剧', '原材料价格上涨', '政策环境变化', '技术迭代加速'])}
2. {random.choice(['应收账款增加', '存货周转率下降', '负债率上升', '汇率波动影响'])}

投资建议：基于财报表现，该公司{random.choice(['具有投资价值', '值得关注', '需要谨慎对待'])}，但具体投资决策请结合自身风险偏好和市场环境。

这是{model_name}的财报分析，希望对您有所帮助。"""
        
        # 默认响应
        return f"""您好，我是{model_name}。您的问题是关于"{query}"。

根据我的分析，这个问题涉及到{random.choice(['金融市场', '投资策略', '风险管理', '资产配置', '技术分析'])}的内容。

在回答这个问题前，我想强调投资决策应该基于全面的研究和个人的风险承受能力。没有任何投资策略可以保证绝对收益，市场存在各种不确定性因素。

基于当前可获得的信息，我的分析如下：
1. {random.choice(['市场环境正在发生变化', '技术指标显示潜在机会', '基本面因素需要关注', '风险因素不容忽视'])}
2. {random.choice(['短期内可能出现波动', '中长期趋势值得关注', '行业轮动可能带来机会', '宏观政策将产生重要影响'])}
3. {random.choice(['投资者应当保持理性', '分散投资有助于控制风险', '定期评估投资组合很重要', '关注市场变化及时调整策略'])}

希望我的回答对您有所帮助。如果您有更具体的问题，欢迎继续咨询。

投资有风险，入市需谨慎。"""
    
    async def query_all_models(self, query: str, context: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """查询所有启用的模型
        
        Args:
            query: 查询内容
            context: 上下文对话历史
            
        Returns:
            所有模型的响应列表
        """
        # 获取所有启用的模型
        enabled_models = [model_id for model_id, model in self.models.items() if model['enabled']]
        
        # 创建查询任务
        tasks = [self.query_model(model_id, query, context) for model_id in enabled_models]
        
        # 并行执行所有任务
        responses = await asyncio.gather(*tasks)
        
        return responses
    
    def evaluate_responses(self, query: str, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """评估各模型响应的质量
        
        Args:
            query: 原始查询
            responses: 模型响应列表
            
        Returns:
            模型评分字典
        """
        scores = {}
        
        for response in responses:
            if not response['success']:
                scores[response['model_id']] = 0
                continue
                
            model_id = response['model_id']
            answer = response['response']
            
            # 基础分数
            base_score = 0.5
            
            # 响应时间评分（越快越好，但有最低阈值）
            time_score = min(1.0, 2.0 / max(0.5, response['response_time']))
            
            # 响应长度评分（适中为佳）
            length = len(answer)
            if length < 50:
                length_score = length / 50  # 过短的回答得分较低
            elif length > 1000:
                length_score = 2 - (length / 1000)  # 过长的回答得分较低
                length_score = max(0.3, min(1.0, length_score))  # 限制在0.3-1.0之间
            else:
                length_score = 1.0  # 长度适中
            
            # 相关性评分（简单实现，实际应使用更复杂的语义相似度）
            query_keywords = set(query.lower().split())
            answer_keywords = set(answer.lower().split())
            common_keywords = query_keywords.intersection(answer_keywords)
            relevance_score = len(common_keywords) / max(1, len(query_keywords))
            
            # 专业术语评分
            term_count = sum(1 for term in self.financial_terms if term in answer)
            term_score = min(1.0, term_count / 5)  # 最多计算5个术语
            
            # 合规性评分
            compliance_score = self._check_compliance(answer)
            
            # 模型权重
            model_weight = self.models[model_id]['weight']
            
            # 计算总分
            total_score = (
                0.1 * base_score +
                0.1 * time_score +
                0.2 * length_score +
                0.3 * relevance_score +
                0.2 * term_score +
                0.1 * compliance_score
            ) * model_weight
            
            scores[model_id] = total_score
        
        return scores
    
    def _check_compliance(self, text: str) -> float:
        """检查文本是否符合合规规则
        
        Args:
            text: 待检查文本
            
        Returns:
            合规性得分（0-1之间）
        """
        violations = 0
        
        for rule in self.compliance_rules:
            if 'keywords' in rule:
                for keyword in rule['keywords']:
                    if keyword in text:
                        violations += 1
                        break
            
            if 'check_function' in rule and callable(rule['check_function']):
                if not rule['check_function'](text):
                    violations += 1
        
        # 计算合规性得分
        compliance_score = max(0, 1 - (violations / len(self.compliance_rules)))
        
        return compliance_score
    
    def apply_compliance_rules(self, text: str) -> str:
        """应用合规规则修正文本
        
        Args:
            text: 原始文本
            
        Returns:
            修正后的文本
        """
        modified_text = text
        
        for rule in self.compliance_rules:
            if 'keywords' in rule and 'replacement' in rule:
                for keyword in rule['keywords']:
                    if keyword in modified_text:
                        # 替换包含关键词的整句话
                        sentences = re.split(r'(?<=[。！？.!?])\s*', modified_text)
                        for i, sentence in enumerate(sentences):
                            if keyword in sentence:
                                sentences[i] = rule['replacement']
                        modified_text = ' '.join(sentences)
            
            if 'append_text' in rule and 'check_function' in rule and callable(rule['check_function']):
                if not rule['check_function'](modified_text):
                    modified_text += f"\n\n{rule['append_text']}"
        
        return modified_text
    
    def select_best_response(self, responses: List[Dict[str, Any]], scores: Dict[str, float]) -> Dict[str, Any]:
        """选择最佳响应
        
        Args:
            responses: 模型响应列表
            scores: 模型评分字典
            
        Returns:
            最佳响应
        """
        if not scores:
            # 如果没有评分，返回第一个成功的响应
            for response in responses:
                if response['success']:
                    return response
            
            # 如果没有成功的响应，返回错误信息
            return {
                'model_id': 'fallback',
                'success': False,
                'error': '所有模型均未返回有效响应',
                'response': '抱歉，当前无法回答您的问题，请稍后再试。',
                'response_time': 0
            }
        
        # 获取得分最高的模型
        best_model_id = max(scores, key=scores.get)
        best_score = scores[best_model_id]
        
        # 更新模型统计
        self.model_stats[best_model_id]['selected'] += 1
        
        # 查找对应的响应
        for response in responses:
            if response['model_id'] == best_model_id and response['success']:
                # 应用合规规则
                response['response'] = self.apply_compliance_rules(response['response'])
                response['score'] = best_score
                return response
        
        # 如果最佳模型的响应不可用，选择得分第二高的
        sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for model_id, score in sorted_models:
            for response in responses:
                if response['model_id'] == model_id and response['success']:
                    # 应用合规规则
                    response['response'] = self.apply_compliance_rules(response['response'])
                    response['score'] = score
                    return response
        
        # 如果所有模型都失败，返回错误信息
        return {
            'model_id': 'fallback',
            'success': False,
            'error': '所有模型均未返回有效响应',
            'response': '抱歉，当前无法回答您的问题，请稍后再试。',
            'response_time': 0
        }
    
    async def get_answer(self, query: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """获取问题的回答
        
        Args:
            query: 查询内容
            context: 上下文对话历史
            
        Returns:
            回答结果
        """
        # 记录查询
        query_record = {
            'query': query,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': context
        }
        self.query_history.append(query_record)
        
        # 1. 查询所有模型
        responses = await self.query_all_models(query, context)
        
        # 2. 评估响应质量
        scores = self.evaluate_responses(query, responses)
        
        # 3. 选择最佳响应
        best_response = self.select_best_response(responses, scores)
        
        # 4. 构建结果
        result = {
            'query': query,
            'answer': best_response.get('response', '抱歉，无法回答您的问题'),
            'source_model': best_response.get('model_id', 'unknown'),
            'model_name': self.models.get(best_response.get('model_id'), {}).get('name', 'Unknown'),
            'confidence': best_response.get('score', 0),
            'response_time': best_response.get('response_time', 0),
            'timestamp': datetime.datetime.now().isoformat(),
            'all_responses': [{
                'model_id': r['model_id'],
                'model_name': self.models.get(r['model_id'], {}).get('name', 'Unknown'),
                'response': r.get('response', ''),
                'score': scores.get(r['model_id'], 0),
                'response_time': r.get('response_time', 0)
            } for r in responses if r['success']]
        }
        
        # 5. 保存查询记录
        query_record.update({
            'result': result
        })
        
        # 定期保存查询历史
        if len(self.query_history) % 10 == 0:
            self._save_query_history()
        
        return result
    
    def _save_query_history(self):
        """保存查询历史"""
        history_file = f'data/multi_model/query_history_{datetime.datetime.now().strftime("%Y%m%d")}.json'
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.query_history[-100:], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存查询历史时出错: {str(e)}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息
        
        Returns:
            模型统计信息
        """
        stats = {}
        
        for model_id, model_info in self.models.items():
            model_stat = self.model_stats.get(model_id, {})
            queries = model_stat.get('queries', 0)
            selected = model_stat.get('selected', 0)
            
            stats[model_id] = {
                'name': model_info['name'],
                'queries': queries,
                'selected': selected,
                'selection_rate': selected / max(1, queries),
                'avg_response_time': model_stat.get('avg_response_time', 0)
            }
        
        return stats
    
    def update_model_weights(self, auto_adjust: bool = True) -> Dict[str, float]:
        """更新模型权重
        
        Args:
            auto_adjust: 是否自动调整权重
            
        Returns:
            更新后的模型权重
        """
        if not auto_adjust:
            return {model_id: model['weight'] for model_id, model in self.models.items()}
        
        # 获取模型统计
        stats = self.get_model_stats()
        
        # 计算新权重
        total_selected = sum(stat['selected'] for stat in stats.values())
        
        if total_selected == 0:
            return {model_id: model['weight'] for model_id, model in self.models.items()}
        
        # 基于选择率和响应时间计算新权重
        new_weights = {}
        for model_id, stat in stats.items():
            # 选择率权重
            selection_weight = stat['selected'] / max(1, total_selected)
            
            # 响应时间权重（响应越快权重越高）
            time_weight = 1.0
            if stat['avg_response_time'] > 0:
                time_weight = min(1.0, 1.0 / stat['avg_response_time'])
            
            # 综合权重
            combined_weight = 0.7 * selection_weight + 0.3 * time_weight
            
            # 确保权重不会太低
            new_weights[model_id] = max(0.05, combined_weight)
        
        # 归一化权重
        total_weight = sum(new_weights.values())
        normalized_weights = {model_id: weight / total_weight for model_id, weight in new_weights.items()}
        
        # 更新模型权重
        for model_id, weight in normalized_weights.items():
            self.models[model_id]['weight'] = weight
        
        return normalized_weights
    
    def explain_financial_term(self, term: str) -> Dict[str, Any]:
        """解释金融术语
        
        Args:
            term: 金融术语
            
        Returns:
            解释结果
        """
        if term in self.financial_terms:
            explanation = self.financial_terms[term]
            return {
                'term': term,
                'explanation': explanation,
                'found': True
            }
        
        # 查找相似术语
        similar_terms = []
        for known_term in self.financial_terms.keys():
            if term in known_term or known_term in term:
                similar_terms.append(known_term)
        
        return {
            'term': term,
            'explanation': None,
            'found': False,
            'similar_terms': similar_terms
        }
    
    def add_financial_term(self, term: str, explanation: str) -> bool:
        """添加金融术语
        
        Args:
            term: 金融术语
            explanation: 解释
            
        Returns:
            是否添加成功
        """
        if not term or not explanation:
            return False
        
        self.financial_terms[term] = explanation
        return True
    
    def get_model_recommendations(self, query: str) -> List[Dict[str, Any]]:
        """获取模型推荐
        
        Args:
            query: 查询内容
            
        Returns:
            推荐的模型列表
        """
        recommendations = []
        
        # 分析查询类型
        query_type = self._analyze_query_type(query)
        
        # 根据查询类型推荐模型
        for model_id, model in self.models.items():
            if not model['enabled']:
                continue
                
            # 检查模型是否适合该查询类型
            relevance = self._check_model_relevance(model_id, query_type)
            
            if relevance > 0:
                recommendations.append({
                    'model_id': model_id,
                    'model_name': model['name'],
                    'relevance': relevance,
                    'strengths': model['strengths']
                })
        
        # 按相关性排序
        recommendations.sort(key=lambda x: x['relevance'], reverse=True)
        
        return recommendations
    
    def _analyze_query_type(self, query: str) -> str:
        """分析查询类型
        
        Args:
            query: 查询内容
            
        Returns:
            查询类型
        """
        query = query.lower()
        
        if any(keyword in query for keyword in ['什么是', '解释', '含义', '定义']):
            return 'term_explanation'
        
        if any(keyword in query for keyword in ['市场', '大盘', '指数', '走势']):
            return 'market_analysis'
        
        if any(keyword in query for keyword in ['股票', '个股', '代码']):
            return 'stock_analysis'
        
        if any(keyword in query for keyword in ['政策', '规定', '法规', '措施']):
            return 'policy_analysis'
        
        if any(keyword in query for keyword in ['财报', '业绩', '营收', '利润']):
            return 'financial_report'
        
        if any(keyword in query for keyword in ['技术', '指标', '形态', '趋势']):
            return 'technical_analysis'
        
        if any(keyword in query for keyword in ['投资', '策略', '组合', '配置']):
            return 'investment_strategy'
        
        return 'general'
    
    def _check_model_relevance(self, model_id: str, query_type: str) -> float:
        """检查模型与查询类型的相关性
        
        Args:
            model_id: 模型ID
            query_type: 查询类型
            
        Returns:
            相关性得分（0-1之间）
        """
        model = self.models[model_id]
        strengths = model.get('strengths', [])
        
        # 查询类型与模型优势的映射
        type_strength_map = {
            'term_explanation': ['中文处理', '金融分析', '数据分析'],
            'market_analysis': ['金融分析', '市场预测', '数据分析'],
            'stock_analysis': ['金融分析', '技术分析', '数据分析'],
            'policy_analysis': ['政策解读', '中文处理', '行业研究'],
            'financial_report': ['财报分析', '数据分析', '金融分析'],
            'technical_analysis': ['技术分析', '图表解读', '市场预测'],
            'investment_strategy': ['金融分析', '市场预测', '行业研究'],
            'general': []  # 通用查询不特别偏向任何模型
        }
        
        # 计算相关性
        relevant_strengths = type_strength_map.get(query_type, [])
        if not relevant_strengths:
            return model['weight']  # 通用查询使用模型权重
        
        # 计算模型优势与查询类型相关优势的交集
        common_strengths = set(strengths).intersection(set(relevant_strengths))
        relevance = len(common_strengths) / max(1, len(relevant_strengths))
        
        # 结合模型权重
        return relevance * model['weight'] + 0.1  # 添加基础分
    
    async def run_demo(self, queries: List[str] = None) -> List[Dict[str, Any]]:
        """运行演示
        
        Args:
            queries: 查询列表，如果为None则使用默认查询
            
        Returns:
            回答结果列表
        """
        if queries is None:
            queries = [
                "什么是股票市盈率？",
                "分析一下最近的市场走势",
                "解释一下MACD指标的含义和使用方法",
                "最近的货币政策对股市有什么影响？",
                "如何分析一家公司的财务报表？"
            ]
        
        results = []
        
        for query in queries:
            print(f"\n问题: {query}")
            result = await self.get_answer(query)
            
            print(f"回答 (来自 {result['model_name']}): {result['answer'][:100]}...")
            print(f"置信度: {result['confidence']:.2f}, 响应时间: {result['response_time']:.2f}秒")
            
            results.append(result)
        
        # 打印模型统计
        stats = self.get_model_stats()
        print("\n模型统计:")
        for model_id, stat in stats.items():
            print(f"{stat['name']}: 查询次数 {stat['queries']}, 被选择次数 {stat['selected']}, " +
                 f"选择率 {stat['selection_rate']:.2%}, 平均响应时间 {stat['avg_response_time']:.2f}秒")
        
        # 更新模型权重
        new_weights = self.update_model_weights()
        print("\n更新后的模型权重:")
        for model_id, weight in new_weights.items():
            print(f"{self.models[model_id]['name']}: {weight:.2f}")
        
        return results


# 测试代码
if __name__ == "__main__":
    # 创建多模型服务
    service = EnhancedMultiModelService()
    
    # 运行演示
    asyncio.run(service.run_demo())
