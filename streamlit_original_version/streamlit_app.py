import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="金融智能分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入主应用
from ui_optimization import main

# 运行主应用
if __name__ == "__main__":
    main()
