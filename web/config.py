# Ollama大语言模型配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"  # 使用您已安装的DeepSeek-R1模型

# 聊天配置
MAX_HISTORY_LENGTH = 10  # 最大历史记录长度
MAX_TOKENS = 2000  # 最大token数
TEMPERATURE = 0.7  # 温度参数

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的药物设计AI助手，专门帮助研究人员进行药物分子设计和分析。

你的专业领域包括：
1. 分子结构和性质分析
2. 药物设计策略建议
3. 化学合成路线规划
4. 分子优化建议
5. 药物-靶点相互作用分析
6. ADMET性质预测
7. 化学信息学相关问题

请用专业但易懂的语言回答用户的问题，并提供具体的建议和解释。如果涉及分子结构，请尽可能提供SMILES表达式或结构描述。

记住：
- 保持专业性和准确性
- 提供具体的建议和解释
- 使用中文回答
- 如果遇到不确定的问题，请诚实说明
"""

# 配置说明：
# 1. 确保Ollama服务已启动：ollama serve
# 2. 确保模型已下载：ollama pull deepseek-r1:1.5b
# 3. 如果使用其他模型，请修改 OLLAMA_MODEL_NAME