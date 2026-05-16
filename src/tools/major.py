from langchain_core.tools import tool
from src.tools.api_client import api_client


@tool
async def analyze_major(major_name: str) -> dict:
    """查询专业详细信息。获取某个专业的特点、难度、就业方向、行业前景、适合人群等信息。

    Args:
        major_name: 专业名称（如 人工智能、计算机科学与技术、临床医学）
    """
    result = await api_client.major_analysis(major_name)
    return result
