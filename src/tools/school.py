from langchain_core.tools import tool
from src.tools.api_client import api_client


@tool
async def analyze_school(school_name: str) -> dict:
    """查询学校详细信息。获取某所大学的层次、强势专业、就业情况、保研率、学费等信息。

    Args:
        school_name: 学校全称（如 浙江大学、北京大学）
    """
    result = await api_client.school_analysis(school_name)
    return result
