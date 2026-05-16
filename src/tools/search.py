from langchain_core.tools import tool
from src.tools.api_client import api_client


@tool
async def search_schools_majors(keyword: str, search_type: str = "all") -> dict:
    """模糊搜索学校或专业。当用户不确定学校或专业全称时使用此工具。

    Args:
        keyword: 搜索关键词
        search_type: 搜索类型 — school(只搜学校)、major(只搜专业)、all(都搜，默认)
    """
    result = await api_client.search(keyword, search_type)
    return result
