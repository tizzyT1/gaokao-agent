from langchain_core.tools import tool
from src.tools.api_client import api_client


@tool
async def query_rank(province: str, category: str, score: float) -> dict:
    """查询分数对应的位次。根据省份、科类、分数查询高考位次，返回近三年位次对比和附近分数参考。

    Args:
        province: 省份（如 辽宁）
        category: 科类 — 物理 或 历史
        score: 高考分数
    """
    result = await api_client.rank_query(province, category, score)
    if result is None:
        return {"error": f"未找到{province}{category}{score}分的位次数据"}
    return result
