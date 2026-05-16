from typing import Optional, List

from langchain_core.tools import tool
from src.tools.api_client import api_client


@tool
async def recommend_schools(
    province: str,
    category: str,
    score: float,
    preferred_majors: Optional[List[str]] = None,
    avoid_majors: Optional[List[str]] = None,
    preferred_regions: Optional[List[str]] = None,
    avoid_regions: Optional[List[str]] = None,
    target_level: Optional[str] = None,
) -> dict:
    """高考志愿推荐。根据考生分数、科类、偏好等因素，推荐冲/稳/保三个层次的院校专业。

    Args:
        province: 省份（如 辽宁）
        category: 科类 — 物理 或 历史
        score: 高考分数
        preferred_majors: 想学的专业列表（可选）
        avoid_majors: 不想学的专业列表（可选）
        preferred_regions: 偏好的地区列表（可选）
        avoid_regions: 规避的地区列表（可选）
        target_level: 目标学校层次（可选，如 985、211、一本）
    """
    # Step 1: 查位次
    rank = None
    rank_info = None
    try:
        rank_info = await api_client.rank_query(province, category, score)
        if rank_info and "rank" in rank_info:
            rank = rank_info["rank"]
    except Exception:
        pass  # 位次查询失败不阻塞，直接传分数给推荐接口

    # Step 2: 调推荐接口
    payload: dict = {
        "province": province,
        "category": category,
        "score": int(score),
    }
    if rank:
        payload["rank"] = rank
    if preferred_majors:
        payload["preferred_majors"] = preferred_majors
    if avoid_majors:
        payload["avoid_majors"] = avoid_majors
    if preferred_regions:
        payload["preferred_regions"] = preferred_regions
    if avoid_regions:
        payload["avoid_regions"] = avoid_regions
    if target_level:
        payload["target_level"] = target_level

    result = await api_client.recommend(payload)
    if "error" in result:
        return {"error": result["error"]}

    # Step 3: 附加位次信息到返回结果
    if rank_info and "rank" in rank_info:
        result["rank_info"] = {
            "score": score,
            "rank": rank_info["rank"],
            "year": rank_info.get("year", 2025),
            "ranks_3year": rank_info.get("ranks_3year", {}),
        }

    return result
