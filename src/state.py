from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def _merge_profile(existing: dict, new: dict) -> dict:
    """增量合并 user_profile，列表去重叠加，非空值覆盖。"""
    if not existing:
        return dict(new)
    if not new:
        return dict(existing)
    merged = {**existing}
    for k, v in new.items():
        if v is None or v == "" or v == []:
            continue
        if k in merged and isinstance(v, list) and isinstance(merged[k], list):
            merged[k] = list(set(merged[k] + v))
        else:
            merged[k] = v
    return merged


class GaokaoState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_profile: Annotated[dict, _merge_profile]
    stage: str  # "collecting" | "recommending" | "deep_dive"


def default_profile() -> dict:
    return {
        "province": "辽宁",
        "category": "",
        "score": 0,
        "rank": 0,
        "preferred_majors": [],
        "avoid_majors": [],
        "preferred_regions": [],
        "avoid_regions": [],
        "target_level": "",
    }
