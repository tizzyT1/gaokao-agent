import json
import re
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from src.state import GaokaoState
from src.config import settings
from src.prompts.system import SYSTEM_PROMPT
from src.tools.recommend import recommend_schools
from src.tools.school import analyze_school
from src.tools.major import analyze_major
from src.tools.search import search_schools_majors
from src.tools.rank_query import query_rank


# ── 双 LLM 实例 ──────────────────────────────────────
llm_route = ChatOpenAI(
    model=settings.deepseek_flash_model,
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    temperature=0,
    streaming=False,
)

llm_respond = ChatOpenAI(
    model=settings.deepseek_pro_model,
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url,
    temperature=0.3,
    streaming=True,
)

# ── LLM 路由 prompt（非推荐意图分类） ──────────────────
ROUTER_PROMPT = """用户消息: {user_msg}

判断用户意图，只输出一行JSON：
- 问某所大学 → {"intent":"school","school_name":"大学名","reply":null}
- 问某个专业 → {"intent":"major","major_name":"专业名","reply":null}
- 模糊搜索 → {"intent":"search","keyword":"关键词","reply":null}
- 闲聊/问候 → {"intent":"chat","reply":"你的回复内容"}

注意：如果用户在提供或补充信息（分数、科类、偏好等），输出 {"intent":"chat","reply":null}
现在输出:"""


# ── 正则信息提取 ──────────────────────────────────────

# 全国省份列表（用于检测用户是否提到非辽宁省份）
ALL_PROVINCES = [
    "辽宁","北京","上海","广东","浙江","江苏","四川","湖北","湖南",
    "河南","山东","河北","陕西","福建","安徽","重庆","天津","吉林",
    "黑龙江","山西","江西","广西","云南","贵州","海南","甘肃","青海",
    "宁夏","新疆","西藏","内蒙古",
]

MAJOR_KEYWORDS = {
    "计算机": "计算机", "计科": "计算机", "软件": "软件工程",
    "人工智能": "人工智能", "电子": "电子信息", "通信": "通信工程",
    "医学": "医学", "临床": "临床医学", "口腔": "口腔医学",
    "金融": "金融学", "会计": "会计学", "法学": "法学", "师范": "师范",
    "土木": "土木工程", "机械": "机械工程", "电气": "电气工程",
    "自动化": "自动化", "数学": "数学", "物理学": "物理学",
    "材料": "材料科学", "生物": "生物工程", "化学": "化学",
    "建筑": "建筑学", "环境": "环境工程", "交通": "交通运输",
    "车辆": "车辆工程", "能源": "能源与动力", "航空航天": "航空航天",
}

REGIONS_MAP = {
    "东北": ["辽宁","吉林","黑龙江"],
    "华东": ["上海","江苏","浙江","安徽","福建","江西","山东"],
    "华南": ["广东","广西","海南"],
    "西南": ["四川","重庆","贵州","云南","西藏"],
    "西北": ["陕西","甘肃","青海","宁夏","新疆"],
    "华北": ["北京","天津","河北","山西","内蒙古"],
    "华中": ["河南","湖北","湖南"],
    "江浙沪": ["江苏","浙江","上海"],
    "北上广": ["北京","上海","广东"],
}


def _extract_user_info(text: str) -> dict:
    """从用户消息中提取关键信息，总是返回 dict（可能部分字段为空）。"""
    info = {}

    # 省份
    for p in ALL_PROVINCES:
        if p in text:
            info["province"] = p
            break

    # 科类
    if "理科" in text or "物理类" in text or "物理" in text:
        info["category"] = "物理"
    elif "文科" in text or "历史类" in text or "历史" in text:
        info["category"] = "历史"

    # 分数
    score_match = re.search(r"(\d{3})\s*分", text)
    if score_match:
        info["score"] = int(score_match.group(1))

    # 专业偏好
    majors = []
    for kw, maj in MAJOR_KEYWORDS.items():
        if kw in text and maj not in majors:
            majors.append(maj)
    if majors:
        info["preferred_majors"] = majors

    # 规避专业
    avoid = []
    for m in re.finditer(r"不想[学去读](\w+)", text):
        avoid.append(m.group(1))
    for m in re.finditer(r"不要(\w+)", text):
        avoid.append(m.group(1))
    if avoid:
        info["avoid_majors"] = avoid

    # 地区偏好
    pref_regions = []
    for region, plist in REGIONS_MAP.items():
        if region in text:
            pref_regions.extend(plist)
    if pref_regions:
        info["preferred_regions"] = list(set(pref_regions))

    # 目标层次
    if "985" in text:
        info["target_level"] = "985"
    elif "211" in text:
        info["target_level"] = "211"
    elif "双一流" in text:
        info["target_level"] = "双一流"
    elif "一本" in text:
        info["target_level"] = "一本"
    elif "二本" in text:
        info["target_level"] = "二本"

    return info


# ── 信息完整性检查 ────────────────────────────────────

MISSING_QUESTIONS = {
    "category": "请问你是物理类（理科）还是历史类（文科）呢？",
    "score": "请问你今年高考考了多少分呀？",
}


def _missing_fields(profile: dict) -> list[str]:
    missing = []
    if not profile.get("category"):
        missing.append("category")
    if not profile.get("score"):
        missing.append("score")
    return missing


def _build_context_prompt(profile: dict) -> str:
    """用 user_profile 生成上下文，注入到 system prompt 前。"""
    parts = []
    if profile.get("province"):
        parts.append(f"当前考生省份：{profile['province']}")
    if profile.get("category"):
        parts.append(f"科类：{profile['category']}")
    if profile.get("score"):
        parts.append(f"分数：{profile['score']}")
    if profile.get("preferred_majors"):
        parts.append(f"偏好的专业：{'、'.join(profile['preferred_majors'])}")
    if profile.get("avoid_majors"):
        parts.append(f"规避的专业：{'、'.join(profile['avoid_majors'])}")
    if profile.get("preferred_regions"):
        parts.append(f"偏好的地区：{'、'.join(profile['preferred_regions'])}")
    if profile.get("target_level"):
        parts.append(f"目标层次：{profile['target_level']}")
    if parts:
        return "## 考生当前信息\n" + "\n".join(parts)
    return ""


# ── 节点 ──────────────────────────────────────────────

async def route_node(state: GaokaoState) -> dict:
    """每轮的入口：提取信息 → 合并 profile → 判断阶段 → 返回 action。"""
    last_human = ""
    for m in reversed(state["messages"]):
        if hasattr(m, "type") and m.type == "human":
            last_human = m.content
            break

    profile = dict(state.get("user_profile", {}))

    # Step 1: 正则提取关键信息
    extracted = _extract_user_info(last_human)

    # Step 2: 省份校验 — 提到非辽宁则拦截
    other_province = extracted.get("province", "")
    if other_province and other_province != "辽宁":
        action = {
            "intent": "chat",
            "reply": f"抱歉，目前我们只支持**辽宁省**的报考数据，暂不支持{other_province}的考生。如果你是辽宁考生，请直接告诉我你的科类和分数，我来帮你推荐。",
        }
        return {
            "messages": [AIMessage(content=action["reply"], additional_kwargs={"action": action})],
        }

    # 辽宁的省份信息保留（或默认）
    if extracted.get("province") == "辽宁":
        pass  # 正常，保留在 extracted 中

    # Step 3: 合并 profile
    merged_profile = dict(profile)
    for k, v in extracted.items():
        if v is not None and v != "" and v != [] and v != 0:
            if k in merged_profile and isinstance(v, list) and isinstance(merged_profile.get(k), list):
                merged_profile[k] = list(set(merged_profile[k] + v))
            else:
                merged_profile[k] = v

    # Step 4: 检测用户是否在问位次
    rank_match = re.search(r"(多少|什么|查|看|问|是).{0,3}(位次|排名|排多少|多少名)", last_human)
    if rank_match and merged_profile.get("score"):
        action = {
            "intent": "rank_query",
            "province": merged_profile.get("province", "辽宁"),
            "category": merged_profile.get("category", "物理"),
            "score": merged_profile["score"],
            "reply": None,
        }
        return {
            "user_profile": merged_profile,
            "stage": "deep_dive",
            "messages": [AIMessage(content="", additional_kwargs={"action": action})],
        }

    # Step 5: 检测用户是否在明确问某所学校
    school_match = re.search(r"([一-龥]{2,6}(?:大学|学院|高校)).*?(?:怎么|如何|是什么|了解)", last_human)
    if school_match:
        school_name = school_match.group(1)
        action = {"intent": "school", "school_name": school_name, "reply": None}
        return {
            "user_profile": merged_profile,
            "stage": "deep_dive",
            "messages": [AIMessage(content="", additional_kwargs={"action": action})],
        }

    # Step 5: 检测用户是否在明确问某个专业
    major_match = re.search(r"([一-龥]{2,4}(?:专业|方向|类)).*?(?:怎么|如何|前景|就业|是什么|了解)", last_human)
    if major_match:
        major_name = major_match.group(1).replace("专业", "").replace("方向", "").replace("类", "")
        action = {"intent": "major", "major_name": major_name, "reply": None}
        return {
            "user_profile": merged_profile,
            "stage": "deep_dive",
            "messages": [AIMessage(content="", additional_kwargs={"action": action})],
        }

    # Step 6: 检测搜索意图
    search_match = re.search(r"(搜索|查找|帮我找|有哪些|有没有).{0,5}?(学校|大学|专业|院校)", last_human)
    if search_match or re.search(r"什么(学校|大学|专业).*?(好|推荐|合适)", last_human):
        kw_match = re.search(r"(?:搜索|查找|找|问|了解)(.{2,10})", last_human)
        kw = kw_match.group(1) if kw_match else last_human[-20:]
        action = {"intent": "search", "keyword": kw, "search_type": "all", "reply": None}
        return {
            "user_profile": merged_profile,
            "stage": "deep_dive",
            "messages": [AIMessage(content="", additional_kwargs={"action": action})],
        }

    # Step 7: 无明确意图 → 按推荐流程检查信息完整性
    missing = _missing_fields(merged_profile)

    if missing:
        # 有缺失 → 生成追问
        confirmed = []
        if merged_profile.get("province"):
            confirmed.append(f"{merged_profile['province']}")
        if merged_profile.get("category"):
            confirmed.append("物理类" if merged_profile["category"] == "物理" else "历史类")
        if merged_profile.get("score"):
            confirmed.append(f"{merged_profile['score']}分")
        if merged_profile.get("preferred_majors"):
            confirmed.append(f"想学{'、'.join(merged_profile['preferred_majors'])}")
        if merged_profile.get("target_level"):
            confirmed.append(f"目标{merged_profile['target_level']}")

        questions = [MISSING_QUESTIONS[f] for f in missing]

        if confirmed:
            reply = f"好的，我记下了：{'，'.join(confirmed)}。"
        else:
            reply = "你好！我是高考志愿填报助手，目前支持**辽宁省**的报考指导。"

        if "score" in missing and "category" in missing:
            reply += f" 还需要你告诉我——{' '.join(questions)}"
        elif len(missing) == 1:
            reply += f" 还需要你告诉我——{questions[0]}"
        elif missing:
            reply += f" 还需要——{' '.join(questions)}"

        action = {"intent": "chat", "reply": reply}
        return {
            "user_profile": merged_profile,
            "stage": "collecting",
            "messages": [AIMessage(content=reply, additional_kwargs={"action": action})],
        }

    # Step 8: 信息齐全 → 走推荐
    action = {
        "intent": "recommend",
        "province": merged_profile.get("province", "辽宁"),
        "category": merged_profile.get("category"),
        "score": merged_profile.get("score"),
        "preferred_majors": merged_profile.get("preferred_majors"),
        "avoid_majors": merged_profile.get("avoid_majors"),
        "preferred_regions": merged_profile.get("preferred_regions"),
        "avoid_regions": merged_profile.get("avoid_regions"),
        "target_level": merged_profile.get("target_level"),
        "reply": None,
    }
    return {
        "user_profile": merged_profile,
        "stage": "recommending",
        "messages": [AIMessage(content="", additional_kwargs={"action": action})],
    }


async def tool_node(state: GaokaoState) -> dict:
    """Execute the appropriate tool based on intent."""
    last_msg = state["messages"][-1]
    action_data = last_msg.additional_kwargs.get("action", {})
    intent = action_data.get("intent", "chat")

    result = None
    tool_name = ""

    try:
        if intent == "recommend":
            tool_name = "recommend_schools"
            result = await recommend_schools.ainvoke({
                "province": action_data.get("province"),
                "category": action_data.get("category"),
                "score": action_data.get("score"),
                "preferred_majors": action_data.get("preferred_majors"),
                "avoid_majors": action_data.get("avoid_majors"),
                "preferred_regions": action_data.get("preferred_regions"),
                "avoid_regions": action_data.get("avoid_regions"),
                "target_level": action_data.get("target_level"),
            })
        elif intent == "school":
            tool_name = "analyze_school"
            result = await analyze_school.ainvoke({
                "school_name": action_data.get("school_name"),
            })
        elif intent == "major":
            tool_name = "analyze_major"
            result = await analyze_major.ainvoke({
                "major_name": action_data.get("major_name"),
            })
        elif intent == "search":
            tool_name = "search_schools_majors"
            result = await search_schools_majors.ainvoke({
                "keyword": action_data.get("keyword"),
                "search_type": action_data.get("search_type", "all"),
            })
        elif intent == "rank_query":
            tool_name = "query_rank"
            result = await query_rank.ainvoke({
                "province": action_data.get("province"),
                "category": action_data.get("category"),
                "score": action_data.get("score"),
            })
    except Exception as e:
        result = {"error": str(e)}

    if result is None:
        result = {}

    return {"messages": [
        HumanMessage(content=f"[工具 {tool_name} 返回结果]\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    ]}


async def response_node(state: GaokaoState) -> dict:
    """Generate final response with context awareness."""
    profile = state.get("user_profile", {})
    stage = state.get("stage", "collecting")
    context_block = _build_context_prompt(profile)

    # 构建带上下文注入的 system message
    system_content = SYSTEM_PROMPT
    if context_block:
        system_content = context_block + "\n\n" + SYSTEM_PROMPT
    if stage == "collecting":
        system_content += "\n\n当前处于信息收集阶段，请友好地引导用户补全信息。"
    elif stage == "recommending":
        system_content += "\n\n当前处于推荐阶段，请严格按照工具返回的数据输出冲/稳/保格式。"
    elif stage == "deep_dive":
        system_content += "\n\n当前处于深度了解阶段，针对用户问的学校或专业给出详细分析。回答完可以轻轻提醒用户如有需要可回到志愿推荐。"

    context_system_msg = SystemMessage(content=system_content)
    response = await llm_respond.ainvoke([context_system_msg] + list(state["messages"]))
    return {"messages": [response]}


def route_after_classify(state: GaokaoState) -> str:
    last_msg = state["messages"][-1]
    action_data = last_msg.additional_kwargs.get("action", {})
    intent = action_data.get("intent", "chat")
    if intent == "chat":
        return END
    return "tools"


# ── 构建图 ────────────────────────────────────────────

def build_graph() -> StateGraph:
    workflow = StateGraph(GaokaoState)

    workflow.add_node("route", route_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("respond", response_node)

    workflow.set_entry_point("route")

    workflow.add_conditional_edges(
        "route",
        route_after_classify,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


graph = build_graph()
