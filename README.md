# 高考志愿填报智能 Agent

基于 LangGraph + DeepSeek V4 的高考志愿填报对话式助手，专为辽宁省考生设计。

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                      用户 / 前端                      │
│                POST /chat  |  POST /chat/stream      │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Agent 层       │  FastAPI :8003
              │   LangGraph      │
              │                  │
              │  route ──► tools ──► respond            │
              │  (意图路由)  (工具调用)  (LLM生成回复)    │
              │                  │
              │  DeepSeek V4     │
              │  ├─ Flash (路由)  │
              │  └─ Pro   (回复)  │
              └────────┬────────┘
                       │ HTTP
              ┌────────▼────────┐
              │   推荐引擎 API    │  FastAPI :8000
              │                  │
              │  /recommend      │  冲/稳/保推荐
              │  /rank_query     │  分数→位次
              │  /score_query    │  位次→分数
              │  /school_analysis│  学校分析
              │  /major_analysis │  专业分析
              │  /search         │  模糊搜索
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   SQLite         │  sessions.db
              │   会话持久化      │  关键字段存储
              └─────────────────┘
```

### 两层架构

| 层 | 端口 | 框架 | 职责 |
|----|------|------|------|
| **Agent 对话层** | 8003 | LangGraph + FastAPI | 理解用户意图、管理对话状态、调用工具、生成自然语言回复 |
| **推荐引擎层** | 8000 | FastAPI | 位次转换、概率计算、冲/稳/保分层、学校/专业分析、纯结构化数据返回 |

### 为什么分两层

- **推荐引擎**是确定性的：位次计算、概率模型、分层逻辑都由规则和数据驱动，不需要 LLM 参与
- **Agent 对话层**是非确定性的：用户表达方式千变万化，需要 LLM 理解意图并生成自然语言
- 分层后各自独立部署、独立迭代，推荐引擎专注数据正确性，Agent 专注对话体验

---

## 技术选型

### Agent 对话层

| 技术 | 用途 | 选型原因 |
|------|------|----------|
| **LangGraph** | 对话流程编排 | StateGraph 状态机，节点/条件边清晰，支持流式事件 |
| **DeepSeek V4 Pro** | 回复生成 | DeepSeek 最新旗舰，中文能力强，API 兼容 OpenAI SDK |
| **DeepSeek V4 Flash** | 意图路由 | 轻量快速，路由只需小段 JSON，不需要 Pro 的能力 |
| **FastAPI** | HTTP 服务 | 异步原生支持，SSE 流式简单，Python 生态契合 |
| **SQLite** | 会话持久化 | 零配置、嵌入式、够用，只存结构化关键字段 |
| **httpx** | HTTP 客户端 | 异步支持，连接池，超时控制 |
| **LangChain** | LLM 抽象 + Tool 定义 | ChatOpenAI 兼容 DeepSeek，@tool 装饰器方便 |

### 推荐引擎层

| 技术 | 用途 | 选型原因 |
|------|------|----------|
| **FastAPI** | API 服务 | 高性能异步，自动 OpenAPI 文档 |
| **Pydantic** | 数据模型 | 请求/响应验证，类型安全 |
| **NumPy / math** | 概率计算 | 正态分布 CDF、变异系数 |
| **JSON 文件** | 数据存储 | 历年录取数据标准化后落盘，启动时加载到内存 |

### 未使用技术及原因

| 技术 | 为什么不选 |
|------|-----------|
| **DeepSeek Tool Calling** | V3 时期 bind_tools 静默失败、tool_choice 报错，V4 待验证 |
| **Redis / PostgreSQL** | 当前数据量不需要，SQLite 够用 |
| **LangGraph 原生 ToolNode** | DeepSeek 不原生支持 tool calling，改为手动路由 + 工具调用 |

---

## 项目结构

```
D:\project\
├── src/
│   ├── main.py              # FastAPI 入口，/chat /chat/stream /health
│   ├── config.py            # 配置（pydantic-settings，读 .env）
│   ├── state.py             # GaokaoState（LangGraph 状态定义 + 自定义 reducer）
│   ├── db.py                # SQLite 持久化（sessions 表 CRUD）
│   ├── graph.py             # LangGraph 图定义（route → tools → respond）
│   ├── prompts/
│   │   └── system.py        # System Prompt（防幻觉、输出格式、降级策略）
│   ├── tools/
│   │   ├── api_client.py    # httpx 异步客户端（封装所有后端端点）
│   │   ├── recommend.py     # 推荐工具（分数→位次→推荐→格式化）
│   │   ├── school.py        # 学校分析工具
│   │   ├── major.py         # 专业分析工具
│   │   ├── search.py        # 模糊搜索工具
│   │   └── rank_query.py    # 位次查询工具（独立）
│   └── utils/
│       └── scoring.py       # 风险窗口计算
├── .env                     # 环境变量（API Key、模型名、后端地址）
├── sessions.db              # SQLite 会话数据（自动生成）
├── QA-review.md             # 设计决策与问题记录
└── README.md                # 本文件
```

---

## 核心设计决策

### 1. 意图路由：正则优先 + LLM 兜底

```
用户消息
  │
  ├─ 正则提取关键信息（省份/科类/分数/偏好...）
  │   └─ 信息不全 → 精准追问缺失字段
  │   └─ 信息齐全 → 触发 recommend
  │
  ├─ 正则匹配显式意图
  │   ├─ "XX大学怎么样" → school
  │   ├─ "XX专业前景" → major
  │   ├─ "位次/排名"    → rank_query
  │   └─ "搜索/有哪些"  → search
  │
  └─ 以上都不匹配 → chat（LLM 通用回复）
```

**原因**：DeepSeek V3 的 JSON 提取不稳定，正则先兜底高频率场景。V4 后待验证 function calling。

### 2. 渐进式信息收集

用户不需要一口气说全，agent 逐轮积累 `user_profile`：

```
"我想学计算机"       → 记住偏好，追问科类+分数
"理科"              → 记住物理类，追问分数
"600分"             → 信息齐全，触发推荐
```

Session 之间完全隔离，profile 持久化到 SQLite，服务重启不丢失。

### 3. 双 LLM 实例

| 实例 | 模型 | streaming | 场景 |
|------|------|-----------|------|
| `llm_route` | deepseek-v4-flash | False | 意图分类、信息提取（低延迟、低成本） |
| `llm_respond` | deepseek-v4-pro | True | 冲/稳/保报告生成（质量优先、支持流式输出） |

### 4. 四阶段主线流程

```
collecting  →  recommending  →  deep_dive  →  adjusting
(收集信息)     (生成推荐)       (深度追问)     (调整偏好)
```

每轮注入当前阶段到 System Prompt，防止 LLM 在长对话中注意力偏移。

### 5. 省份限定：仅辽宁

- 省份默认为辽宁，不需要用户主动提供
- 用户提到其他省份 → 明确告知 "仅支持辽宁省，其他省份暂未覆盖"
- 必填字段只有：**科类** + **分数**

### 6. 推荐链路：先查位次再推荐

```
用户分数 → /rank_query（分数→位次+三年对比）
         → /recommend（分数+位次+偏好 → 冲/稳/保）
         → LLM 格式化（附位次信息）
```

位次查询失败不阻塞推荐，后端内部可自转位次。

---

## API 端点

### Agent 服务 (8003)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 对话，返回完整 JSON |
| POST | `/chat/stream` | 流式对话，SSE 逐字推送 |
| GET | `/health` | 健康检查 |
| DELETE | `/session/{id}` | 清除指定会话 |

**请求格式**：
```json
{
    "session_id": "唯一会话标识",
    "message": "用户消息"
}
```

### 后端推荐引擎 (8000)

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/recommend` | 志愿推荐（冲/稳/保） |
| GET | `/rank_query` | 分数 → 位次 |
| GET | `/score_query` | 位次 → 分数 |
| GET | `/school_analysis` | 学校详情 |
| GET | `/major_analysis` | 专业详情 |
| GET | `/search` | 模糊搜索 |
| GET | `/provinces` | 省份列表 |
| GET | `/school_tiers` | 学校层次说明 |
| GET | `/major_categories` | 专业类别 |
| GET | `/health` | 健康检查 |

---

## 数据流（以推荐为例）

```
用户: "计算机" "理科" "600分"
        │
        ▼
┌─ route_node ───────────────────────┐
│ 1. 正则提取: category=物理, score=600 │
│ 2. 合并 profile: +preferred=计算机    │
│ 3. 检查: 科类✅ 分数✅ → 触发推荐      │
│ 4. 返回 action: intent=recommend     │
└─────────────┬───────────────────────┘
              │
        route_after_classify → "tools"
              │
              ▼
┌─ tool_node ────────────────────────┐
│ 1. await rank_query(辽宁,物理,600)   │
│    → rank=13322, 三年对比           │
│ 2. await recommend(辽宁,物理,600,    │
│    rank=13322, preferred=[计算机])   │
│    → 冲x5 / 稳x10 / 保x10          │
│ 3. 返回结构化数据 + rank_info        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─ response_node ────────────────────┐
│ System Prompt + 工具结果 + user_profile │
│ DeepSeek V4 Pro 生成:              │
│ 📊 基本信息(位次13322, 风险窗口)      │
│ 🎯 冲一冲(5所, 附概率趋势)           │
│ ✅ 稳一稳(10所, 附211标注)           │
│ 🛡️ 保一保(10所, CV风险提醒)         │
│ ⚠️ 特别提醒(大小年预警)              │
└─────────────────────────────────────┘
```

---

## 启动方式

### 1. 启动后端推荐引擎（终端 1）

```bash
cd 后端目录
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. 启动 Agent 服务（终端 2）

```bash
cd D:\project
python -m uvicorn src.main:app --host 0.0.0.0 --port 8003
```

### 3. 测试

```bash
curl -X POST http://127.0.0.1:8003/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"t1","message":"辽宁物理600分，想学计算机"}'
```

---

## 环境变量（.env）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 必填 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_PRO_MODEL` | 回复层模型 | `deepseek-v4-pro` |
| `DEEPSEEK_FLASH_MODEL` | 路由层模型 | `deepseek-v4-flash` |
| `BACKEND_API_URL` | 推荐引擎地址 | `http://127.0.0.1:8000` |
| `BACKEND_TIMEOUT` | 后端超时（秒） | `30.0` |
