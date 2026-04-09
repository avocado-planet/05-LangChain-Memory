# LangChain メモリコンポーネント 

## 全体像

LangChain 1.0 のメモリは **短期メモリ** と **長期メモリ** の2層構造です。

```
                    LangChain のメモリ構造
┌──────────────────────────────────────────────┐
│                                              │
│  短期メモリ（Short-term Memory）              │
│  ─────────────────────────────               │
│  スコープ: 1つの thread（会話）内             │
│  保存対象: メッセージ履歴                     │
│  仕組み:   checkpointer + thread_id          │
│                                              │
│  ┌─ thread-1 ─┐  ┌─ thread-2 ─┐            │
│  │ Human: ... │  │ Human: ... │             │
│  │ AI: ...    │  │ AI: ...    │             │
│  │ Human: ... │  │ Human: ... │             │
│  └────────────┘  └────────────┘             │
│       ↑ 独立 ↑        ↑ 独立 ↑               │
│                                              │
├──────────────────────────────────────────────┤
│                                              │
│  長期メモリ（Long-term Memory）               │
│  ─────────────────────────────               │
│  スコープ: thread をまたぐ                    │
│  保存対象: 任意の JSON データ                 │
│  仕組み:   Store (namespace + key)           │
│                                              │
│  ┌─ ("users",) ────────────────┐            │
│  │ "user_001" → {name: "タロウ"} │            │
│  │ "user_002" → {name: "ハナコ"} │            │
│  └─────────────────────────────┘            │
│       ↑ 全 thread から共有 ↑                  │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Part 1: 短期メモリ

### checkpointer

メッセージ履歴を保存し、同じ `thread_id` で `invoke` するたびに前回の続きとして扱う仕組みです。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[],
    checkpointer=InMemorySaver(),  # これだけで短期メモリが有効に
)
```

checkpointer がない場合、毎回の `invoke` はまっさらな状態から始まります。前の会話内容は一切引き継がれません。

### thread_id

checkpointer がメッセージ履歴を保存する単位です。

```python
config = {"configurable": {"thread_id": "session-123"}}
agent.invoke({"messages": [...]}, config=config)
```

- 同じ `thread_id` → 前回の会話の続き
- 違う `thread_id` → 新しい会話（前の記憶は見えない）

ユーザーごと・セッションごとに異なる `thread_id` を使うことで会話を分離できます。

### コンテキスト管理の3つの手法

会話が長くなるとコンテキストウィンドウを超えます。以下の3つの手法で対処します。

#### 1. トリミング（Trim）

古いメッセージを切り捨て、最新のN件だけ残します。`@before_model` ミドルウェアで実装します。

```python
@before_model
def trim_messages(state, runtime):
    messages = state["messages"]
    if len(messages) <= 4:
        return None
    first_msg = messages[0]
    recent = messages[-3:]
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg, *recent,
        ]
    }
```

メリット: シンプル、トークン消費が少ない。
デメリット: 古い情報が完全に失われる。

#### 2. 削除（Delete）

特定のメッセージを State から削除します。センシティブ情報のフィルタリング等に使います。

```python
@after_model
def filter_sensitive(state, runtime):
    last_message = state["messages"][-1]
    if "password" in last_message.content:
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None
```

注意: 削除後のメッセージ履歴が有効な形式を保つように気をつけてください。特に「`tool_calls` 付き AIMessage の後に ToolMessage がない」状態を作ると OpenAI API がエラーを返します。

#### 3. 要約（Summarize）

`SummarizationMiddleware` がビルトインで用意されています。閾値を超えた古いメッセージを自動的に要約して残すので、情報を保持しつつコンテキストを圧縮できます。

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",  # 要約用のモデル（安価）
            trigger=("tokens", 4000),      # 4000トークンで発動
            keep=("messages", 20),         # 最新20件は保持
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

メリット: 古い情報を要約として保持できる。
デメリット: 要約に API コストがかかる、要約精度に依存する。

### 3つの手法の比較

| 手法 | 情報の保持 | コスト | 実装の簡単さ |
|---|---|---|---|
| トリミング | 古い情報は消える | 低い | 簡単 |
| 削除 | 特定メッセージだけ消す | 低い | 簡単 |
| 要約 | 要約として残る | 中（要約API） | ビルトイン |

---

## Part 2: 長期メモリ

### Store とは

thread をまたいで持続するデータストアです。`namespace`（タプル、フォルダのようなもの）と `key`（文字列、ファイル名のようなもの）でデータを整理します。

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 書き込み
store.put(
    ("users",),        # namespace
    "user_001",        # key
    {"name": "タロウ"}, # value（任意のJSON）
)

# 読み取り
item = store.get(("users",), "user_001")
print(item.value)  # {"name": "タロウ"}

# 検索
items = store.search(("users",))
```

### namespace の階層構造

namespace はタプルなので階層化できます:

```python
("users",)                          # ユーザー全体
("users", "user_001",)              # 特定ユーザー
("users", "user_001", "preferences") # そのユーザーの設定
```

### エージェントからの使い方

Store にアクセスするにはツールの `runtime.store` を使います。

```python
from langchain.tools import ToolRuntime, tool
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str

@tool
def get_profile(runtime: ToolRuntime[UserContext]) -> str:
    """ユーザー情報を取得"""
    item = runtime.store.get(("users",), runtime.context.user_id)
    return str(item.value) if item else "情報なし"

@tool
def save_profile(name: str, runtime: ToolRuntime[UserContext]) -> str:
    """ユーザー情報を保存"""
    runtime.store.put(("users",), runtime.context.user_id, {"name": name})
    return "保存しました"

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_profile, save_profile],
    store=InMemoryStore(),
    context_schema=UserContext,
)

# context で user_id を渡す
result = agent.invoke(
    {"messages": [...]},
    context=UserContext(user_id="user_001"),
)
```

`runtime` パラメータはモデルからは見えない（hidden）ため、LLM がツールを呼ぶときに混乱しません。

### context_schema

`invoke` 時にエージェントに渡す追加情報の型定義です。`user_id` のようにリクエストごとに異なる値を渡すのに使います。

```python
@dataclass
class UserContext:
    user_id: str

agent = create_agent(
    ...,
    context_schema=UserContext,
)

# invoke 時に context を渡す
agent.invoke(
    {"messages": [...]},
    context=UserContext(user_id="user_001"),
)
```

ツール内では `runtime.context.user_id` でアクセスできます。

---

## 短期 + 長期の組み合わせ

本番のエージェントでは両方を併用します:

```python
agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_profile, save_profile],
    checkpointer=InMemorySaver(),  # 短期: 会話履歴
    store=InMemoryStore(),         # 長期: ユーザー情報
    context_schema=UserContext,
    middleware=[
        SummarizationMiddleware(   # 短期: 自動要約
            model="openai:gpt-4.1-mini",
            trigger=("tokens", 4000),
        ),
    ],
)
```

この構成では:
- 同じ `thread_id` 内 → メッセージ履歴が checkpointer に保存される
- 長い会話 → SummarizationMiddleware が自動要約
- ユーザー情報 → Store にツール経由で保存/取得
- 別の thread → 短期メモリはリセットされるが、Store のデータは残る

---

## 本番環境

| 開発用 | 本番用 |
|---|---|
| `InMemorySaver()` | `PostgresSaver` / `SqliteSaver` |
| `InMemoryStore()` | `PostgresStore` |

```bash
pip install langgraph-checkpoint-postgres
```

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://..."

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    with PostgresStore.from_conn_string(DB_URI) as store:
        checkpointer.setup()
        store.setup()
        agent = create_agent(
            model="...",
            tools=[...],
            checkpointer=checkpointer,
            store=store,
        )
```

---

## 参考リンク

- [Short-term memory 公式ドキュメント](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [Long-term memory 公式ドキュメント](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangMem SDK](https://blog.langchain.com/langmem-sdk-launch/)
