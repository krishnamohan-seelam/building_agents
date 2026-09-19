from types import SimpleNamespace

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from learning_langchain.planner_executor import planner_executor_agent, trip_planner


class FakePlannerLLM:
    def __init__(self, structured: bool = False) -> None:
        self.structured = structured
        self.chat_model = FakeListChatModel(responses=["Research completed."])

    def with_structured_output(self, _schema):
        return FakePlannerLLM(structured=True)

    def bind_tools(self, _tools):
        return FakePlannerLLM()

    def invoke(self, messages):
        if self.structured:
            return SimpleNamespace(steps=["Complete the requested research."])
        response = self.chat_model.invoke(messages)
        return AIMessage(content=response.content)


@pytest.mark.parametrize("module", [trip_planner, planner_executor_agent])
def test_planner_memory_survives_restart_and_isolates_threads(
    module, tmp_path, monkeypatch
):
    monkeypatch.setattr(module, "get_llm", lambda bind_tools=False: FakePlannerLLM())
    database_path = tmp_path / "planner.sqlite3"
    thread_id = "conversation-a"

    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        app = module.build_graph(checkpointer=checkpointer)
        run = module.run_trip_advisor if module is trip_planner else module.run_agent
        run("Plan a trip to Kyoto.", app=app, thread_id=thread_id)

    with SqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        app = module.build_graph(checkpointer=checkpointer)
        run = module.run_trip_advisor if module is trip_planner else module.run_agent
        run("What should I pack?", app=app, thread_id=thread_id)
        state = app.get_state({"configurable": {"thread_id": thread_id}}).values
        run("Plan a trip to Oslo.", app=app, thread_id="conversation-b")
        isolated_state = app.get_state(
            {"configurable": {"thread_id": "conversation-b"}}
        ).values

    messages = state["messages"]
    assert [message.content for message in messages if message.type == "human"] == [
        "Plan a trip to Kyoto.",
        "What should I pack?",
    ]
    assert len([message for message in messages if message.type == "ai"]) == 2
    assert state["results"] == ["Step 1 (Complete the requested research.): Research completed."]
    assert [message.content for message in isolated_state["messages"] if message.type == "human"] == [
        "Plan a trip to Oslo.",
    ]