import pytest
import tempfile
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from prompt_helper import (
    render_system_message,
    render_human_message,
    render_ai_message,
    validate_variables,
    get_template_variables,
    TemplateRegistry,
    SimpleStrategy,
    FewShotStrategy,
    PTCFStrategy,
    PromptBuilder,
    MissingVariableError,
    UnknownTemplateError,
)

# --- Message Factory & Validators Tests ---

def test_get_template_variables():
    assert get_template_variables("Hello {name}, welcome to {place}!") == ["name", "place"]
    assert get_template_variables("Hello World!") == []

def test_validate_variables():
    template = "Hello {name}!"
    # Should not raise anything
    validate_variables(template, {"name": "Alice"})
    
    # Should raise MissingVariableError
    with pytest.raises(MissingVariableError) as exc_info:
        validate_variables(template, {})
    assert "name" in exc_info.value.missing_variables

def test_render_messages():
    sys_msg = render_system_message("System instruction: {instruction}", instruction="Be polite")
    assert isinstance(sys_msg, SystemMessage)
    assert sys_msg.content == "System instruction: Be polite"

    human_msg = render_human_message("Hello, {name}!", name="Alice")
    assert isinstance(human_msg, HumanMessage)
    assert human_msg.content == "Hello, Alice!"

    ai_msg = render_ai_message("Response: {response}", response="Got it")
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Response: Got it"
    assert ai_msg.tool_calls == []

    tool_calls = [{"name": "search", "args": {"query": "test"}, "id": "1", "type": "tool_call"}]
    ai_msg_with_tools = render_ai_message("Thinking...", tool_calls=tool_calls)
    assert ai_msg_with_tools.tool_calls == tool_calls

# --- Template Registry Tests ---

def test_template_registry():
    registry = TemplateRegistry()
    registry.register("welcome", "Welcome, {name}!", version="v1")
    registry.register("welcome", "Welcome back, {name}!", version="v2")

    assert registry.get("welcome", "v1") == "Welcome, {name}!"
    assert registry.get("welcome", "v2") == "Welcome back, {name}!"
    assert registry.list_templates() == ["welcome"]

    with pytest.raises(UnknownTemplateError):
        registry.get("nonexistent")

def test_template_registry_yaml():
    registry = TemplateRegistry()
    yaml_content = """
welcome_msg: "Welcome, {name}!"
farewell_msg:
    v1: "Goodbye, {name}!"
    v2: "See you later, {name}!"
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as temp:
        temp.write(yaml_content)
        temp_path = temp.name

    try:
        registry.load_from_yaml(temp_path)
        assert registry.get("welcome_msg") == "Welcome, {name}!"
        assert registry.get("farewell_msg", "v1") == "Goodbye, {name}!"
        assert registry.get("farewell_msg", "v2") == "See you later, {name}!"
    finally:
        os.remove(temp_path)

# --- Strategies Tests ---

def test_simple_strategy():
    strategy = SimpleStrategy("Hello {name}!", role="human")
    msgs = strategy.render(name="Bob")
    assert len(msgs) == 1
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "Hello Bob!"

def test_few_shot_strategy():
    examples = [{"input": "cat", "output": "feline"}, {"input": "dog", "output": "canine"}]
    example_prompt = "Input: {input}\nOutput: {output}"
    
    strategy = FewShotStrategy(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Translate animals:",
        suffix="Input: {animal}\nOutput:",
        input_variables=["animal"],
        role="human",
    )
    
    msgs = strategy.render(animal="bird")
    assert len(msgs) == 1
    assert "Translate animals:" in msgs[0].content
    assert "Input: cat\nOutput: feline" in msgs[0].content
    assert "Input: bird\nOutput:" in msgs[0].content

def test_ptcf_strategy():
    strategy = PTCFStrategy(
        persona="Expert coder",
        task="Write code for {topic}",
        context="Use language {lang}",
        format_instructions="Return plain text code block",
        role="system"
    )
    
    msgs = strategy.render(topic="sorting", lang="Python")
    assert len(msgs) == 1
    assert isinstance(msgs[0], SystemMessage)
    content = msgs[0].content
    assert "Persona:\nExpert coder" in content
    assert "Task:\nWrite code for sorting" in content
    assert "Context:\nUse language Python" in content
    assert "Format Instructions:\nReturn plain text code block" in content

# --- Prompt Builder Tests ---

def test_prompt_builder():
    builder = (
        PromptBuilder("test_builder")
        .system("System prompt: {sys_var}")
        .human("Human prompt: {human_var}")
        .ai("AI prompt: {ai_var}")
    )
    
    msgs = builder.build(sys_var="S", human_var="H", ai_var="A")
    assert len(msgs) == 3
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "System prompt: S"
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "Human prompt: H"
    assert isinstance(msgs[2], AIMessage)
    assert msgs[2].content == "AI prompt: A"

def test_prompt_builder_history():
    history_msgs = [
        HumanMessage(content="Previous human"),
        AIMessage(content="Previous AI"),
    ]
    
    # History with system message should place history right after system message
    builder_with_sys = (
        PromptBuilder()
        .system("System prompt")
        .human("Current human: {var}")
    )
    
    msgs = builder_with_sys.build(history=history_msgs, var="now")
    assert len(msgs) == 4
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[1] == history_msgs[0]
    assert msgs[2] == history_msgs[1]
    assert isinstance(msgs[3], HumanMessage)
    assert msgs[3].content == "Current human: now"

    # History without system message should prepend history
    builder_no_sys = (
        PromptBuilder()
        .human("Current human: {var}")
    )
    msgs = builder_no_sys.build(history=history_msgs, var="now")
    assert len(msgs) == 3
    assert msgs[0] == history_msgs[0]
    assert msgs[1] == history_msgs[1]
    assert isinstance(msgs[2], HumanMessage)

def test_prompt_builder_to_chat_prompt_template():
    builder = (
        PromptBuilder()
        .system("System prompt")
        .human("Human prompt: {var}")
    )
    
    chat_prompt = builder.to_chat_prompt_template()
    assert isinstance(chat_prompt, ChatPromptTemplate)
    # Check variables
    assert "var" in chat_prompt.input_variables
    assert "history" in chat_prompt.input_variables or "history" in chat_prompt.optional_variables

# --- Error and Edge Cases Tests ---

def test_invalid_roles():
    # SimpleStrategy
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        SimpleStrategy("test", role="invalid").render()

    # PTCFStrategy
    with pytest.raises(ValueError, match="Unsupported role: invalid"):
        PTCFStrategy("P", "T", "C", "F", role="invalid").render()

def test_few_shot_roles():
    strategy_sys = FewShotStrategy(
        examples=[{"input": "a"}],
        example_prompt="Prompt {input}",
        prefix="Prefix",
        suffix="Suffix",
        input_variables=[],
        role="system",
    )
    msgs = strategy_sys.render()
    assert isinstance(msgs[0], SystemMessage)

    strategy_ai = FewShotStrategy(
        examples=[{"input": "a"}],
        example_prompt="Prompt {input}",
        prefix="Prefix",
        suffix="Suffix",
        input_variables=[],
        role="ai",
    )
    msgs = strategy_ai.render()
    assert isinstance(msgs[0], AIMessage)

def test_get_template_variables_parsing_error():
    # A malformed template that should trigger exception handling in get_template_variables
    assert get_template_variables("{invalid syntax") == []

def test_template_registry_invalid_yaml():
    registry = TemplateRegistry()
    # Invalid template value (int instead of str)
    yaml_content = """
invalid_template: 123
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as temp:
        temp.write(yaml_content)
        temp_path = temp.name
    try:
        with pytest.raises(ValueError, match="Invalid YAML structure"):
            registry.load_from_yaml(temp_path)
    finally:
        os.remove(temp_path)

    # Invalid dictionary structure (nested int instead of str)
    yaml_content2 = """
invalid_dict_template:
    v1: 123
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as temp:
        temp.write(yaml_content2)
        temp_path = temp.name
    try:
        with pytest.raises(ValueError, match="Invalid template type"):
            registry.load_from_yaml(temp_path)
    finally:
        os.remove(temp_path)

