import pytest
from unittest.mock import MagicMock, patch
from langchain_agents.skills.code_reviewer import CodeReviewer, review_code_tool
from langchain_agents.skills.code_generator import CodeGenerator, generate_code_tool

class TestLangChainSkills:
    
    @patch("langchain_agents.skills.code_reviewer.ChatGoogleGenerativeAI")
    @patch("langchain_agents.skills.code_reviewer.configure_environment")
    def test_code_reviewer_init(self, mock_config, mock_llm_class):
        """Test initialization of CodeReviewer."""
        reviewer = CodeReviewer()
        assert reviewer.llm is not None
        mock_config.assert_called_once()

    @patch("langchain_agents.skills.code_reviewer.ChatGoogleGenerativeAI")
    @patch("langchain_agents.skills.code_reviewer.configure_environment")
    def test_code_reviewer_review(self, mock_config, mock_llm_class):
        """Test the review method with mocked chain."""
        reviewer = CodeReviewer()
        reviewer.chain = MagicMock()
        reviewer.chain.invoke.return_value = "# Premium Review\n\n- No issues found."
        
        result = reviewer.review("def test(): pass")
        assert "# Premium Review" in result
        reviewer.chain.invoke.assert_called_once()

    @patch("langchain_agents.skills.code_reviewer.ChatGoogleGenerativeAI")
    @patch("langchain_agents.skills.code_reviewer.configure_environment")
    def test_code_reviewer_empty(self, mock_config, mock_llm_class):
        """Test review with empty input."""
        reviewer = CodeReviewer()
        result = reviewer.review("  ")
        assert "Error" in result

    @patch("langchain_agents.skills.code_generator.ChatGoogleGenerativeAI")
    @patch("langchain_agents.skills.code_generator.configure_environment")
    def test_code_generator_init(self, mock_config, mock_llm_class):
        """Test initialization of CodeGenerator."""
        generator = CodeGenerator()
        assert generator.llm is not None
        mock_config.assert_called_once()

    @patch("langchain_agents.skills.code_generator.ChatGoogleGenerativeAI")
    @patch("langchain_agents.skills.code_generator.configure_environment")
    def test_code_generator_generate(self, mock_config, mock_llm_class):
        """Test the generate method with mocked chain."""
        generator = CodeGenerator()
        generator.chain = MagicMock()
        generator.chain.invoke.return_value = "```python\nclass MyAgent:\n    pass\n```"
        
        result = generator.generate("Simple agent")
        assert "```python" in result
        generator.chain.invoke.assert_called_once()

    @patch("langchain_agents.skills.code_reviewer.CodeReviewer")
    def test_review_code_tool(self, mock_reviewer_class):
        """Test the review_code_tool function wrapper."""
        mock_reviewer = MagicMock()
        mock_reviewer_class.return_value = mock_reviewer
        mock_reviewer.review.return_value = "Good code"
        
        # Test direct call (simulating how langgraph/langchain invokes it)
        result = review_code_tool.invoke({"code": "print(1)"})
        assert result == "Good code"
        mock_reviewer.review.assert_called_with("print(1)")

    @patch("langchain_agents.skills.code_generator.CodeGenerator")
    def test_generate_code_tool(self, mock_generator_class):
        """Test the generate_code_tool function wrapper."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate.return_value = "print(2)"
        
        result = generate_code_tool.invoke({"task": "Say hi"})
        assert result == "print(2)"
        mock_generator.generate.assert_called_with("Say hi")
