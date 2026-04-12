import pytest
from unittest.mock import MagicMock, patch
from langchain_agents.agent import get_dev_agent

class TestDevAgent:
    
    @patch("langchain_agents.agent.create_react_agent")
    @patch("langchain_agents.agent.ChatGoogleGenerativeAI")
    @patch("langchain_agents.agent.configure_environment")
    def test_get_dev_agent(self, mock_config, mock_llm_class, mock_create_agent):
        """Test the agent factory function."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        
        agent = get_dev_agent()
        
        assert agent == mock_agent
        mock_config.assert_called_once()
        mock_create_agent.assert_called_once()
        
        # Verify tools are passed
        args, kwargs = mock_create_agent.call_args
        tools = args[1]
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "review_code_tool" in tool_names
        assert "generate_code_tool" in tool_names
