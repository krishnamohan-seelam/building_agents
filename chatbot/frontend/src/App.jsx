import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Send, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'ai', content: "Hello! I'm your unified AI assistant. I have internet and MCP capabilities. How can I help you today?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/api/chat', {
        user_id: 'web_user',
        session_id: 'session_1',
        message: userMessage
      });
      
      setMessages(prev => [...prev, { role: 'ai', content: response.data.reply }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { role: 'ai', content: '**Error System:** Connection failed or server error.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="chat-header">
        <Bot size={28} color="#6366f1" />
        <h1>Agentic AI Workspace</h1>
        <div className="status-indicator" title="Connected"></div>
      </header>

      <main className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.role === 'ai' ? (
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            ) : (
              msg.content
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message ai">
            <div className="typing-indicator">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <form onSubmit={handleSend} className="input-container">
        <input
          type="text"
          className="chat-input"
          placeholder="Type your message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="send-button" disabled={!input.trim() || isLoading}>
          <Send size={20} />
        </button>
      </form>
    </div>
  );
}

export default App;
