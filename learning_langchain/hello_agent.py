"""
Example: 1
This is a simple example of a LangChain agent which simple prompt and llm combination to create very basic LLM agent
""" 
from langchain_core.prompts import ChatPromptTemplate
from load_env import configure_environment
from langchain_openai import ChatOpenAI
import os

def get_llm(apikey,model= "gpt-4o", temperature =0):
    return ChatOpenAI(model=model, temperature=temperature, api_key=apikey)

     

def main():
    configure_environment()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OpenAI API key is not set")

    llm = get_llm(openai_api_key)
    
    prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{query}"),
        ]
    )

    llm_chain = prompt | llm
    
    response = llm_chain.invoke({"query": "Tell me a fun fact about space."})
    print("\nAI Agent says:")
    print(response.content)
    
if __name__ == "__main__":
    main()