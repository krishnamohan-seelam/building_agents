
## Notes

### Why is the name LangChain?  
LangChain allows developers to create workflows by chaining together multiple steps, It basically turns complex logic into a functional pipeline, where the output of one component automatically becomes the input for the next.
This way of expression is called LangChain Expression Language.  

**Just like pipes in unix or linux**  

LCEL is best for linear or branching workflows like RAG (Retrieval-Augmented Generation) or basic content pipelines. For more complex logic with cycles or loops (where the output needs to go back to a previous step), LangGraph is recommended  

```python
    llm = get_llm(openai_api_key)
    
    prompt  = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{question}"),
        ]
    )

    llm_chain = prompt | llm # this is called LCEL (LangChain Expression Language)
```
### Types of memory in LangChain  
#### 1. ConversationBufferMemory
This is the most basic type of memory. It stores the entire conversation history in memory and its great for short conversations.    
#### 2. ConversationSummaryMemory
This type of memory is usually used for longer chats, it summarizes the conversation history and stores the summary in memory.  
#### 3. ConversationBufferWindowMemory
This type of memory keeps a "sliding window" of the last \(k\) interactions. Once you hit that limit, the oldest exchange is dropped. This prevents your prompt from getting too long while keeping the most recent context fresh.  

#### 4. ConversationTokenBufferMemory
This type of memory is similar to ConversationBufferWindowMemory but it stores the last n tokens instead of last n messages.You set a max_token_limit. It flushes the oldest messages only when the total token count exceeds that limit, ensuring you never go over your LLM's context window or budget.  

#### 5. ConversationSummaryBufferMemory
This type of memory is a hybrid form of ConversationBufferMemory and ConversationSummaryMemory. It keeps a buffer of recent messages in their raw form, but instead of just deleting old ones, it summarizes them into a rolling narrative. You get the accuracy of recent history plus the long-term context of the older parts of the conversation.  

#### 6. VectorStoreRetrieverMemory
This type of memory stores the conversation history in a Vector Database. When a user asks a question, LangChain searches the database for the most semantically relevant past messages and injects them into the prompt. It’s great for "long-term memory" where you need to recall specific facts from weeks ago.  

#### 7. EntityMemory  
This type of memory focuses on extracting and remembering specific facts about entities (people, places, objects). It builds a profile for each entity mentioned.It saves the detail to an "Entity Store" and retrieves it when needed.
#### 8. ConversationKGMemory
This type of memory is used to store the conversation history in a Knowledge Graph. It identifies triplets (Subject -> Predicate -> Object) from the chat. It doesn't just remember what was said; it understands the relationships between things, allowing the model to make more complex inferences based on history.
 