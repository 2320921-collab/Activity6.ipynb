# Activity6.ipynb

import os
import numexpr
import wikipedia
from datetime import datetime
from docx import Document
from typing import List, Optional

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# Document Retrieval Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# 0. Global LLM Initialization

# We initialize the LLM globally so that our custom document creation 
# tools can use it internally to format text before saving.
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio", 
    temperature=0.3,
    model="qwen/qwen3-4b"
)
# 1. Base Utility Functions

def save_to_word(content: str, prefix: str = "Doc") -> str:
    """Helper function to cleanly save text to a .docx file."""
    folder = "generated_docs"
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(folder, f"{prefix}_{timestamp}.docx")
    
    try:
        doc = Document()
        for line in content.split('\n'):
            if line.strip():
                doc.add_paragraph(line.strip())
        doc.save(filepath)
        return f"Document successfully created and saved at: {filepath}"
    except Exception as e:
        return f"Failed to save document: {str(e)}"
# 2. Agent Tools

@tool
def fetch_wiki_data(subject: str) -> str:
    """
    Independent tool to retrieve raw information from Wikipedia.
    Use this when you need to gather data to show the user or to process before saving.
    """
    api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
    try:
        return api.run(subject)
    except Exception as e:
        return f"Wikipedia Retrieval Error: {str(e)}"

@tool
def export_text_to_file(text_content: str, file_label: str = "Export") -> str:
    """
    Independent tool to save ANY provided text into a Word document.
    Use this when the user specifically asks to 'save this' or 'put that result into a file'.
    """
    return save_to_word(text_content, file_label)


@tool
def solve_math(expression: str) -> str:
    """
    Solves arithmetic expressions securely.
    Input must be a valid mathematical equation (e.g., '10 + 1 * 1000').
    """
    try:
        result = numexpr.evaluate(expression)
        return str(result.item())
    except Exception:
        return "Invalid arithmetic expression. Please check your formatting."

@tool
def generate_structured_word_doc(topic_or_instructions: str) -> str:
    """
    Generates a well-structured Word document from a user's request.
    Use this when the user asks to "create a document" or "write an essay".
    """
    formatting_prompt = f"Create a structured, comprehensive document for: {topic_or_instructions}"
    structured_content = llm.invoke(formatting_prompt).content
    return save_to_word(structured_content, "Generated")

@tool
def wikipedia_to_word_doc(search_query: str) -> str:
    """
    Extracts info from Wikipedia and compiles it into a structured Word document.
    Use this when asked to create a document specifically based on Wikipedia.
    """
    try:
        wiki_summary = wikipedia.summary(search_query, sentences=10)
        formatting_prompt = f"Format this Wikipedia info into a structured document:\n\n{wiki_summary}"
        structured_content = llm.invoke(formatting_prompt).content
        return save_to_word(structured_content, "Wiki_Doc")
    except Exception as e:
        return f"Wikipedia Document Error: {str(e)}"

# --- RAG / LOCAL PDF SEARCH LOGIC ---

def setup_local_pdf_search(pdf_filename: str = "short_story.pdf") -> Optional[callable]:
    """Sets up the FAISS vector database for local PDF search."""
    if not os.path.exists(pdf_filename) and not os.path.exists("faiss_index"):
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(pdf_filename)
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(loader.load())
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    @tool
    def search_local_pdf(query: str) -> str:
        """Searches the uploaded local PDF for answers."""
        try:
            found_docs = retriever.invoke(query)
            return "\n\n".join([doc.page_content for doc in found_docs]) if found_docs else "No info found."
        except Exception as e:
            return f"Retrieval error: {str(e)}"
    
    return search_local_pdf
# 3. Agent Configuration & Execution Loop

def build_agent() -> AgentExecutor:
    # Standard Chat Tool
    standard_wiki_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500),
        description="Search Wikipedia for quick facts in the chat."
    )

    tools = [
        solve_math, standard_wiki_tool, 
        fetch_wiki_data, export_text_to_file,  # Modular tools
        generate_structured_word_doc, wikipedia_to_word_doc # Compound tools
    ]
    
    pdf_tool = setup_local_pdf_search()
    if pdf_tool: tools.append(pdf_tool)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent AI assistant. DECISION RULES:\n"
                   "1. Modular Flow: Use 'fetch_wiki_data' to get info, then 'export_text_to_file' to save it.\n"
                   "2. Compound Flow: Use 'generate_structured_word_doc' or 'wikipedia_to_word_doc' for formatted reports.\n"
                   "3. Local Files: Use 'search_local_pdf' for the uploaded PDF context.\n"
                   "4. Math: Use 'solve_math' for calculations."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
if __name__ == "__main__":
    print("=" * 70)
    print(" Advanced Multi-Tool Agent Initialized. Type 'exit' to quit.")
    print("=" * 70)
    
    agent_chain = build_agent()
    chat_memory: List = []
    
    while True:
        try:
            user_query = input("\nType 'exit' to end session: ")
            print("\nYou: " + user_query)

            if user_query.strip().lower() in ['exit', 'quit']:
                print("\nExiting...\nThank you for using the AI assistant. Goodbye!")
                break
                
            if not user_query.strip():
                continue
                
            print("\n> Thinking...")
            
            response = agent_chain.invoke({
                "input": user_query,
                "chat_history": chat_memory
            })
            
            # Save interaction to memory
            chat_memory.append(HumanMessage(content=user_query))
            chat_memory.append(AIMessage(content=response["output"]))
            
            print("-" * 150)
            print("AI Response:")
            print(f">> {response['output']}")
            print("-" * 150)
            
        except KeyboardInterrupt:
            print("\n\nExiting...\nThank you for using the AI assistant. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {str(e)}")
