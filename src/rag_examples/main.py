"""
  Main call for running the RAG backed LLM
"""
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector_store import retriever

model = OllamaLLM(model = "llama3.2")
template = """
  You are an exeprt in answering questions about a pizza restaurant.
  Base answers only on reviews provided.

  Here are some relevant reviews: {reviews}

  Here is the question to answer: {question}
  """

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def main():
    """Main entry point for the application."""
    # print("Hello from my project!  dddd")
    while(True):
        print("\n\n~~~~~~~~~~~~~~~~~")
        question = input("Ask your question (q to quit): ")
        print("\n")
        if question == "q":
            break

        reviews = retriever.invoke(question)
        print("\t=========")
        print("\t  * Using the following reviews:")
        for review in reviews:
          print(f"\t |")
          print(f"\t + {review}")
        print("\t=========")
        result = chain.invoke({"reviews": reviews, "question": question})
        print(result)

if __name__ == "__main__":
    main()
