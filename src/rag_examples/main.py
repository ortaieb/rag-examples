"""
  Main call for running the RAG backed LLM
"""
from langchain_core.prompts import ChatPromptTemplate
from vector_store import create_vector_store
from llm_provider import ClaudeLLMProvider

# Handle imports for both script and module execution
try:
    from .config import CONFIG, get_cli_overrides
except ImportError:
    # When running as script, use absolute imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag_examples.config import CONFIG, get_cli_overrides


def create_llm_chain(model_name: str, temperature: float) -> tuple[ClaudeLLMProvider, ChatPromptTemplate]:
    """
    Create LLM chain with immutable configuration.

    Args:
        model_name: Name of the Claude model to use
        temperature: Temperature setting for the model

    Returns:
        Tuple of (model, prompt_template)
    """
    model = ClaudeLLMProvider(model=model_name, temperature=temperature)

    template = """
      You are an exeprt in answering questions about a pizza restaurant.
      Base answers only on reviews provided.

      Here are some relevant reviews: {reviews}

      Here is the question to answer: {question}
      """
    prompt = ChatPromptTemplate.from_template(template)

    return model, prompt


def main():
    """Main entry point for the application."""
    cli_config = get_cli_overrides()

    # Create vector store with CLI arguments
    retriever = create_vector_store(
        db_location=cli_config.store_location,
        embedding_model=cli_config.emb_model,
        k_value=int(cli_config.n_reviews)
    )

    model, prompt = create_llm_chain(cli_config.model, cli_config.temperature)
    # chain = prompt | model  # Remove this, as we will call model.generate directly

    print(f"Using model: {cli_config.model}")
    print(f"Using api_key: {cli_config.anthropic_api_key}")
    print(f"Temperature: {cli_config.temperature}")
    print(f"Store location: {cli_config.store_location}")
    print(f"Number of reviews: {cli_config.n_reviews}")

    while True:
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
        prompt_text = prompt.format(reviews=reviews, question=question)
        result = model.generate(prompt_text)
        print(result)


if __name__ == "__main__":
    main()
