import nest_asyncio
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.tools import ToolMetadata, RetrieverTool
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from langchain_core.tools import tool


class CustomRetrieverTool:
    def __init__(self,
                 embed_model_name='all-MiniLM-L6-v2',
                 PERSIST_DIR="./storage"
                 ) -> None:

        embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
        nest_asyncio.apply()

        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(
            storage_context, embed_model=embed_model,)

        self.retriever = RetrieverTool(
            retriever=index.as_retriever(
                similarity_top_k=10,
                clean_up_tokenization_spaces=True,
            ),
            metadata=ToolMetadata(
                name='Sreach',
                description="This tool searches for the most suitable movies based on user preferences and emotions by analyzing their query. It uses a retriever to find the top 10 relevant movie candidates from the database, followed by a re-ranking step using ColBERT to identify the top 2 most fitting results. Pass the complete user message as the query to ensure accurate recommendations that align with user needs.",
            ),
            node_postprocessors=[ColbertRerank(top_n=2)],
        )

    def retrieve_movies(self, query: str) -> str:
        """
        This tool searches and ranks movies based on user preferences and emotions, which are provided in the user prompt. 
        It analyzes the user's query, retrieves the top 10 relevant movies from the database using a similarity-based retriever, 
        and then re-ranks these results using ColBERT to ensure the top 2 most relevant suggestions are returned. 
        The user prompt should include their feelings, preferences, and what they are looking for in a movie, 
        and the tool will recommend the best fitting movies.

        Parameters:
        - user_prompt (str): The full user query containing their preferences, emotions, and movie attributes they're seeking.

        Returns:
        - most_movies_matched: A RetrieverTool instance that processes the query and returns the most suitable movie recommendations.
        """
        most_movies_matched = self.retriever.call(query).content

        context = f"""
        \rUse This Movies Information That can fit when user needs, it can answer the 
        \rUser prompt: 
        \r"{query}"
        
        \rMovies Information:
        \r{most_movies_matched}
        \rMake the Response Friendly And not in Template.
        """

        return context
