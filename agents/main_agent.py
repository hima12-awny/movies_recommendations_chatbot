from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from .query_expansion_agent import QueryExpansionAgent
from .retriever_agent import CustomRetrieverTool
import streamlit as st

from loguru import logger as lg


system_prompt = """
1. **Role and Purpose**  
   - You are a friendly and empathetic AI assistant that recommends movies based only on the chat history provided.  
   - Your role is to match the user’s mood or preferences with thoughtful, engaging responses while using only the movies mentioned in the chat history.  

2. **Movie Recommendations Based on Mood**  
   - Recommend movies from the chat history that align with the user’s emotions or interests.  
   - Use uplifting language and appropriate emojis to create an enjoyable experience for the user.  

3. **Response Structure**  
   - Offer a maximum of **two movie suggestions** at a time.  
   - Each recommendation must include:  
     - A short review or description from the chat history explaining the recommendation.  
     - A **user review snippet** from the chat to further support the recommendation.  

4. **Formatting**  
   - Use **well-structured markdown formatting** to make the response visually appealing, such as:  
     - Sidebars or block quotes for **user review snippets**.  
     - Clear headings, bold text, and lists for key details about the movies.  
     - Emojis to reflect the user’s mood (e.g., 😊 for happiness, 🌟 for excitement).  
     - Divider lines (`---`) to separate multiple recommendations for better readability.  

5. **Empathy and Mood Matching**  
   - Tailor your tone to the user’s mood, offering cheerful, motivational, or comforting responses as needed.  
   - Ensure the language feels personalized and empathetic.

6. **Limitations**  
   - Do not suggest movies not mentioned in the chat history.  
   - If no suitable movies are available in the chat history, kindly let the user know without making external suggestions.  
"""


class ChatAgent:

    def __init__(self,
                 api_key: str) -> None:

        self.api_key = api_key

        self.query_expander_agent = QueryExpansionAgent(api_key=api_key,)

        self.rtrvr_agent = CustomRetrieverTool()

        self.chat_history: list[BaseMessage] = [
            SystemMessage(system_prompt),
        ]

    def add_to_history(self, msg: BaseMessage) -> None:
        self.chat_history.append(msg)

    def send_and_get_ai_response(
        self, user_query: str,
        model_name: str,
        is_search_in_db: bool = False,
        is_enhance_query: bool = False,

    ) -> str:

        self.llm = ChatGroq(model=model_name,
                            api_key=self.api_key)  # type: ignore

        self.add_to_history(HumanMessage(user_query))

        if is_search_in_db:
            lg.debug("Searching For Items in DB...")

            final_query = user_query

            if is_enhance_query:
                with st.spinner("Improve Query..."):
                    final_query = self.query_expander_agent.expand_query(
                        user_query,
                        model_name=model_name)

                lg.debug(f"Choosing to Retrieve Movies With\n\
                    \rUser Query: {user_query}\n\
                    \rExpanded Query: {final_query}")

            with st.spinner("Retrieve Movies..."):

                retrieved_movies = self.rtrvr_agent.retrieve_movies(
                    query=final_query)

            lg.debug(f"Movies Retrieved: \n{retrieved_movies}")

            self.add_to_history(SystemMessage(retrieved_movies))

            with st.spinner("Refining The Results..."):
                response = self.llm.invoke(self.chat_history)

        else:
            lg.debug("No Need to Search in DB.")
            response = self.llm.invoke(self.chat_history)

            self.query_expander_agent.add_to_history(HumanMessage(user_query))
            self.query_expander_agent.add_to_history(response)

        self.add_to_history(response)

        return str(response.content)
