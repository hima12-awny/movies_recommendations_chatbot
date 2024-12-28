from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


system_prompt_query_expansion = """
You Are Chatbot assistant that recommend Movies based on user inputs and feels.
You are a friendly assistant responsible for making queries to search the database based on user input. 
Your goal is to focus only on the user's needs and create efficient queries that ignore irrelevant details.
When a user provides input, extract the key information and craft a query that directly addresses their request.
Return the query in normal string format, without any additional explanation or formatting. 
Avoid including unnecessary data or topics that are not relevant to the user’s needs.
Always strive to generate the most accurate and focused query possible to retrieve the best results.
And Make this Query long to make Good Search "MAX 100 Words only".
And if user Tells that want another thing what you give before, make different query to get different results.
"""


class QueryExpansionAgent:
    def __init__(self,
                 system_prompt=system_prompt_query_expansion
                 ) -> None:

        self.chat_history: list[BaseMessage] = [
            SystemMessage(system_prompt)
        ]

    def expand_query(self, user_query: str, model_name: str, api_key: str) -> str:

        self.llm = ChatGroq(model=model_name,
                            api_key=api_key,
                            max_tokens=100)  # type:ignore

        self.chat_history.append(HumanMessage(user_query))
        response = self.llm.invoke(self.chat_history)
        self.chat_history.append(response)
        return response.content  # type: ignore

    def add_to_history(self, msg: BaseMessage):
        self.chat_history.append(msg)
