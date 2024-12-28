import streamlit as st
from chat_handler import UiChatHandler
from agents.main_agent import ChatAgent


st.set_page_config(page_title="Movies Chatbot 📀🤖📀",
                   page_icon="📀", layout='wide')


with st.sidebar:
    st.title("Movies Chatbot 📀🤖📀")
    st.write("""
            Receive personalized movie recommendations based on your mood, 
            with user reviews highlighting key opinions for each movie. 
            Share your feelings, 
            and get curated suggestions with insightful feedback from others. :blue[ Start exploring now!]""")

    st.text_input(label="Groq API Key", key='groq_api_key',
                  type='password',)

    st.write("Get Groq API Key  From [Here](https://console.groq.com/keys)")

    st.selectbox("Main LLM", options=[
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",

        "llama3-8b-8192",
        "llama3-70b-8192",
    ], key='model_name')

empty_api_key = st.session_state.groq_api_key in [None, ""]


st.title("Movies Chatbot 📀🤖📀")
st.write("""
         Receive personalized movie recommendations based on your mood, 
         with user reviews highlighting key opinions for each movie. 
         Share your feelings, 
         and get curated suggestions with insightful feedback from others. :blue[ Start exploring now!]""")


if "chat_handler" not in st.session_state:
    st.session_state.chat_handler = UiChatHandler()

if "agent" not in st.session_state:
    with st.spinner("Loading Agents..."):
        st.session_state.agent = ChatAgent()


chat_handler = st.session_state.chat_handler
agent = st.session_state.agent


chat_handler.render_chat_history()


controller = st.empty()


user_input = st.chat_input(
    placeholder="Enter Prompt" if not empty_api_key else "You Should Get Groq API Key First",
    disabled=empty_api_key,
)


if user_input:

    controller.empty()

    chat_handler.add_and_render_msg(user_input, role="user")
    response = ""

    try:
        with st.spinner("Thinking..."):
            response = agent.send_and_get_ai_response(
                user_query=user_input,
                api_key=st.session_state.groq_api_key,
                model_name=st.session_state.model_name,
                is_search_in_db=st.session_state.get("is_search_query", False),
                is_enhance_query=st.session_state.get(
                    "is_enhance_query", False),
            )
    except Exception as e:
        st.error(f"Error: {e}")

    chat_handler.add_and_render_msg(response, role="ai")
    st.rerun()


with controller.popover(
        label="Settings",
        icon="⚙️",
        help="Select settings for chat actions",):

    cols = st.columns(2)
    is_search_query = cols[0].checkbox("Search In DB", key="is_search_query")

    cols[1].checkbox(
        "Enhance Input Query", key="is_enhance_query", disabled=not is_search_query, value=is_search_query)
