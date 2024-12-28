import streamlit as st
import vars
from datetime import datetime
from time import sleep


class UiChatHandler:

    def __init__(self,) -> None:
        self.chat_history = []

    def get_cur_time(self) -> str:
        return datetime.now().strftime("%H:%M")

    def add_time_to_msg(self, msg, time: str | None = None,) -> str:

        time = self.get_cur_time() if time is None else time
        return f"{msg}<br>{time}"

    def render_msg(
            self,
            msg: str,
            role: str,
            time: str | None = None,
            is_stream: bool = False) -> None:

        avatar = vars.avatar_src_user if role == 'user' else vars.avatar_src_ai
        name = vars.user_name if role == 'user' else vars.ai_name
        final_msg = self.add_time_to_msg(msg, time)

        with st.chat_message(role, avatar=avatar):
            st.markdown(
                f'<span style="font-size:22px; font-weight:bold;">{name}</span>', unsafe_allow_html=True)

            if not is_stream:
                st.markdown(final_msg, unsafe_allow_html=True)
                return

            empty = st.empty()
            ui_msg_content = ''

            for word in final_msg.split(" "):
                empty.markdown((ui_msg_content := ui_msg_content +
                                word + ' '), unsafe_allow_html=True)
                sleep(vars.stream_delay)

    def render_chat_history(self) -> None:

        for msg in self.chat_history:
            if msg['role'] not in ['ai', 'user']:
                continue
            self.render_msg(
                msg=msg['msg'],
                role=msg['role'],
                time=msg["time"],
            )

    def add_to_history(self, msg, role):
        msg = msg.strip()
        time = self.get_cur_time()

        self.chat_history.append(dict(
            msg=msg,
            role=role,
            time=time,
        ))

    def add_and_render_msg(self, msg: str, role: str) -> None:
        msg = msg.strip()
        time = self.get_cur_time()

        self.chat_history.append(dict(
            msg=msg,
            role=role,
            time=time,
        ))

        self.render_msg(msg, role=role, time=time, is_stream=role == 'ai')
