import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo Groq client
groq_api_key = os.getenv('GROQ_API_KEY')
chat = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Template để đảm bảo câu trả lời bằng tiếng Việt
PROMPT_TEMPLATE = """
Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau bằng tiếng Việt một cách rõ ràng, chính xác và tự nhiên:
Câu hỏi: {question}
"""

# Giao diện Streamlit
st.title("🤖 Chatbot Grok")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý câu trả lời từ Groq
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            try:
                # Tạo prompt với yêu cầu trả lời bằng tiếng Việt
                prompt_template = PromptTemplate(
                    template=PROMPT_TEMPLATE,
                    input_variables=["question"]
                )
                chain = prompt_template | chat
                response = chain.invoke({"question": prompt})
                answer = response.content
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Lưu câu trả lời vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")