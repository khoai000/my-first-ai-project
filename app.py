import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# --- LỆNH st.set_page_config() PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN TRONG SCRIPT! ---
st.set_page_config(page_title="🤖 Chatbot AI", layout="wide")


# --- Khởi tạo Groq client ---
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY không được tìm thấy trong file .env. Vui lòng thêm khóa API của Groq.")
    st.stop()

chat = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192"
)

# Template chính cho RAG (sử dụng cả context và input)
PROMPT_TEMPLATE = """
Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau bằng tiếng Việt một cách rõ ràng, chính xác và tự nhiên.
Sử dụng các đoạn ngữ cảnh được cung cấp để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

Ngữ cảnh:
{context}

Câu hỏi: {input}
"""

# --- Khởi tạo Session State Flags cho thông báo ---
if 'initial_embed_toast_shown' not in st.session_state:
    st.session_state.initial_embed_toast_shown = False
if 'initial_embed_error_toast_shown' not in st.session_state:
    st.session_state.initial_embed_error_toast_shown = False
if 'initial_faiss_loaded_toast_shown' not in st.session_state:
    st.session_state.initial_faiss_loaded_toast_shown = False
if 'initial_faiss_not_found_toast_shown' not in st.session_state:
    st.session_state.initial_faiss_not_found_toast_shown = False
if 'initial_faiss_error_toast_shown' not in st.session_state:
    st.session_state.initial_faiss_error_toast_shown = False


# --- ĐẶT ĐƯỜNG DẪN CỤC BỘ MỘT CÁCH VỮNG CHẮC Ở ĐÂY ---
current_script_directory = os.path.dirname(os.path.abspath(__file__))
local_embedding_model_path = os.path.join(current_script_directory, "local_models", "multilingual-e5-large")


# --- Hàm tải Embedding Model (chỉ chứa logic tính toán, không có lệnh Streamlit UI) ---
@st.cache_resource
def _get_huggingface_embeddings_pure(local_model_path: str):
    try:
        embed_model = HuggingFaceEmbeddings(
            model_name=local_model_path,
            model_kwargs={'device': 'cpu'}
        )
        return embed_model, "loaded_successfully"
    except Exception as e:
        return None, f"load_error:{e}"

# --- Logic gọi hàm tải Embedding và xử lý UI dựa trên kết quả ---
if "embeddings_object" not in st.session_state:
    st.session_state.embeddings_object = None

if st.session_state.embeddings_object is None:
    with st.spinner(f"Đang tải mô hình Embedding từ '{os.path.basename(local_embedding_model_path)}'..."):
        embed_model_result, status = _get_huggingface_embeddings_pure(local_embedding_model_path)

    st.session_state.embeddings_object = embed_model_result

    if status == "loaded_successfully":
        if not st.session_state.initial_embed_toast_shown:
            st.toast(f"Mô hình Embedding từ '{os.path.basename(local_embedding_model_path)}' đã được tải thành công!", icon="✅")
            st.session_state.initial_embed_toast_shown = True
    elif status.startswith("load_error"):
        error_message = status.split(":", 1)[1]
        if not st.session_state.initial_embed_error_toast_shown:
            st.error(f"Lỗi khi khởi tạo HuggingFaceEmbeddings từ đường dẫn cục bộ '{local_embedding_model_path}': {error_message}. Vui lòng kiểm tra xem mô hình đã được tải xuống đầy đủ và đúng vị trí chưa.")
            st.session_state.initial_embed_error_toast_shown = True

embeddings = st.session_state.embeddings_object


# --- Cấu hình đường dẫn lưu FAISS index ---
FAISS_PATH = "faiss_index_data_multilingual"
if not os.path.exists(FAISS_PATH):
    os.makedirs(FAISS_PATH)

# --- Quản lý FAISS Vector Store trong session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False


# --- Hàm tải FAISS index (chỉ chứa logic tính toán, không có lệnh Streamlit UI) ---
@st.cache_resource(hash_funcs={HuggingFaceEmbeddings: lambda _: None})
def get_faiss_index_pure(current_embeddings, path):
    """
    Hàm này chỉ thực hiện việc tải FAISS index vào bộ nhớ.
    Tuyệt đối KHÔNG chứa các lệnh Streamlit UI như st.spinner, st.toast, st.error.
    """
    if current_embeddings is None:
        return None, "embeddings_not_ready"

    if os.path.exists(path) and os.listdir(path):
        try:
            vector_store = FAISS.load_local(
                path,
                current_embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store, "loaded_successfully"
        except Exception as e:
            return None, f"load_error:{e}"
    else:
        return None, "not_found"


# --- Logic gọi hàm tải FAISS và xử lý UI dựa trên kết quả ---
if st.session_state.vector_store is None:
    with st.spinner(f"Đang tải dữ liệu AI từ FAISS index đã lưu trong '{FAISS_PATH}'..."):
        st.session_state.vector_store, status = get_faiss_index_pure(embeddings, FAISS_PATH)

    if status == "loaded_successfully":
        if not st.session_state.initial_faiss_loaded_toast_shown:
            st.toast(f"FAISS index đã được tải thành công từ thư mục '{FAISS_PATH}'.", icon="✅")
            st.session_state.initial_faiss_loaded_toast_shown = True
        st.session_state.processing_done = True
    elif status.startswith("load_error"):
        error_message = status.split(":", 1)[1]
        if not st.session_state.initial_faiss_error_toast_shown:
            st.error(f"Lỗi khi tải FAISS index từ cục bộ: {error_message}. Vui lòng thử tải lại tài liệu.")
            st.session_state.initial_faiss_error_toast_shown = True
        st.session_state.processing_done = False
        st.session_state.vector_store = None
    elif status == "not_found":
        if not st.session_state.initial_faiss_not_found_toast_shown:
            st.toast(f"Chưa tìm thấy FAISS index trong thư mục '{FAISS_PATH}'. Vui lòng tải lên tài liệu.", icon="ℹ️")
            st.session_state.initial_faiss_not_found_toast_shown = True
        st.session_state.processing_done = False
        st.session_state.vector_store = None
    elif status == "embeddings_not_ready":
        pass


# --- Menu điều hướng dọc ở Sidebar ---
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Chatbot", "Quản lý dữ liệu"],
        icons=["chat", "file-earmark-text"],
        menu_icon="list",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "orange", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2ca8f2"},
        }
    )

# --- Logic hiển thị nội dung dựa trên lựa chọn menu ---
if selected == "Chatbot":
    st.title("🤖 Chatbot Grok")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Đang xử lý..."):
            try:
                if st.session_state.vector_store: # Kích hoạt RAG nếu có vector_store
                    retriever = st.session_state.vector_store.as_retriever()
                    rag_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
                    combine_docs_chain = create_stuff_documents_chain(chat, rag_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response["answer"]

                else:
                    # Fallback nếu không có vector_store (chatbot tổng quát)
                    st.warning("Không tìm thấy dữ liệu cá nhân. Chatbot sẽ trả lời dựa trên kiến thức chung.")
                    # Template đơn giản hơn, chỉ cần "input" khi không có RAG
                    fallback_prompt_template = """
Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau bằng tiếng Việt một cách rõ ràng, chính xác và tự nhiên:
Câu hỏi: {input}
"""
                    default_prompt = PromptTemplate(
                        template=fallback_prompt_template, # Sử dụng template riêng cho fallback
                        input_variables=["input"]
                    )
                    chain = default_prompt | chat
                    response = chain.invoke({"input": prompt})
                    answer = response.content

                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")

elif selected == "Quản lý dữ liệu":
    st.header("Trang Quản lý Dữ liệu")
    st.write("Tại đây, bạn có thể tải lên tài liệu mới để cập nhật dữ liệu cho chatbot hoặc xem trạng thái dữ liệu hiện có.")

    if embeddings is None:
        st.warning("Mô hình Embedding không khả dụng. Vui lòng kiểm tra các thông báo lỗi ở trên để biết chi tiết.")

    uploaded_file = st.file_uploader(
        "Chọn một file PDF, TXT, DOCX hoặc XLSX để cập nhật dữ liệu",
        type=["pdf", "txt", "docx", "xlsx"],
        accept_multiple_files=False,
        help="Tải lên tài liệu của bạn để chatbot có thể trả lời các câu hỏi dựa trên nội dung đó. Việc tải file mới sẽ ghi đè dữ liệu cũ."
    )

    if uploaded_file:
        if embeddings is None:
            st.error("Không thể xử lý tài liệu vì mô hình Embedding không khả dụng. Vui lòng kiểm tra các thông báo lỗi trên cùng.")
        else:
            with st.spinner("Đang xử lý tài liệu... Vui lòng chờ trong giây lát."):
                try:
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    temp_file_path = f"temp_uploaded_file.{file_extension}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    docs = []
                    if file_extension == "pdf":
                        loader = PyPDFLoader(temp_file_path)
                        docs = loader.load()
                    elif file_extension == "txt":
                        loader = TextLoader(temp_file_path)
                        docs = loader.load()
                    elif file_extension == "docx":
                        loader = UnstructuredWordDocumentLoader(temp_file_path)
                        docs = loader.load()
                    elif file_extension == "xlsx":
                        loader = UnstructuredExcelLoader(temp_file_path)
                        docs = loader.load()
                    else:
                        st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên file PDF, TXT, DOCX, hoặc XLSX.")
                        os.remove(temp_file_path)
                        st.stop()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(docs)

                    st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                    st.session_state.vector_store.save_local(FAISS_PATH)

                    st.success(f"Tài liệu đã được xử lý và FAISS index đã được lưu vào thư mục '{FAISS_PATH}'.")

                    # Sau khi tạo/cập nhật FAISS, xóa cache cho hàm tải FAISS
                    get_faiss_index_pure.clear()
                    st.session_state.initial_faiss_loaded_toast_shown = False
                    st.session_state.initial_faiss_not_found_toast_shown = False
                    st.session_state.initial_faiss_error_toast_shown = False

                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

                except Exception as e:
                    st.error(f"Lỗi khi xử lý tài liệu: {e}. Vui lòng kiểm tra file hoặc cài đặt thư viện 'unstructured'.")

    st.markdown("---")
    st.subheader("Trạng thái dữ liệu hiện tại:")
    if st.session_state.vector_store:
        st.write("✅ Dữ liệu riêng tư đã được tải và sẵn sàng sử dụng trong Chatbot.")
    else:
        st.write("❌ Chưa có dữ liệu riêng tư nào được tải. Vui lòng tải lên tài liệu.")