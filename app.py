import os
import streamlit as st
# Thay đổi: Import ChatGoogleGenerativeAI thay vì ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate # Thêm import này cho Gemini

load_dotenv()

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(page_title="🤖 Chatbot AI", layout="wide")

# --- Khởi tạo GenAI client ---
genai_api_key = os.getenv('GOOGLE_API_KEY')
if not genai_api_key:
    st.error("GOOGLE_API_KEY không được tìm thấy trong file .env. Vui lòng thêm khóa API của Gemini.")
    st.stop()

genai.configure(api_key=genai_api_key)

chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=genai_api_key,
    convert_system_message_to_human=True
)

# Template chính cho RAG (sử dụng cả context và input)
# Điều chỉnh prompt template một chút cho phù hợp với cách Gemini xử lý system prompt hơn
PROMPT_TEMPLATE = """
Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi sau bằng tiếng Việt một cách rõ ràng và chính xác.
Sử dụng các đoạn ngữ cảnh được cung cấp để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố bịa ra câu trả lời.

Ngữ cảnh:
{context}
"""
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_TEMPLATE),
    ("human", "{input}")
])


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
if 'initial_faiss_load_attempted' not in st.session_state:
    st.session_state.initial_faiss_load_attempted = False


# --- ĐẶT ĐƯỜNG DẪN CỤC BỘ ---
current_script_directory = os.path.dirname(os.path.abspath(__file__))
local_embedding_model_path = os.path.join(current_script_directory, "local_models", "multilingual-e5-large")
REMOTE_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# REMOTE_EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"


# --- Hàm tải Embedding Model (CHỈ HÀM NÀY DÙNG @st.cache_resource) ---
@st.cache_resource
def get_huggingface_embeddings_model(model_path: str, is_local: bool):
    try:
        # Kiểm tra sự tồn tại của model cục bộ nếu is_local là True
        if is_local and not (os.path.exists(model_path) and
                             (os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or
                              os.path.exists(os.path.join(model_path, 'model.safetensors')) or
                              os.path.exists(os.path.join(model_path, 'config.json')))):
            return None, "load_error:Local model path provided but no valid model files found."

        embed_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'} # Có thể đổi thành 'cuda' nếu có GPU
        )
        return embed_model, "loaded_successfully"
    except Exception as e:
        return None, f"load_error:{e}"

# --- Logic tải mô hình Embedding khi khởi động ứng dụng ---
if "embeddings_object" not in st.session_state or st.session_state.embeddings_object is None:
    # 1. Thử tải từ cục bộ trước
    with st.spinner(f"Đang kiểm tra và tải mô hình Embedding từ cục bộ ({local_embedding_model_path})..."):
        embed_model_result, status = get_huggingface_embeddings_model(local_embedding_model_path, is_local=True)

    if embed_model_result:
        st.session_state.embeddings_object = embed_model_result
        if not st.session_state.initial_embed_toast_shown:
            st.toast(f"Mô hình Embedding đã được tải thành công từ cục bộ!", icon="✅")
            st.session_state.initial_embed_toast_shown = True
    else:
        # 2. Thử tải từ Internet nếu cục bộ không thành công
        with st.spinner(f"Không tìm thấy mô hình cục bộ. Đang thử tải từ Internet ({REMOTE_EMBEDDING_MODEL_NAME})..."):
            st.toast(f"Mô hình Embedding cục bộ không tìm thấy. Đang thử tải từ Internet ({REMOTE_EMBEDDING_MODEL_NAME})...", icon="🌐")
            embed_model_result, status = get_huggingface_embeddings_model(REMOTE_EMBEDDING_MODEL_NAME, is_local=False)

        if embed_model_result:
            st.session_state.embeddings_object = embed_model_result
            if not st.session_state.initial_embed_toast_shown:
                st.toast(f"Mô hình Embedding đã được tải thành công từ Internet!", icon="✅")
                st.session_state.initial_embed_toast_shown = True
        else:
            # Nếu cả hai cách đều lỗi
            if not st.session_state.initial_embed_error_toast_shown:
                error_message = status.split(":", 1)[1] if ":" in status else status
                st.error(f"Lỗi khi tải mô hình Embedding: {error_message}. Vui lòng kiểm tra kết nối Internet hoặc đường dẫn mô hình.")
                st.session_state.initial_embed_error_toast_shown = True
            st.stop()

embeddings = st.session_state.embeddings_object # Lấy đối tượng embeddings đã được tải và cache

# --- Cấu hình đường dẫn lưu FAISS index ---
FAISS_PATH = "faiss_index_data_multilingual"
if not os.path.exists(FAISS_PATH):
    os.makedirs(FAISS_PATH)

# --- Quản lý FAISS Vector Store trong session state ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False


# --- Hàm tải FAISS index (KHÔNG DÙNG @st.cache_resource) ---
# Hàm này chỉ chứa logic tải/kiểm tra, không có lệnh Streamlit UI
def load_faiss_index_from_disk(path, embeddings_obj):
    if embeddings_obj is None: # Đảm bảo embeddings đã sẵn sàng
        return None, "embeddings_not_ready"

    if os.path.exists(path) and os.listdir(path):
        try:
            vector_store = FAISS.load_local(
                path,
                embeddings_obj, # Sử dụng đối tượng embeddings đã được tải và cache
                allow_dangerous_deserialization=True
            )
            return vector_store, "loaded_successfully"
        except Exception as e:
            return None, f"load_error:{e}"
    else:
        return None, "not_found"


# --- Logic tải FAISS ban đầu (chỉ chạy một lần sau khi embeddings có sẵn) ---
# Chỉ cố gắng tải FAISS từ đĩa nếu nó chưa được tải vào session_state VÀ chưa từng thử tải
if st.session_state.vector_store is None and not st.session_state.initial_faiss_load_attempted:
    with st.spinner(f"Đang tải dữ liệu AI từ FAISS index đã lưu trong '{FAISS_PATH}'..."):
        st.session_state.vector_store, status = load_faiss_index_from_disk(FAISS_PATH, embeddings) # Truyền embeddings đã được cache

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
        # Điều này sẽ không xảy ra nếu logic tải embeddings chạy trước
        pass
    st.session_state.initial_faiss_load_attempted = True # Đánh dấu là đã thử tải lần đầu


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
    # Thay đổi tiêu đề chatbot từ Grok sang Gemini
    st.title("🤖 Chatbot Gemini")
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
                    # Sử dụng rag_prompt (ChatPromptTemplate) đã định nghĩa ở trên
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
                    # Sử dụng ChatPromptTemplate cho fallback cũng vậy
                    default_prompt = ChatPromptTemplate.from_messages([
                        ("system", fallback_prompt_template),
                        ("human", "{input}")
                    ])
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
            # Tạo một thư mục tạm thời để lưu file
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)

            with st.spinner("Đang xử lý tài liệu... Vui lòng chờ trong giây lát."):
                try:
                    file_extension = uploaded_file.name.split(".")[-1].lower()

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
                        shutil.rmtree(temp_dir)
                        st.stop()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(docs)

                    # Luôn tạo mới hoặc ghi đè FAISS index khi tải file mới
                    st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                    st.session_state.vector_store.save_local(FAISS_PATH)

                    st.success(f"Tài liệu đã được xử lý và FAISS index đã được lưu vào thư mục '{FAISS_PATH}'.")

                    # Reset các cờ trạng thái để thông báo FAISS có thể hiển thị lại nếu cần
                    st.session_state.initial_faiss_loaded_toast_shown = False
                    st.session_state.initial_faiss_not_found_toast_shown = False
                    st.session_state.initial_faiss_error_toast_shown = False
                    st.session_state.initial_faiss_load_attempted = False # Buộc tải lại FAISS từ đĩa nếu trang được làm mới

                except Exception as e:
                    st.error(f"Lỗi khi xử lý tài liệu: {e}. Vui lòng kiểm tra file hoặc cài đặt thư viện 'unstructured'.")
                finally:
                    # Dọn dẹp: Xóa thư mục tạm thời
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)


    st.markdown("---")
    st.subheader("Trạng thái dữ liệu hiện tại:")
    if st.session_state.vector_store:
        st.write("✅ Dữ liệu riêng tư đã được tải và sẵn sàng sử dụng trong Chatbot.")
    else:
        st.write("❌ Chưa có dữ liệu riêng tư nào được tải. Vui lòng tải lên tài liệu.")

    if st.button("Xóa FAISS Index hiện có"):
        if os.path.exists(FAISS_PATH):
            import shutil
            try:
                shutil.rmtree(FAISS_PATH)
                st.session_state.vector_store = None
                # Reset tất cả các cờ trạng thái liên quan đến FAISS để thông báo hiển thị lại
                st.session_state.initial_faiss_loaded_toast_shown = False
                st.session_state.initial_faiss_not_found_toast_shown = False
                st.session_state.initial_faiss_error_toast_shown = False
                st.session_state.initial_faiss_load_attempted = False # Buộc tải lại FAISS từ đĩa lần sau
                st.success("FAISS index đã được xóa thành công. Vui lòng tải lại trang hoặc tải lên tài liệu mới để tạo lại.")
                st.rerun() # Yêu cầu Streamlit chạy lại app để cập nhật trạng thái
            except Exception as e:
                st.error(f"Lỗi khi xóa FAISS index: {e}")
        else:
            st.info("Không tìm thấy FAISS index để xóa.")