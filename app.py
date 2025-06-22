import os
import streamlit as st
# Thay đổi: Import ChatGoogleGenerativeAI thay vì ChatGroq
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader
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

# Cờ cho Base Document FAISS (MỚI)
if 'initial_base_faiss_loaded_toast_shown' not in st.session_state:
    st.session_state.initial_base_faiss_loaded_toast_shown = False
if 'initial_base_faiss_not_found_toast_shown' not in st.session_state:
    st.session_state.initial_base_faiss_not_found_toast_shown = False
if 'initial_base_faiss_error_toast_shown' not in st.session_state:
    st.session_state.initial_base_faiss_error_toast_shown = False
if 'initial_base_faiss_load_attempted' not in st.session_state:
    st.session_state.initial_base_faiss_load_attempted = False
# Cờ cho Evaluation Document FAISS (MỚI)
if 'initial_eval_faiss_loaded_toast_shown' not in st.session_state:
    st.session_state.initial_eval_faiss_loaded_toast_shown = False
if 'initial_eval_faiss_not_found_toast_shown' not in st.session_state:
    st.session_state.initial_eval_faiss_not_found_toast_shown = False
if 'initial_eval_faiss_error_toast_shown' not in st.session_state:
    st.session_state.initial_eval_faiss_error_toast_shown = False
if 'initial_eval_faiss_load_attempted' not in st.session_state:
    st.session_state.initial_eval_faiss_load_attempted = False

if "check_button_pressed" not in st.session_state:
    st.session_state.check_button_pressed = False


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

embeddings = st.session_state.embeddings_object

BASE_FAISS_PATH = "faiss_index_base_docs" 
EVAL_FAISS_PATH = "faiss_index_eval_docs"

if not os.path.exists(BASE_FAISS_PATH):
    os.makedirs(BASE_FAISS_PATH)
if not os.path.exists(EVAL_FAISS_PATH): # MỚI
    os.makedirs(EVAL_FAISS_PATH)

if "base_vector_store" not in st.session_state: 
    st.session_state.base_vector_store = None
if "eval_vector_store" not in st.session_state: 
    st.session_state.eval_vector_store = None


# Hàm này chỉ chứa logic tải/kiểm tra, không có lệnh Streamlit UI
def load_faiss_index_from_disk(path, embeddings_obj):
    if embeddings_obj is None:
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

# --- Hàm xử lý file upload ---
def process_uploaded_file(uploaded_file, faiss_path, vector_store_key, doc_type_name):
    if embeddings is None:
        st.error("Không thể xử lý tài liệu vì mô hình Embedding không khả dụng. Vui lòng kiểm tra các thông báo lỗi trên cùng.")
        return False

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
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(temp_file_path)
                docs = loader.load()
            elif file_extension == "xlsx":
                loader = UnstructuredExcelLoader(temp_file_path)
                docs = loader.load()
            else:
                st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên file PDF, DOCX, hoặc XLSX.")
                shutil.rmtree(temp_dir)
                return False

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            new_vector_store = FAISS.from_documents(splits, embeddings)
            new_vector_store.save_local(faiss_path)
            st.session_state[vector_store_key] = new_vector_store

            st.success(f"Tài liệu '{doc_type_name}' đã được xử lý")

            # Reset các cờ trạng thái FAISS của loại tài liệu này
            if vector_store_key == "base_vector_store":
                st.session_state.initial_base_faiss_loaded_toast_shown = False
                st.session_state.initial_base_faiss_not_found_toast_shown = False
                st.session_state.initial_base_faiss_error_toast_shown = False
                st.session_state.initial_base_faiss_load_attempted = False
            elif vector_store_key == "eval_vector_store":
                st.session_state.initial_eval_faiss_loaded_toast_shown = False
                st.session_state.initial_eval_faiss_not_found_toast_shown = False
                st.session_state.initial_eval_faiss_error_toast_shown = False
                st.session_state.initial_eval_faiss_load_attempted = False
            return True

        except Exception as e:
            st.error(f"Lỗi khi xử lý tài liệu: {e}. Vui lòng kiểm tra file hoặc cài đặt thư viện 'unstructured'.")
            return False
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# --- Logic tải FAISS ban đầu cho Base Document ---
if st.session_state.base_vector_store is None and not st.session_state.initial_base_faiss_load_attempted:
    with st.spinner(f"Đang tải dữ liệu tài liệu cơ sở từ FAISS index '{BASE_FAISS_PATH}'..."): 
        st.session_state.base_vector_store, status = load_faiss_index_from_disk(BASE_FAISS_PATH, embeddings)

    if status == "loaded_successfully":
        if not st.session_state.initial_base_faiss_loaded_toast_shown: 
            # st.toast(f"FAISS index tài liệu cơ sở đã được tải thành công từ '{BASE_FAISS_PATH}'.", icon="✅")
            st.session_state.initial_base_faiss_loaded_toast_shown = True
    elif status.startswith("load_error"):
        error_message = status.split(":", 1)[1]
        if not st.session_state.initial_base_faiss_error_toast_shown: 
            st.error(f"Lỗi khi tải FAISS index tài liệu cơ sở: {error_message}. Vui lòng thử tải lại tài liệu.")
            st.session_state.initial_base_faiss_error_toast_shown = True
        st.session_state.base_vector_store = None
    elif status == "not_found":
        if not st.session_state.initial_base_faiss_not_found_toast_shown:
            # st.toast(f"Chưa tìm thấy FAISS index tài liệu cơ sở trong thư mục '{BASE_FAISS_PATH}'. Vui lòng tải lên tài liệu.", icon="ℹ️")
            st.session_state.initial_base_faiss_not_found_toast_shown = True
        st.session_state.base_vector_store = None
    elif status == "embeddings_not_ready":
        pass
    st.session_state.initial_base_faiss_load_attempted = True

# --- Logic tải FAISS ban đầu cho Evaluation Document ---
if st.session_state.eval_vector_store is None and not st.session_state.initial_eval_faiss_load_attempted:
    with st.spinner(f"Đang tải dữ liệu tài liệu đánh giá từ FAISS index '{EVAL_FAISS_PATH}'..."):
        st.session_state.eval_vector_store, status = load_faiss_index_from_disk(EVAL_FAISS_PATH, embeddings)

    if status == "loaded_successfully":
        if not st.session_state.initial_eval_faiss_loaded_toast_shown:
            # st.toast(f"FAISS index tài liệu đánh giá đã được tải thành công từ '{EVAL_FAISS_PATH}'.", icon="✅")
            st.session_state.initial_eval_faiss_loaded_toast_shown = True
    elif status.startswith("load_error"):
        error_message = status.split(":", 1)[1]
        if not st.session_state.initial_eval_faiss_error_toast_shown:
            st.error(f"Lỗi khi tải FAISS index tài liệu đánh giá: {error_message}. Vui lòng thử tải lại tài liệu.")
            st.session_state.initial_eval_faiss_error_toast_shown = True
        st.session_state.eval_vector_store = None
    elif status == "not_found":
        if not st.session_state.initial_eval_faiss_not_found_toast_shown:
            # st.toast(f"Chưa tìm thấy FAISS index tài liệu đánh giá trong thư mục '{EVAL_FAISS_PATH}'. Vui lòng tải lên tài liệu.", icon="ℹ️")
            st.session_state.initial_eval_faiss_not_found_toast_shown = True
        st.session_state.eval_vector_store = None
    st.session_state.initial_eval_faiss_load_attempted = True

def handle_canchu_check_button(chat_model, base_store, eval_store):
    if not base_store or not eval_store:
        return "Để thực hiện 'kiểm tra căn cứ', vui lòng tải lên cả **Tài liệu Cơ sở** và **Tài liệu Đánh giá**."

    # Prompt giả định cho việc truy xuất khi nhấn nút
    # (Có thể là một chuỗi rỗng hoặc "kiểm tra" tùy cách bạn muốn retriever hoạt động)
    # Vì bạn muốn nó hoạt động giống như khi người dùng nhập "kiểm tra", chúng ta sẽ dùng "kiểm tra" làm prompt để retriever tìm kiếm ngữ cảnh.
    button_prompt_query = "kiểm tra"

    # Truy xuất ngữ cảnh từ cả hai tài liệu
    base_docs = []
    eval_docs = []
    base_context = ""
    eval_context = ""

    if base_store:
        base_retriever = base_store.as_retriever(search_kwargs={"k": 3})
        base_docs = base_retriever.invoke(button_prompt_query)
        base_context = "\n".join([doc.page_content for doc in base_docs])

    if eval_store:
        eval_retriever = eval_store.as_retriever(search_kwargs={"k": 3})
        eval_docs = eval_retriever.invoke(button_prompt_query)
        eval_context = "\n".join([doc.page_content for doc in eval_docs])

    # Prompt tùy chỉnh cho tác vụ "căn cứ"
    final_prompt_template_content = f"""
        Bạn là một trợ lý AI chuyên về trích xuất dữ liệu.
        Nhiệm vụ của bạn là:

        1.  **Xác định các dòng có từ "căn cứ"**: Từ "Ngữ cảnh từ Tài liệu ĐÁNH GIÁ", hãy tìm và trích xuất tất cả các dòng văn bản bắt đầu bằng từ "căn cứ".
        2.  **Tìm kiếm trong Tài liệu Cơ sở**: Với mỗi dòng "căn cứ" đã trích xuất từ Tài liệu Đánh giá, hãy sử dụng nội dung của dòng đó làm truy vấn để tìm kiếm thông tin liên quan và khớp trong "Ngữ cảnh từ Tài liệu CƠ SỞ". Hãy tìm các đoạn văn bản có nội dung tương đồng hoặc liên quan chặt chẽ (khoảng 90% độ khớp về ý nghĩa) với dòng "căn cứ" đó, không cần phải khớp chính xác từng từ.
        3.  **Định dạng kết quả dạng bảng**: Trả về thông tin tìm được từ "Ngữ cảnh từ Tài liệu CƠ SỞ" dưới dạng một bảng duy nhất.
            * **Quan trọng:** Bảng này phải chứa **tất cả các cột dữ liệu có thể nhận diện được** từ Tài liệu Cơ sở liên quan đến thông tin tìm được. Nếu Tài liệu Cơ sở của bạn có cấu trúc giống bảng hoặc dữ liệu có thể được phân loại thành các cột (ví dụ: "Tiêu đề", "Nội dung", "Ngày", "Mã", "Mô tả", v.v.), hãy sử dụng chúng làm tên cột. Nếu không, hãy sử dụng các tiêu đề cột hợp lý nhất để trình bày thông tin một cách có cấu trúc.
            * Mỗi hàng trong bảng phải là một mục dữ liệu khớp được tìm thấy trong Tài liệu Cơ sở.

        Câu hỏi của người dùng: {button_prompt_query}

        ---
        Ngữ cảnh từ Tài liệu CƠ SỞ:
        {base_context}
        ---
        Ngữ cảnh từ Tài liệu ĐÁNH GIÁ:
        {eval_context}
        ---

        Nếu không có thông tin liên quan để tạo bảng hoặc không thể định dạng thành bảng, hãy nói rằng bạn không tìm thấy kết quả phù hợp trong tài liệu, đừng cố bịa ra câu trả lời. Trả lời trực tiếp bằng bảng (hoặc thông báo không tìm thấy) mà không có bất kỳ lời dẫn hay kết luận nào.
        """

    messages_for_llm = ChatPromptTemplate.from_messages([
        ("system", final_prompt_template_content),
        ("human", button_prompt_query)
    ])

    chain = messages_for_llm | chat_model
    try:
        response = chain.invoke({"input": button_prompt_query})
        return response.content
    except Exception as e:
        return f"Có lỗi xảy ra khi tạo câu trả lời: {str(e)}"

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
    st.title("🤖 Chatbot")
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
                answer = ""
                retrieved_docs = []
                base_context = ""
                eval_context = ""

                if st.session_state.base_vector_store:
                    base_retriever = st.session_state.base_vector_store.as_retriever(search_kwargs={"k": 3})
                    base_docs = base_retriever.invoke(prompt)
                    retrieved_docs.extend(base_docs)
                    base_context = "\n".join([doc.page_content for doc in base_docs])

                if st.session_state.eval_vector_store:
                    eval_retriever = st.session_state.eval_vector_store.as_retriever(search_kwargs={"k": 3})
                    eval_docs = eval_retriever.invoke(prompt)
                    retrieved_docs.extend(eval_docs)
                    eval_context = "\n".join([doc.page_content for doc in eval_docs])
                

                if prompt.lower() == "kiểm tra":
                    if st.session_state.base_vector_store and st.session_state.eval_vector_store:
                        final_prompt_template_content = f"""
                            Bạn là một trợ lý AI chuyên về trích xuất dữ liệu.
                            Nhiệm vụ của bạn là:

                            1.  **Xác định các dòng có từ "căn cứ"**: Từ "Ngữ cảnh từ Tài liệu ĐÁNH GIÁ", hãy tìm và trích xuất tất cả các dòng văn bản bắt đầu bằng từ "căn cứ".
                            2.  **Tìm kiếm trong Tài liệu Cơ sở**: Với mỗi dòng "căn cứ" đã trích xuất từ Tài liệu Đánh giá, hãy sử dụng nội dung của dòng đó làm truy vấn để tìm kiếm thông tin liên quan và khớp trong "Ngữ cảnh từ Tài liệu CƠ SỞ". Hãy tìm các đoạn văn bản có nội dung tương đồng hoặc liên quan chặt chẽ (khoảng 90% độ khớp về ý nghĩa) với dòng "căn cứ" đó, không cần phải khớp chính xác từng từ.
                            3.  **Định dạng kết quả dạng bảng**: Trả về thông tin tìm được từ "Ngữ cảnh từ Tài liệu CƠ SỞ" dưới dạng một bảng duy nhất.
                                * **Quan trọng:** Bảng này phải chứa **tất cả các cột dữ liệu có thể nhận diện được** từ Tài liệu Cơ sở liên quan đến thông tin tìm được. Nếu Tài liệu Cơ sở của bạn có cấu trúc giống bảng hoặc dữ liệu có thể được phân loại thành các cột (ví dụ: "Tiêu đề", "Nội dung", "Ngày", "Mã", "Mô tả", v.v.), hãy sử dụng chúng làm tên cột. Nếu không, hãy sử dụng các tiêu đề cột hợp lý nhất để trình bày thông tin một cách có cấu trúc.
                                * Mỗi hàng trong bảng phải là một mục dữ liệu khớp được tìm thấy trong Tài liệu Cơ sở.

                            Câu hỏi của người dùng: {prompt}

                            ---
                            Ngữ cảnh từ Tài liệu CƠ SỞ:
                            {base_context}
                            ---
                            Ngữ cảnh từ Tài liệu ĐÁNH GIÁ:
                            {eval_context}
                            ---

                            Nếu không có thông tin liên quan để tạo bảng hoặc không thể định dạng thành bảng, hãy nói rằng bạn không tìm thấy kết quả phù hợp trong tài liệu, đừng cố bịa ra câu trả lời. Trả lời trực tiếp bằng bảng (hoặc thông báo không tìm thấy) mà không có bất kỳ lời dẫn hay kết luận nào.
                            """

                        messages_for_llm = ChatPromptTemplate.from_messages([
                            ("system", final_prompt_template_content),
                            ("human", prompt)
                        ])

                        chain = messages_for_llm | chat
                        response = chain.invoke({"input": prompt})
                        answer = response.content

                    else:
                        answer = "Để thực hiện 'kiểm tra' theo yêu cầu, vui lòng đảm bảo bạn đã tải lên cả **Tài liệu Cơ sở** và **Tài liệu Đánh giá** trong mục 'Quản lý dữ liệu'."
                        st.warning(answer)
                    st.session_state.check_button_pressed = False

                else:
                    if st.session_state.base_vector_store or st.session_state.eval_vector_store:
                        if not retrieved_docs:
                            answer = "Không tìm được kết quả phù hợp trong tài liệu."
                        else:
                            rag_chain = create_stuff_documents_chain(chat, rag_prompt)
                            response = rag_chain.invoke({"context": retrieved_docs, "input": prompt})
                            if isinstance(response, dict) and "answer" in response:
                                answer = response["answer"]
                            else:
                                answer = response

                            if not answer or answer.strip().lower() == "tôi không biết." or "không tìm thấy thông tin" in answer.lower():
                                answer = "Không tìm được kết quả phù hợp trong tài liệu."
                    else:
                        st.warning("Không tìm thấy dữ liệu tài liệu nào. Chatbot sẽ trả lời dựa trên kiến thức chung.")
                        default_prompt = ChatPromptTemplate.from_messages([
                            ("system", PROMPT_TEMPLATE),
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
    st.header("Tải lên Tài liệu")

    if embeddings is None:
        st.warning("Mô hình Embedding không khả dụng. Vui lòng kiểm tra các thông báo lỗi ở trên để biết chi tiết.")

    st.info("Tải lên file PDF, DOCX hoặc XLSX. Sau đó, chọn loại tài liệu (Cơ sở hoặc Đánh giá).")
    # Thêm Radio Button để chọn loại tài liệu (MỚI)
    document_type = st.radio(
        "Đây là loại tài liệu gì?",
        ("Tài liệu Cơ sở", "Tài liệu cần Kiểm tra"),
        key="doc_type_selector",
        index=1, # Mặc định là "Tài liệu Đánh giá/Kiểm tra"
        help="Chọn 'Tài liệu Cơ sở' cho dữ liệu chính của chatbot, hoặc 'Tài liệu cần Kiểm tra' cho dữ liệu phụ để so sánh."
    )

    uploaded_file = st.file_uploader(
        "Chọn một file để tải lên",
        type=["pdf", "txt", "docx", "xlsx"],
        key="single_file_uploader",
        help="Tải lên tài liệu của bạn. Bạn sẽ chọn đây là tài liệu cơ sở hay tài liệu đánh giá."
    )

    if uploaded_file:
        if document_type == "Tài liệu Cơ sở":
            target_faiss_path = BASE_FAISS_PATH
            target_vector_store_key = "base_vector_store"
            doc_type_name = "Tài liệu Cơ sở"
        else:
            target_faiss_path = EVAL_FAISS_PATH
            target_vector_store_key = "eval_vector_store"
            doc_type_name = "Tài liệu cần Kiểm tra"

        current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{document_type}"

        # Kiểm tra xem file hiện tại có khác với file đã xử lý gần nhất cho loại tài liệu này không
        if st.session_state.get(f'last_processed_file_{target_vector_store_key}') != current_file_id:
            if process_uploaded_file(uploaded_file, target_faiss_path, target_vector_store_key, doc_type_name):
                st.session_state[f'last_processed_file_{target_vector_store_key}'] = current_file_id
                st.rerun()
        else:
            pass

    st.markdown("---")
    st.subheader("Trạng thái dữ liệu hiện tại:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Tài liệu Cơ sở:")
        if st.session_state.base_vector_store:
            st.write("✅ Đã tải và sẵn sàng.")
        else:
            st.write("❌ Chưa tải.")
        if st.button("Xóa Tài liệu Cơ sở", key="delete_base_faiss"):
            if os.path.exists(BASE_FAISS_PATH):
                import shutil
                try:
                    shutil.rmtree(BASE_FAISS_PATH)
                    st.session_state.base_vector_store = None
                    st.session_state.initial_base_faiss_loaded_toast_shown = False
                    st.session_state.initial_base_faiss_not_found_toast_shown = False
                    st.session_state.initial_base_faiss_error_toast_shown = False
                    st.session_state.initial_base_faiss_load_attempted = False
                    st.success("Tài liệu Cơ sở đã được xóa.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa Tài liệu Cơ sở: {e}")
            else:
                st.info("Không tìm thấy Tài liệu Cơ sở để xóa.")

    with col2:
        st.write("#### Tài liệu cần Kiểm tra:")
        if st.session_state.eval_vector_store:
            st.write("✅ Đã tải và sẵn sàng.")
        else:
            st.write("❌ Chưa tải.")
        if st.button("Xóa Tài liệu cần kiểm tra", key="delete_eval_faiss"):
            if os.path.exists(EVAL_FAISS_PATH):
                import shutil
                try:
                    shutil.rmtree(EVAL_FAISS_PATH)
                    st.session_state.eval_vector_store = None
                    st.session_state.initial_eval_faiss_loaded_toast_shown = False
                    st.session_state.initial_eval_faiss_not_found_toast_shown = False
                    st.session_state.initial_eval_faiss_error_toast_shown = False
                    st.session_state.initial_eval_faiss_load_attempted = False
                    st.success("FAISS index Tài liệu cần kiểm tra đã được xóa.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xóa Tài liệu cần kiểm tra: {e}")
            else:
                st.info("Không tìm thấy Tài liệu cần kiểm tra để xóa.")


    st.subheader("Kiểm tra và Trích xuất Dữ liệu")
    st.write("Nhấn nút dưới đây để thực hiện kiểm tra tự động")

    
    if st.button("Thực hiện Kiểm tra"):

        canchu_check_result_placeholder = st.empty()
        with st.spinner("Đang thực hiện kiểm tra"):
            result_for_button = handle_canchu_check_button(
                chat,
                st.session_state.get("base_vector_store"),
                st.session_state.get("eval_vector_store")
            )
    canchu_check_result_placeholder.markdown(result_for_button)