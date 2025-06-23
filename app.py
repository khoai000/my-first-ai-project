import os
import streamlit as st
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
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from datetime import datetime
import math
import re

load_dotenv()

st.set_page_config(page_title="🤖 Chatbot AI", layout="wide")

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
            model_kwargs={'device': 'cpu'}
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
            if not st.session_state.initial_embed_error_toast_shown:
                error_message = status.split(":", 1)[1] if ":" in status else status
                st.error(f"Lỗi khi tải mô hình Embedding: {error_message}. Vui lòng kiểm tra kết nối Internet hoặc đường dẫn mô hình.")
                st.session_state.initial_embed_error_toast_shown = True
            st.stop()

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

def read_excel_to_array():
    file_path = "./data/1. AI . DM nội dung trình HĐTV từ Ban Kế hoạch (B02) final.xlsx"
    sheet_name = "Danh sách văn bản"
    try:
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)
        data_array = df.values.tolist()
        return data_array
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{file_path}'")
        return None
    except KeyError:
        print(f"Lỗi: Không tìm thấy sheet '{sheet_name}' trong file.")
        return None
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file Excel: {e}")
        return None
    
def handleCheck(llm_model, vector_store):
    data_from_excel = read_excel_to_array()
    if vector_store:
        try:
            retriever = vector_store.as_retriever()
            check_prompt_template = """
                Bạn là một trợ lý AI chuyên nghiệp và tỉ mỉ.

                **Nhiệm vụ:**
                Từ ngữ cảnh được cung cấp bên dưới, hãy **tìm và liệt kê chính xác TẤT CẢ các dòng hoặc đoạn văn bản** mà trong đó xuất hiện từ hoặc cụm từ **'căn cứ'**.

                **Lưu ý quan trọng:**
                * Hãy chú ý đến **ngữ cảnh và cấu trúc câu** để đảm bảo 'căn cứ' được sử dụng đúng nghĩa.
                * **Tránh các lỗi định dạng hoặc ký tự lạ** khi trích xuất. Chỉ trả về phần văn bản gốc, chuẩn xác.
                * Nếu một dòng/đoạn có chứa 'căn cứ' và sau đó lại bị sửa đổi hoặc hủy bỏ bởi một ghi chú, bạn vẫn liệt kê nó nhưng có thể ghi chú thêm nếu thông tin đó rõ ràng trong ngữ cảnh.

                **Định dạng kết quả:**
                Liệt kê mỗi dòng/đoạn tìm được trên một dòng riêng.
                Nếu không tìm thấy bất kỳ dòng/đoạn nào chứa 'căn cứ', hãy trả lời rõ ràng: "Không tìm thấy bất kỳ dữ liệu trong ngữ cảnh được cung cấp."

                **Ngữ cảnh:**
                {context}
                """
            check_rag_prompt = ChatPromptTemplate.from_messages([
                ("system", check_prompt_template),
                ("human", "{input}")
            ])

            combine_docs_chain = create_stuff_documents_chain(llm_model, check_rag_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            response = retrieval_chain.invoke({"input": "Tìm những dòng có từ 'căn cứ' trong văn bản"})

            results = []
            date_pattern = re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})')
            for item in data_from_excel:
                stt = item[0] 
                doc_title = item[1] 
                found_date_str_in_title = None 
                status = item[3]

                if isinstance(doc_title, str):
                    match_obj = date_pattern.search(doc_title)
                    
                    if match_obj:
                        found_date_str_in_title = match_obj.group(0)
                text_matches = isinstance(doc_title, str) and doc_title.lower() in response["answer"].lower()
                date_matches = False
                if found_date_str_in_title:
                    date_matches = found_date_str_in_title in response["answer"] 

                if text_matches or date_matches:
                    new_record = [stt, doc_title, found_date_str_in_title if found_date_str_in_title is not None else "", status if status is not None else ""]
                    results.append(new_record)
            return results
        except Exception as e:
            return f"Lỗi khi thực hiện kiểm tra: {str(e)}"
    else:
        return "Không có dữ liệu tải lên để thực hiện kiểm tra. Vui lòng tải lên tài liệu trước."


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
                if st.session_state.vector_store:
                    retriever = st.session_state.vector_store.as_retriever()
                    # Sử dụng rag_prompt (ChatPromptTemplate) đã định nghĩa ở trên
                    combine_docs_chain = create_stuff_documents_chain(chat, rag_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                    response = retrieval_chain.invoke({"input": prompt})
                    answer = response["answer"]

                else:
                    st.warning("Không tìm thấy dữ liệu cá nhân. Chatbot sẽ trả lời dựa trên kiến thức chung.")
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
        "Chọn một file PDF, DOCX hoặc XLSX để cập nhật dữ liệu",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=False,
        help="Tải lên tài liệu của bạn để chatbot có thể trả lời các câu hỏi dựa trên nội dung đó. Việc tải file mới sẽ ghi đè dữ liệu cũ."
    )

    if 'last_processed_file_info' not in st.session_state:
        st.session_state.last_processed_file_info = None

    if uploaded_file:
        current_file_info = (uploaded_file.name, uploaded_file.size)
        if current_file_info != st.session_state.last_processed_file_info:
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
                        elif file_extension == "docx":
                            loader = UnstructuredWordDocumentLoader(temp_file_path)
                            docs = loader.load()
                        elif file_extension == "xlsx":
                            loader = UnstructuredExcelLoader(temp_file_path)
                            docs = loader.load()
                        else:
                            st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên file PDF, DOCX, hoặc XLSX.")
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

                        st.success(f"Tài liệu đã được xử lý.")
                        st.session_state.last_processed_file_info = current_file_info

                        # Reset các cờ trạng thái để thông báo FAISS có thể hiển thị lại nếu cần
                        st.session_state.initial_faiss_loaded_toast_shown = False
                        st.session_state.initial_faiss_not_found_toast_shown = False
                        st.session_state.initial_faiss_error_toast_shown = False
                        st.session_state.initial_faiss_load_attempted = False # Buộc tải lại FAISS từ đĩa nếu trang được làm mới

                    except Exception as e:
                        st.error(f"Lỗi khi xử lý tài liệu: {e}. Vui lòng kiểm tra file hoặc cài đặt!")
                    finally:
                        # Dọn dẹp: Xóa thư mục tạm thời
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
        else:
            st.session_state.last_processed_file_info = None

    st.markdown("---")
    st.subheader("Trạng thái dữ liệu hiện tại:")
    if st.session_state.vector_store:
        st.write("✅ Dữ liệu đã được tải và sẵn sàng sử dụng trong Chatbot.")
    else:
        st.write("❌ Chưa có dữ liệu nào được tải. Vui lòng tải lên tài liệu.")

    # if st.button("Xóa FAISS Index hiện có"):
    #     if os.path.exists(FAISS_PATH):
    #         import shutil
    #         try:
    #             shutil.rmtree(FAISS_PATH)
    #             st.session_state.vector_store = None
    #             st.session_state.initial_faiss_loaded_toast_shown = False
    #             st.session_state.initial_faiss_not_found_toast_shown = False
    #             st.session_state.initial_faiss_error_toast_shown = False
    #             st.session_state.initial_faiss_load_attempted = False
    #             st.success("FAISS index đã được xóa thành công. Vui lòng tải lại trang hoặc tải lên tài liệu mới để tạo lại.")
    #             st.rerun()
    #         except Exception as e:
    #             st.error(f"Lỗi khi xóa FAISS index: {e}")
    #     else:
    #         st.info("Không tìm thấy FAISS index để xóa.")
    st.subheader("Kiểm tra tài liệu")
    st.write("Nhấn nút dưới đây để thực hiện kiểm tra tự động")

    if 'docs_check_result' not in st.session_state:
        st.session_state.docs_check_result = ""
    
    if st.button("Thực hiện Kiểm tra"):
        canchu_check_result_placeholder = st.empty()
        with st.spinner("Đang thực hiện kiểm tra"):
            result_for_button = handleCheck(
                chat,
                st.session_state.vector_store
            )
            st.session_state.docs_check_result = result_for_button
    
    if st.session_state.docs_check_result:
        columns = ["STT", "Tên văn bản", "Ngày phát hành", "Trạng thái"]
        df = pd.DataFrame(st.session_state.docs_check_result, columns=columns)
        
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Không có dữ liệu nào để hiển thị.")