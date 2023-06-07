# from configs.model_config import *
from configs.scnu_model_config import *
# from chains.local_doc_qa import LocalDocQA
from chains.scnu_local_doc_qa import LocalDocQA
import os
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

def main():
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(llm_model=LLM_MODEL,
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device=EMBEDDING_DEVICE,
                          llm_history_len=LLM_HISTORY_LEN,
                          top_k=VECTOR_SEARCH_TOP_K)
    # 默认知识库
    vs_path = 'vector_store/SCNU_KB_v4'
    
    while not vs_path:
        filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
        # 判断 filepath 是否为空，如果为空的话，重新让用户输入,防止用户误触回车
        if not filepath:
            continue
        vs_path, _ = local_doc_qa.init_knowledge_vector_store(filepath)
        
    history = []
    
    while True:
        query = input("Input your question 请输入问题：")
        last_print_len = 0
        print(type(query))
        # 多轮对话
        for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                     vs_path=vs_path,
                                                                     chat_history=history,
                                                                     streaming=STREAMING):
            if STREAMING:
                print(resp["result"][last_print_len:], end="", flush=True)
                last_print_len = len(resp["result"])
            else:
                print(resp["result"])
        if REPLY_WITH_SOURCE:
            source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
                           f"""片段相关度：{doc.metadata['score']}\n\n"""
                           for inum, doc in
                           enumerate(resp["source_documents"])]
            print("\n\n" + "\n\n".join(source_text))
        
        


if __name__ == "__main__":
    main()
