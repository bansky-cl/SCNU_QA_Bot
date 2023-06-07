from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from configs.scnu_model_config import *
import datetime
from textsplitter import ChineseTextSplitter
from typing import List, Tuple
from langchain.docstore.document import Document
import numpy as np
from utils import torch_gc
from tqdm import tqdm
import json
from pypinyin import lazy_pinyin
# from loader import UnstructuredPaddleImageLoader
# from loader import UnstructuredPaddlePDFLoader

DEVICE_ = EMBEDDING_DEVICE
DEVICE_ID = "0" if torch.cuda.is_available() else None
DEVICE = f"{DEVICE_}:{DEVICE_ID}" if DEVICE_ID else DEVICE_


def load_file(filepath, sentence_size=SENTENCE_SIZE):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredPaddlePDFLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True, sentence_size=sentence_size)
        docs = loader.load_and_split(textsplitter)
    elif filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
        loader = UnstructuredPaddleImageLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    else: # 普通文本分词
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
        docs = loader.load_and_split(text_splitter=textsplitter)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    fout = open(fp, 'a')
    fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
    fout.write('\n')
    for i in docs:
        fout.write(str(i))
        fout.write('\n')
    fout.close()

# relative + query 一起給模型做阅读理解 PROMPT_TEMPLATE在config.py里面
def generate_prompt(related_docs: List[str], query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    
    # context = "\n".join(['相关文档片段：\n' + doc.page_content + '\n相似度是' + str(round((doc.metadata['score']) / 1100, 4)) + '\n' for doc in related_docs] )
    context = "\n".join(['相关文档片段：\n' + doc.page_content +  '\n' for doc in related_docs] )
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    
    # logger.info(f"###############\n\n{prompt}\n\n################")
    
    return prompt


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


# def similarity_search_with_score(
#         self, query: str, k: int = 4
#     ) -> List[Tuple[Document, float]]:
#     """
#         Return docs most similar to query.

#         Args:
#             query: Text to look up documents similar to.
#             k: Number of Documents to return. Defaults to 4.
#             filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

#         Returns:
#             List of Documents most similar to the query and score for each
#     """
#     embedding = self.embedding_function.embed_query(query)
#     docs = self.similarity_search_with_score_by_vector(
#             embedding=embedding, k=k
#     )
    
    
    
#     return docs


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
) -> List[Tuple[Document, float]]:
    # logger.info(f"embedding shape:{embedding}\n") # 一个1024维的实数矩阵
    # logger.info(f"np.embedding shape:{np.array([embedding], dtype=np.float32).shape}\n") # # np.embedding shape:(1, 1024)
    
    # 这个index应该是faiss的index
    # logger.info(f"self.index type:{type(self.index)}") #  self.index type:<class 'faiss.swigfaiss_avx2.IndexFlat'>
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    
    # logger.info(f"scores type:{type(scores)}\n") # scores type:<class 'numpy.ndarray'>
    # logger.info(f"scores shape:{scores.shape}\n") # scores shape:(1, 5)
    # logger.info(f"indices type:{type(indices)}\n") # indices type:<class 'numpy.ndarray'>
    # logger.info(f"indices shape:{indices.shape}\n") # indices shape:(1, 5)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        # logger.info(f"indices[0] type:{type(indices[0])}\n")  # type:<class 'numpy.ndarray'>
        # logger.info(f"indices[0] shape:{indices[0].shape}\n") # shape:(5,)
        # logger.info(f"scores[0][j] type:{type(scores[0][j])}\n") # type:<class 'numpy.float32'>
        # logger.info(f"scores[0][j] value:{(scores[0][j])}\n") 
        if i == -1 or 0 < self.score_threshold < scores[0][j]: # scores[0][j] value:102.94046783447266 这个就是相关度
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i] # index和id做了映射
        # logger.info(f"i value:{i}\n")  # i value:55621
        # logger.info(f"_id value:{_id}\n")  # _id value:2cdd0006-0389-439c-a73e-61ae069e0f78
        
        doc = self.docstore.search(_id)
        # logger.info(f"doc value:{_id}\n") # 貌似跟_id一样 doc value:2cdd0006-0389-439c-a73e-61ae069e0f78
        
        if not self.chunk_conent:
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc.metadata["score"] = int(scores[0][j])
            docs.append(doc)
            continue
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    if not self.chunk_conent:
        return docs
    if len(id_set) == 0 and self.score_threshold > 0:
        return []
    
    # logger.info(f"id_set value:{id_set}\n")
    
    id_list = sorted(list(id_set))
    # logger.info(f"id_list value:{id_list}\n")

    id_lists = seperate_list(id_list)
    # logger.info(f"id_lists value:{id_lists}\n")
    
    for id_seq in id_lists: # 遍历5次
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                # logger.info(f"_id value:{_id}\n")
                doc = self.docstore.search(_id)
                # logger.info(f"doc value:{doc}\n")
            else:
                _id0 = self.index_to_docstore_id[id]
                # logger.info(f"_id0 value:{_id0}\n")
                doc0 = self.docstore.search(_id0)
                # logger.info(f"doc0 value:{doc0}\n")
                
                doc.page_content += doc0.page_content
                # logger.info(f"doc0.page_content value:{doc0.page_content}\n")
                
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
        # logger.info(f"doc_score type:{type(doc_score)}\n") # 'numpy.float32'
        # logger.info(f"doc_score value:{doc_score}\n")
        doc.metadata["score"] = int(doc_score)
        # logger.info(f"doc.metadata[score] value:{doc.metadata['score']}\n")
        
        # logger.info(f"doc value:{doc}\n")
        # filename_path = doc.metadata['filename'].replace('.txt','').rsplit('/', 1)
        # logging.info(f"current doc's filename:{filename}") # data/SCNU_KB_v2/重磅！华师砺儒高中落地信宜！.txt
        # logging.info(f"current doc's filename type:{type(filename)}") # str
        # _, filename = filename_path.replace('.txt','').rsplit('/', 1)[1]
        # logging.info(f"filename: {os.path.split(doc.metadata['source'])[-1]}")
        # logging.info(f"filename type: {type(os.path.split(doc.metadata['source'])[-1])}")
        # filename_embedding = self.embedding_function.embed_query(filename) # 转不了=-=
        # logging.info(f"filename_embedding type: {type(filename_embedding)}")
        # logging.info(f"filename_embedding: {filename_embedding}")
        
        
        docs.append(doc)
    torch_gc()
    return docs


class LocalDocQA:
    llm: object = None
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE
    chunk_conent: bool = True
    score_threshold: int = VECTOR_SEARCH_SCORE_THRESHOLD

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL,
                 embedding_device=EMBEDDING_DEVICE,
                 llm_history_len: int = LLM_HISTORY_LEN,
                 llm_model: str = LLM_MODEL,
                 llm_device=LLM_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 use_ptuning_v2: bool = USE_PTUNING_V2,
                 use_lora: bool = USE_LORA,
                 ):
        if llm_model.startswith('moss'):
            from models.moss_llm import MOSS
            self.llm = MOSS()
        else:
            from models.chatglm_llm import ChatGLM
            self.llm = ChatGLM()
        self.llm.load_model(model_name_or_path=llm_model_dict[llm_model],
                            llm_device=llm_device, use_ptuning_v2=use_ptuning_v2, use_lora=use_lora)
        self.llm.history_len = llm_history_len

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    sentence_size=SENTENCE_SIZE):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("路径不存在")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath, sentence_size)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(filepath)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in tqdm(os.listdir(filepath), desc="加载文件"):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath, sentence_size)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        logger.error(e)
                        failed_files.append(file)

                if len(failed_files) > 0:
                    logger.info("以下文件未能成功加载：")
                    for file in failed_files:
                        logger.info(f"{file}\n")

        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            
            # 加文档
            logger.info("文件加载完毕，正在生成向量库")
            
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings) # embeddings是编码模型。hf的text2vec
                # add 文档
                vector_store.add_documents(docs)
                torch_gc()
            else:# 创建第一个文档
                if not vs_path:
                    # 向量库名称
                    vs_path = os.path.join(VS_ROOT_PATH,
                                           f"""{"".join(lazy_pinyin(os.path.splitext(file)[0]))}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                vector_store = FAISS.from_documents(docs, self.embeddings)  # docs 为Document列表
                torch_gc()

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    def one_knowledge_add(self, vs_path, one_title, one_conent, one_content_segmentation, sentence_size):
        try:
            if not vs_path or not one_title or not one_conent:
                logger.info("知识库添加错误，请确认知识库名字、标题、内容是否正确！")
                return None, [one_title]
            docs = [Document(page_content=one_conent + "\n", metadata={"source": one_title})]
            if not one_content_segmentation:
                text_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
                docs = text_splitter.split_documents(docs)
            if os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
            else:
                vector_store = FAISS.from_documents(docs, self.embeddings)  ##docs 为Document列表
            torch_gc()
            vector_store.save_local(vs_path)
            return vs_path, [one_title]
        except Exception as e:
            logger.error(e)
            return None, [one_title]
        
    # key function
    def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector # 重写了
        # FAISS.similarity_search_with_score = similarity_search_with_score # 额外重写了
        vector_store.chunk_size = self.chunk_size
        vector_store.chunk_conent = self.chunk_conent
        vector_store.score_threshold = self.score_threshold
        # 下面的函数调用了上面重写过的similarity_search_with_score_by_vector函数
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k) # 没重写，从vs里面搜索前五的
        torch_gc()
        prompt = generate_prompt(related_docs_with_score, query)

        for result, history in self.llm._call(prompt=prompt,
                                              history=chat_history,
                                              streaming=streaming):
            torch_gc()
            history[-1][0] = query
            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs_with_score}
            yield response, history
            torch_gc()
            
    # def get_knowledge_based_answer(self, query, vs_path, chat_history=[], streaming: bool = STREAMING):
    #     vector_store = FAISS.load_local(vs_path, self.embeddings)
    #     FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector # 重写了
    #     # FAISS.similarity_search_with_score = similarity_search_with_score # 额外重写了
    #     vector_store.chunk_size = self.chunk_size
    #     vector_store.chunk_conent = self.chunk_conent
    #     vector_store.score_threshold = self.score_threshold
    #     # 下面的函数调用了上面重写过的similarity_search_with_score_by_vector函数
    #     related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k) # 没重写，从vs里面搜索前五的
    #     torch_gc()
    #     prompt = generate_prompt(related_docs_with_score, query)
        

    #     # logger.info(prompt[:10])
        
    #     response = {"query": query, "prompt": str(prompt), "source_documents": related_docs_with_score}
        
        
        
    #     return response
        # torch_gc()
        # for result, history in self.llm._call(prompt=prompt,
        #                                       history=chat_history,
        #                                       streaming=streaming):
            # history[-1][0] = query
        # response = {"query": query,'prompt': prompt,"source_documents": related_docs_with_score}

        # with open('llm_query.json', 'a', encoding='utf-8') as opf:
        #     json.dump({'query' : prompt}, opf, ensure_ascii=False)
        #     opf.write('\n')
        #     opf.close()

    # query      查询内容
    # vs_path    知识库路径
    # chunk_conent   是否启用上下文关联
    # score_threshold    搜索匹配score阈值
    # vector_search_top_k   搜索知识库内容条数，默认搜索5条结果
    # chunk_sizes    匹配单段内容的连接上下文长度
    def get_knowledge_based_conent_test(self, query, vs_path, chunk_conent,
                                        score_threshold=VECTOR_SEARCH_SCORE_THRESHOLD,
                                        vector_search_top_k=VECTOR_SEARCH_TOP_K, chunk_size=CHUNK_SIZE):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_conent = chunk_conent
        vector_store.score_threshold = score_threshold
        vector_store.chunk_size = chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=vector_search_top_k) 
        if not related_docs_with_score:
            response = {"query": query,
                        "source_documents": []}
            return response, ""
        torch_gc()
        prompt = "\n".join([doc.page_content for doc in related_docs_with_score])
        response = {"query": query,
                    "source_documents": related_docs_with_score}
        return response, prompt


if __name__ == "__main__":
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg()
    query = "本项目使用的embedding模型是什么，消耗多少显存"
    vs_path = "/Users/liuqian/Downloads/glm-dev/vector_store/aaa"
    last_print_len = 0
    
    
    
    files = os.listdir('')
    
    for file in files:
        local_doc_qa.get_knowledge_based_answer(query=file.replace('.txt',''),vs_path=vs_path,streaming=True)
    
    # for resp, history in local_doc_qa.get_knowledge_based_answer(query=query,
    #                                                              vs_path=vs_path,
    #                                                              chat_history=[],
    #                                                              streaming=True):
    #     logger.info(resp["result"][last_print_len:], end="", flush=True)
    #     last_print_len = len(resp["result"])
    
    # source_text = [f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
    #             #    f"""相关度：{doc.metadata['score']}\n\n"""
    #                for inum, doc in
    #                enumerate(resp["source_documents"])]
    # logger.info("\n\n" + "\n\n".join(source_text))
    pass
