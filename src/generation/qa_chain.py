from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import YandexGPT
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class QASystem:
    """Question-Answering system with RAG"""
    
    def __init__(self, retriever, llm=None):
        self.retriever = retriever
        self.llm = llm or self._create_llm()
        self.qa_chain = self._create_qa_chain()
    
    def _create_llm(self):
        """Create YandexGPT LLM instance"""
        return YandexGPT(
            folder_id=settings.FOLDER_ID,
            api_key=settings.API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS
        )
    
    def _create_qa_chain(self):
        """Create QA chain with custom prompt"""
        prompt_template = """Ты специалист по российскому праву. 
Используй предоставленный контекст из системы Консультант Плюс, чтобы подробно ответить на вопрос. 
Если в контексте нет достаточной информации, используй свои знания, но укажи это.

Контекст:
{context}

Вопрос: {question}

Требования к ответу:
1. Подробно объясни ответ со ссылками на законодательство
2. Укажи конкретные статьи и нормативные акты
3. Будь точным и структурированным
4. Укажи источники информации

Ответ:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str, system_prompt: str = None) -> dict:
        """Query the QA system"""
        try:
            if system_prompt:
                full_question = f"{system_prompt}\n\nВопрос: {question}"
            else:
                full_question = question
            
            result = self.qa_chain({"query": full_question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "question": question
            }
        except Exception as e:
            logger.error(f"Error querying QA system: {e}")
            raise