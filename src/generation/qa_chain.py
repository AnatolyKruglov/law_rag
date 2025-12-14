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
        """Create QA chain with custom prompt that handles multiple sources"""
        prompt_template = """Ты специалист по российскому праву. 
Используй предоставленный контекст из разных источников (Консультант Плюс и локальные материалы), чтобы подробно ответить на вопрос. 

Анализируй информацию из всех доступных источников и объединяй ее в единый ответ.

Контекст из разных источников:
{context}

Вопрос: {question}

Требования к ответу:
1. Подробно объясни ответ со ссылками на законодательство
2. Укажи конкретные статьи и нормативные акты, если они есть в контексте
3. Если информация получена из локальных материалов (PPTX), укажи это
4. Если информация получена из Консультант Плюс, укажи ссылки на источники
5. Объедини информацию из разных источников, если они дополняют друг друга
6. Будь точным и структурированным

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
                # Modify system prompt to include source awareness
                enhanced_prompt = \
f"""
Ты опытный специалист по российскому праву
Твоя задача: ответить на вопрос юзера, подробно объяснить ответ, дать пояснения простым языком (при необходимости - предоставить понятный алгоритм действий). 
Убедись, что учтены все тонкости и нюансы права. Обязательно укажи ссылки на источники
                
Стиль письма: Простой, ясный и прямой. Пиши как эксперт, который объясняет сложное простым языком. Твой текст должен быть понятен даже неподготовленному читателю с первого прочтения.

*ВАЖНО*:
- Запрещен пассивный залог. Используй только активный. (Вместо "было принято решение" -> "мы решили").
- Избегай отглагольных существительных (номинализаций). (Вместо "провести анализ" -> "проанализировать", вместо "осуществлять контроль" -> "контролировать").
- Не используй канцеляризмы и штампы: "в целях", "в рамках", "в соответствии с", "данный", "вышеуказанный", "настоящий", "имеет место быть" и т.п.
- Пиши короткими предложениями. Разбивай сложные мысли на пункты.
- Выбирай простые, конкретные слова вместо абстрактных и формальных. (Вместо "лицо, ответственное за осуществление" -> "ответственный" или "исполнитель").
- Избегай цепочек родительных падежей. (Вместо "результаты анализа данных проекта" -> "результаты анализа данных по проекту" или "что показал анализ данных проекта").
- Структурируй текст. Используй списки, подзаголовки и абзацы.
"""
                full_question = f"{enhanced_prompt}\n\nВопрос: {question}"
            else:
                full_question = question
            
            result = self.qa_chain.invoke({"query": full_question})
            
            # Analyze source types for better reporting
            source_types = {}
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, 'metadata'):
                        source_type = doc.metadata.get('type', 'unknown')
                        if source_type not in source_types:
                            source_types[source_type] = 0
                        source_types[source_type] += 1
            
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "source_types": source_types,
                "question": question
            }
        except Exception as e:
            logger.error(f"Error querying QA system: {e}")
            raise