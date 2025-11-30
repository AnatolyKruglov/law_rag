from langchain_community.llms import YandexGPT
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class QueryReformulator:
    """Reformulates natural language questions into keyword queries for Consultant Plus"""
    
    def __init__(self, llm=None):
        self.llm = llm or self._create_llm()
    
    def _create_llm(self):
        """Create YandexGPT LLM instance for query reformulation"""
        return YandexGPT(
            folder_id=settings.FOLDER_ID,
            api_key=settings.API_KEY,
            temperature=0.1,  # Low temperature for consistent reformulation
            max_tokens=100
        )


    def reformulate_for_consultant_plus(self, natural_question: str) -> str:
        """Reformulate natural language question into keyword query"""
        prompt = f"""
        Ты - помощник для поиска юридической информации в системе Консультант Плюс.
        Задача: преобразовать вопрос пользователя в формате естественного языка в краткий поисковый запрос из ключевых слов.
        
        Требования к результату:
        - Только ключевые слова (2-4 слова)
        - Без вопросительных слов (как, сколько, когда, почему и т.д.)
        - Без местоимений (я, мне, мой и т.д.)
        - Без предлогов и союзов
        - Только существительные и прилагательные в именительном падеже
        - Сохрани юридическую терминологию
        - Не добавляй кавычки или пояснения
        
        Примеры:
        "Сколько дней отпуска в год я могу взять?" -> "трудовой кодекс отпуск"
        "Какие налоговые вычеты можно получить при покупке квартиры?" -> "налоговый кодекс вычеты недвижимость" 
        "Сколько налогов я заплачу с зарплаты X рублей в месяц?" -> "налоговый кодекс ставка ндфл"
        
        Вопрос: "{natural_question}"
        
        Краткий поисковый запрос:"""
        
        try:
            reformulated_query = self.llm.invoke(prompt).strip()
            logger.info(f"Query reformulated: '{natural_question}' -> '{reformulated_query}'")
            return reformulated_query
        except Exception as e:
            logger.error(f"Error reformulating query: {e}")
            # Fallback: return original question without question words
            return self._fallback_reformulation(natural_question)
    
    def _fallback_reformulation(self, question: str) -> str:
        """Simple fallback reformulation without LLM"""
        # Remove common question words and pronouns
        question_words = ['как', 'сколько', 'когда', 'почему', 'зачем', 'что', 'кто', 'чей', 'куда', 'откуда']
        pronouns = ['я', 'мне', 'мой', 'меня', 'мною', 'мы', 'нам', 'наш', 'нас', 'нами']
        words = question.lower().split()
        filtered_words = [word for word in words if word not in question_words + pronouns]
        return ' '.join(filtered_words)