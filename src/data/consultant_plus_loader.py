import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from typing import List, Optional
import urllib.parse
import logging
import time
import re
import sys
import os

# Add the parent directory to path to import from processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.processing.query_reformulator import QueryReformulator

logger = logging.getLogger(__name__)

class ConsultantPlusLoader:
    """Loader for Консультант Плюс search results and document content"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.base_url = "https://www.consultant.ru"
        self.search_url = "https://www.consultant.ru/search/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.query_reformulator = QueryReformulator()
    
    def _reformulate_query(self, natural_query: str) -> str:
        """Reformulate natural language query into keyword search for Consultant Plus"""
        logger.info(f"Reformulating query: '{natural_query}'")
        reformulated_query = self.query_reformulator.reformulate_for_consultant_plus(natural_query)
        logger.info(f"Reformulated query: '{reformulated_query}'")
        return reformulated_query
    
    def search_documents(self, query: str) -> List[dict]:
        """Search documents on Консультант Плюс and return results with metadata"""
        try:
            # Reformulate the query for better search results
            search_query = self._reformulate_query(query)
            encoded_query = urllib.parse.quote(search_query.encode('utf-8'))
            url = f"{self.search_url}?q={encoded_query}"
            
            logger.info(f"Fetching search results from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Use lxml parser if available, otherwise use html.parser
            try:
                soup = BeautifulSoup(response.content, 'lxml')
            except:
                soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Find search results
            search_results = soup.find('ol', class_='search-results')
            if not search_results:
                logger.warning("No search results found on page")
                return results
            
            items = search_results.find_all('li', class_=re.compile('search-results__item'))[:self.max_results]
            logger.info(f"Found {len(items)} search result items")
            
            for i, item in enumerate(items):
                try:
                    # Skip revoked documents and unavailable ones
                    item_classes = item.get('class', [])
                    if 'search-results__item_revoke' in item_classes:
                        logger.debug(f"Skipping revoked document at position {i}")
                        continue
                    
                    # Check availability
                    availability_icon = item.find('i', class_='search-results__icon')
                    if availability_icon and 'недоступен' in availability_icon.get('title', ''):
                        logger.debug(f"Skipping unavailable document at position {i}")
                        continue
                    
                    link = item.find('a', class_='search-results__link')
                    if not link:
                        continue
                    
                    href = link.get('href')
                    if not href:
                        continue
                    
                    # Make absolute URL
                    if href.startswith('//'):
                        doc_url = 'https:' + href
                    elif href.startswith('/'):
                        doc_url = self.base_url + href
                    else:
                        doc_url = href
                    
                    # Extract title
                    title_elem = link.find('p', class_='search-results__link-inherit')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract description
                    desc_elem = link.find('p', class_='search-results__descr')
                    description = desc_elem.get_text(strip=True) if desc_elem else ""
                    
                    # Extract text info
                    text_elem = link.find('p', class_='search-results__text')
                    text_info = text_elem.get_text(strip=True) if text_elem else ""
                    
                    results.append({
                        'url': doc_url,
                        'title': title,
                        'description': description,
                        'text_info': text_info,
                        'relevance_score': len(results) + 1,
                        'position': i + 1,
                        'original_query': query,  # Store original query for reference
                        'search_query': search_query  # Store reformulated query for reference
                    })
                    
                    logger.debug(f"Added result: {title[:50]}...")
                    
                except Exception as e:
                    logger.warning(f"Error processing search result {i}: {e}")
                    continue
            
            logger.info(f"Processed {len(results)} valid search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Consultant Plus: {e}")
            return []
    
    def load_document_content(self, url: str) -> Optional[str]:
        """Load and extract text content from a document URL"""
        try:
            logger.debug(f"Loading document content from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if it's XML content
            content_type = response.headers.get('content-type', '').lower()
            if 'xml' in content_type or url.endswith('.cgi'):
                # Use XML parser for XML content
                try:
                    soup = BeautifulSoup(response.content, 'xml')
                except:
                    soup = BeautifulSoup(response.content, 'lxml')
            else:
                # Use HTML parser for regular pages
                try:
                    soup = BeautifulSoup(response.content, 'lxml')
                except:
                    soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try to find main content areas - Консультант Плюс specific selectors
            content_selectors = [
                '.document-page__text',
                '.text',
                'article',
                'main',
                '.content',
                '.document-text',
                '#content',
                '.law-content'
            ]
            
            content = None
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    if len(content) > 100:  # Only use if we got substantial content
                        break
            
            # If no specific content found, get body text but clean it up
            if not content or len(content) < 100:
                body = soup.find('body')
                if body:
                    # Remove empty elements and navigation
                    for element in body.find_all(['div', 'span', 'p']):
                        if len(element.get_text(strip=True)) < 10:
                            element.decompose()
                    content = body.get_text(separator=' ', strip=True)
            
            # Clean up the text - remove extra whitespace
            if content:
                content = re.sub(r'\s+', ' ', content)
                content = content.strip()
                
                # Basic validation - if content is too short, it might be an error page
                if len(content) < 50:
                    logger.warning(f"Document content too short ({len(content)} chars), may be invalid")
                    return None
                
                logger.debug(f"Extracted {len(content)} characters from document")
            
            return content
            
        except Exception as e:
            logger.error(f"Error loading document from {url}: {e}")
            return None
    
    def load_documents(self, query: str) -> List[Document]:
        """Main method to load documents based on search query"""
        logger.info(f"Searching Consultant Plus for: '{query}'")
        
        search_results = self.search_documents(query)
        if not search_results:
            logger.warning("No search results found")
            return []
        
        documents = []
        for i, result in enumerate(search_results):
            logger.info(f"Loading document {i+1}/{len(search_results)}: {result['title'][:80]}...")
            
            content = self.load_document_content(result['url'])
            if content:
                # Create LangChain Document with metadata
                metadata = {
                    'source': result['url'],
                    'title': result['title'],
                    'description': result['description'],
                    'text_info': result['text_info'],
                    'relevance_score': result['relevance_score'],
                    'position': result['position'],
                    'original_query': result['original_query'],  # Include original query
                    'search_query': result['search_query']  # Include reformulated query
                }
                
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(document)
                logger.info(f"Successfully loaded document {i+1}")
            else:
                logger.warning(f"Failed to load content for document {i+1}")
            
            # Be respectful with requests
            time.sleep(1.5)
        
        logger.info(f"Loaded {len(documents)} documents from Consultant Plus")
        return documents