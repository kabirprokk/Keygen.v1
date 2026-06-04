import re
import random
import json
import os
import ssl
import urllib.request
import urllib.parse
import urllib.error
import hashlib
import time
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

# Try to import TextBlob, but provide fallback if not available
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

class KeyGenAI:
    def __init__(self, knowledge_dir="knowledge", data_file="data.json", gk_file="gk_knowledge.json"):
        self.name = "KeyGen.ai"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_dir = os.path.join(self.script_dir, knowledge_dir)
        self.data_file = os.path.join(self.script_dir, data_file)
        self.gk_file = os.path.join(self.script_dir, gk_file)
        
        # Persistent Memory Files
        self.user_mem_file = os.path.join(self.knowledge_dir, "user_mem.txt")
        self.verified_web_file = os.path.join(self.knowledge_dir, "verified_web.txt")
        self.search_cache_file = os.path.join(self.knowledge_dir, "search_cache.json")
        
        self.raw_data_chunks = []
        self.markov_graph = defaultdict(list)
        self.knowledge_base = []
        self.gk_base = []
        self.search_cache = {}
        self.stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                         "to", "at", "by", "for", "of", "with", "in", "on", "that", "this"}
        
        # Enhanced greetings database
        self.greetings = {
            "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                        "howdy", "greetings", "sup", "what's up", "yo", "hola", "bonjour"],
            "responses": [
                "Hello! I'm {name}, your AI assistant. How can I help you today?",
                "Hi there! I'm {name}. What would you like to know?",
                "Hey! I'm {name}, ready to assist you. Ask me anything!",
                "Greetings! I'm {name}. I can search the internet and answer questions. What can I do for you?",
                "Hello! {name} at your service. How may I help you?",
                "Hi! Nice to meet you! I'm {name}. What can I help you with?",
                "Hey there! {name} here. Feel free to ask me anything!",
                "Welcome! I'm {name}, your intelligent assistant. How can I assist you today?"
            ],
            "follow_ups": [
                " What would you like to learn about?",
                " How can I assist you?",
                " Ask me anything!",
                " I can search the internet for you!",
                " Need help with something?",
                " What's on your mind?"
            ]
        }
        
        self.emotions = {
            "happy": ["I'm delighted to see you're in a good mood!", "That's wonderful news!", "I'm glad you're feeling positive!"],
            "sad": ["I'm sorry you're feeling this way. I'm here to help.", "I understand. Sometimes things are difficult."],
            "angry": ["I hear you're frustrated. Let's try to resolve this together.", "I sense some tension."],
            "lonely": ["I may be a program, but I am always here to talk.", "You're not alone while I'm active."]
        }
        
        # SSL context for HTTPS requests
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize directories and load data
        os.makedirs(self.knowledge_dir, exist_ok=True)
        self.load_all_data()
        self.load_search_cache()

    def load_search_cache(self):
        """Load cached search results"""
        try:
            if os.path.exists(self.search_cache_file):
                with open(self.search_cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
        except Exception:
            self.search_cache = {}

    def save_search_cache(self):
        """Save search results to cache"""
        try:
            if len(self.search_cache) > 100:
                keys = list(self.search_cache.keys())[-100:]
                self.search_cache = {k: self.search_cache[k] for k in keys}
            
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, indent=2)
        except Exception:
            pass

    def tokenize(self, text):
        """Fixed tokenization with better pattern matching"""
        if not text:
            return []
        return re.findall(r'\b\w+\b', str(text).lower())

    def build_markov(self, tokens):
        """Build Markov chain with better error handling"""
        if not tokens or len(tokens) < 2:
            return
        for i in range(len(tokens) - 1):
            if tokens[i] and tokens[i+1]:
                self.markov_graph[tokens[i]].append(tokens[i+1])

    def is_greeting(self, text):
        """Check if input is a greeting"""
        text_lower = text.lower().strip().rstrip('!.,?')
        for pattern in self.greetings["patterns"]:
            if text_lower == pattern or text_lower.startswith(pattern):
                return True
        return False

    def get_greeting_response(self):
        """Generate a random greeting response"""
        response = random.choice(self.greetings["responses"]).format(name=self.name)
        follow_up = random.choice(self.greetings["follow_ups"])
        return response + follow_up

    def get_emotion_prefix(self, text):
        """Enhanced emotion detection"""
        if not text:
            return ""
        text_lower = text.lower()
        for emotion, responses in self.emotions.items():
            if emotion in text_lower:
                return random.choice(responses) + " "
        return ""

    def grammar_checker(self, text):
        """Enhanced grammar checker"""
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        # Basic capitalization
        if len(text) > 1:
            text = text[0].upper() + text[1:]
        else:
            text = text.upper()
        
        # Ensure proper ending punctuation
        if text[-1] not in ".!?:":
            text += "."
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common grammar issues
        text = re.sub(r'\bi\b(?![\'\.])', 'I', text)
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        
        return text

    def rephraser(self, text, style="clean"):
        """Enhanced rephraser"""
        if not text:
            return ""
        
        if not TEXTBLOB_AVAILABLE:
            return self._basic_rephrase(text, style)
        
        try:
            blob = TextBlob(text)
            
            if style == "clean":
                words = text.split()
                cleaned = []
                for i, word in enumerate(words):
                    if word.lower() not in self.stopwords or (i > 0 and i < len(words)-1):
                        cleaned.append(word)
                result = " ".join(cleaned) if cleaned else text
                
            elif style == "professional":
                result = text.replace("I think", "Based on analysis")
                result = result.replace("maybe", "potentially")
                result = result.replace("a lot", "significantly")
                
            elif style == "simple":
                sentences = re.split(r'(?<=[.!?])\s+', text)
                simple_sentences = []
                for sentence in sentences:
                    words = sentence.split()
                    if len(words) > 20:
                        chunk_size = 15
                        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                        simple_sentences.extend(chunks)
                    else:
                        simple_sentences.append(sentence)
                result = ". ".join(simple_sentences)
                
            else:
                result = text
                creative_phrases = ["Interestingly, ", "Notably, ", "Furthermore, "]
                if len(result.split()) > 5:
                    insert_pos = len(result) // 3
                    result = result[:insert_pos] + random.choice(creative_phrases) + result[insert_pos:]
            
            return self.grammar_checker(result)
        except Exception:
            return self._basic_rephrase(text, style)

    def _basic_rephrase(self, text, style):
        """Basic rephrasing without TextBlob"""
        if style == "clean":
            words = [w for w in text.split() if w.lower() not in self.stopwords]
            return " ".join(words) if words else text
        elif style == "simple":
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return ". ".join(sentences[:3])
        return text

    def make_http_request(self, url, timeout=10):
        """Centralized HTTP request handler with SSL support"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"HTTP request error: {e}")
            return None

    def google_search(self, query):
        """Enhanced multi-engine search with caching and fallbacks"""
        if not query:
            return None
            
        print(f"🔍 Searching internet for: {query}")
        
        # Check cache first
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 3600:
                print("✓ Using cached result")
                return cache_entry['data']
        
        clean_query = query.strip()
        
        # Multiple search engines
        search_engines = [
            {
                "name": "Google",
                "url": f"https://www.google.com/search?q={urllib.parse.quote(clean_query)}",
                "parser": self._parse_google_results
            },
            {
                "name": "Bing",
                "url": f"https://www.bing.com/search?q={urllib.parse.quote(clean_query)}",
                "parser": self._parse_bing_results
            },
            {
                "name": "DuckDuckGo",
                "url": f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(clean_query)}",
                "parser": self._parse_duckduckgo_results
            }
        ]
        
        for engine in search_engines:
            try:
                html = self.make_http_request(engine['url'], timeout=8)
                if html:
                    results = engine['parser'](html)
                    if results:
                        best_result = self._select_best_result(results)
                        if best_result and len(best_result) > 100:
                            # Cache the result
                            self.search_cache[cache_key] = {
                                'data': best_result,
                                'timestamp': time.time()
                            }
                            self.save_search_cache()
                            
                            polished = self.polish_and_save_web_data(best_result)
                            print(f"✓ Found answer via {engine['name']}")
                            return polished
                            
            except Exception as e:
                print(f"Search error with {engine['name']}: {e}")
                continue
        
        # Try Wikipedia as last resort
        wiki_result = self._search_wikipedia(query)
        if wiki_result:
            return wiki_result
        
        return None

    def _parse_google_results(self, html):
        """Parse Google search results"""
        results = []
        patterns = [
            r'<div class="BNeawe s3v9rd AP7Wnd">(.*?)</div>',
            r'<span class="st">(.*?)</span>',
            r'<div[^>]*class="[^"]*BNeawe[^"]*"[^>]*>(.*?)</div>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                clean_text = re.sub(r'<.*?>', '', match)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                if len(clean_text) > 100:
                    results.append(clean_text)
        return results

    def _parse_bing_results(self, html):
        """Parse Bing search results"""
        results = []
        patterns = [
            r'<p[^>]*>(.*?)</p>',
            r'<div class="b_caption[^"]*"[^>]*>.*?<p[^>]*>(.*?)</p>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                clean_text = re.sub(r'<.*?>', '', match)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                if len(clean_text) > 100:
                    results.append(clean_text)
        return results

    def _parse_duckduckgo_results(self, html):
        """Parse DuckDuckGo results"""
        results = []
        patterns = [
            r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>',
            r'<td class="result-sn-abstract[^"]*">(.*?)</td>',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                clean_text = re.sub(r'<.*?>', '', match)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                if len(clean_text) > 100:
                    results.append(clean_text)
        return results

    def _select_best_result(self, results):
        """Select the most relevant and comprehensive result"""
        if not results:
            return None
            
        scored_results = []
        for result in results:
            score = 0
            score += len(result) / 100
            score += len(re.findall(r'[.!?]', result)) * 5
            score -= result.count('<') * 10
            factual_words = ['is', 'are', 'was', 'were', 'has', 'have', 'according', 'research', 'study']
            score += sum(2 for word in factual_words if word in result.lower())
            scored_results.append((score, result))
        
        scored_results.sort(reverse=True)
        best_result = scored_results[0][1]
        
        if len(best_result) > 1500:
            best_result = best_result[:1500] + "..."
            
        return best_result

    def _search_wikipedia(self, query):
        """Search Wikipedia API"""
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=3"
            
            req = urllib.request.Request(api_url, headers={'User-Agent': 'KeyGenAI/1.0'})
            with urllib.request.urlopen(req, timeout=10, context=self.ssl_context) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('query', {}).get('search'):
                page_id = data['query']['search'][0]['pageid']
                
                extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&pageids={page_id}&format=json"
                req = urllib.request.Request(extract_url, headers={'User-Agent': 'KeyGenAI/1.0'})
                with urllib.request.urlopen(req, timeout=10, context=self.ssl_context) as response:
                    extract_data = json.loads(response.read().decode('utf-8'))
                
                pages = extract_data.get('query', {}).get('pages', {})
                for pid, page_data in pages.items():
                    extract = page_data.get('extract', '')
                    if extract:
                        if len(extract) > 1500:
                            extract = extract[:1500] + "..."
                        return self.polish_and_save_web_data(extract)
                        
        except Exception as e:
            print(f"Wikipedia API error: {e}")
        
        return None

    def polish_and_save_web_data(self, text):
        """Enhanced web data cleaning and storage"""
        if not text:
            return text
            
        clean = re.sub(r'<.*?>', '', text)
        
        noise_patterns = [
            r'(?i)click here', r'(?i)read more', r'(?i)cookies?',
            r'(?i)privacy policy', r'(?i)subscribe', r'(?i)advertisement',
            r'(?i)accept cookies', r'(?i)terms of (use|service)',
            r'(?i)all rights reserved', r'(?i)copyright \d{4}'
        ]
        
        for pattern in noise_patterns:
            clean = re.sub(pattern, '', clean)
        
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        if len(clean) > 50:
            try:
                with open(self.verified_web_file, 'a', encoding='utf-8') as f:
                    f.write(clean + "\n\n")
                self.raw_data_chunks.append(clean)
            except Exception:
                pass
        
        return clean

    def calculate_relevance_score(self, question, text):
        """Enhanced relevance scoring"""
        if not question or not text:
            return 0
            
        question_words = set(self.tokenize(question))
        text_words = set(self.tokenize(text))
        
        if not question_words:
            return 0
        
        intersection = len(question_words.intersection(text_words))
        union = len(question_words.union(text_words))
        
        if union == 0:
            return 0
            
        return intersection / union

    def get_detailed_answer(self, question):
        """Get detailed answer - ALWAYS searches internet if local data insufficient"""
        if not question:
            return None
        
        # Check local knowledge base first
        best_match = None
        highest_score = 0
        best_match_text = ""
        
        for sentence in self.raw_data_chunks:
            score = self.calculate_relevance_score(question, sentence)
            if score > highest_score:
                highest_score = score
                best_match_text = sentence
        
        # If local match found and it's substantial
        if best_match_text and highest_score > 0.3 and len(best_match_text) > 100:
            return self._format_answer(question, best_match_text, "local")
        
        # If local answer is poor or non-existent, ALWAYS search internet
        print("📡 Local knowledge insufficient, searching internet...")
        search_result = self.google_search(question)
        if search_result:
            return self._format_answer(question, search_result, "internet")
        
        # Try Wikipedia as final fallback
        wiki_result = self._search_wikipedia(question)
        if wiki_result:
            return self._format_answer(question, wiki_result, "wikipedia")
        
        return None

    def _format_answer(self, question, content, source):
        """Format answer based on question type and source"""
        question_lower = question.lower().strip()
        
        # Determine question type
        if question_lower.startswith("what"):
            prefix = "Here's what I found"
            if source == "internet":
                prefix = "According to my internet search, here's what I found"
        elif question_lower.startswith("why"):
            prefix = "Here's the explanation"
        elif question_lower.startswith("how"):
            prefix = "Let me explain how this works"
        elif question_lower.startswith("where"):
            prefix = "Here's the location information"
        elif question_lower.startswith("when"):
            prefix = "Here's the timeline"
        elif question_lower.startswith("who"):
            prefix = "Here's who I found"
        elif question_lower.startswith(("which", "can", "is", "are", "do", "does")):
            prefix = "Based on my research"
        else:
            prefix = "Here's what I found"
        
        # Format the response
        response = f"{prefix}:\n\n{content}"
        
        # Add source attribution
        if source == "internet":
            response += "\n\n(Source: Internet search)"
        elif source == "wikipedia":
            response += "\n\n(Source: Wikipedia)"
        
        return response

    def learn_from_user(self, text):
        """Enhanced autonomous learning"""
        if not text or len(text.split()) < 8 or "?" in text:
            return False
            
        factual_patterns = [" is ", " was ", " are ", " were ", " has ", " have ", " will "]
        
        if any(pattern in text.lower() for pattern in factual_patterns):
            try:
                with open(self.user_mem_file, 'a', encoding='utf-8') as f:
                    f.write(text.strip() + ".\n")
                self.raw_data_chunks.append(text.strip())
                return True
            except Exception:
                pass
        return False

    def load_all_data(self):
        """Enhanced data loading"""
        # Load JSON data files
        json_files = [
            (self.data_file, 'knowledge_base'),
            (self.gk_file, 'gk_base')
        ]
        
        for file_path, attr_name in json_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        setattr(self, attr_name, json.load(f))
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump([], f)
                    setattr(self, attr_name, [])
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                setattr(self, attr_name, [])
        
        # Load text knowledge files
        all_tokens = []
        if os.path.exists(self.knowledge_dir):
            for filename in os.listdir(self.knowledge_dir):
                if filename.endswith(".txt"):
                    filepath = os.path.join(self.knowledge_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            if text.strip():
                                sentences = re.split(r'(?<=[.!?])\s+', text)
                                self.raw_data_chunks.extend([s.strip() for s in sentences if len(s) > 10])
                                all_tokens.extend(self.tokenize(text))
                    except Exception:
                        pass
        
        if all_tokens:
            self.build_markov(all_tokens)
        
        print(f"✓ {self.name} SYSTEM ONLINE")
        print(f"✓ Knowledge base: {len(self.raw_data_chunks)} sentences loaded")

    def get_response(self, user_input):
        """Main response handler - FIXED and OPTIMIZED"""
        if not user_input or not user_input.strip():
            return "Please type something and I'll help you!"
        
        raw_input = user_input.strip()
        raw_input_lower = raw_input.lower()
        
        # 1. CHECK FOR GREETINGS FIRST (Fixed!)
        if self.is_greeting(raw_input):
            return self.get_greeting_response()
        
        # 2. Check for emotion
        emotion_prefix = self.get_emotion_prefix(raw_input_lower)
        
        # 3. Learn from user input
        self.learn_from_user(user_input)
        
        # 4. Handle "learn about" command
        if raw_input_lower.startswith("learn about "):
            topic = raw_input[12:].strip()  # Remove "learn about "
            result = self.google_search(topic)
            if result:
                result = self.rephraser(result, "clean")
                return self.grammar_checker(emotion_prefix + f"I learned about {topic}:\n\n{result}")
            return f"I couldn't find information about '{topic}'. Please try a different topic."
        
        # 5. Check if it's a question (ALWAYS search internet for questions)
        is_question = ("?" in raw_input or 
                      raw_input_lower.startswith(("what", "why", "how", "where", "when", "who", 
                                                   "which", "can", "is", "are", "do", "does",
                                                   "explain", "tell", "describe", "define")))
        
        if is_question:
            answer = self.get_detailed_answer(raw_input)
            if answer:
                answer = self.rephraser(answer, "clean")
                return self.grammar_checker(emotion_prefix + answer)
            else:
                return "I searched everywhere but couldn't find a reliable answer. Could you rephrase your question?"
        
        # 6. Check fact engine (GK)
        for fact in self.gk_base:
            if fact.get("q", "").lower() in raw_input_lower:
                result = self.rephraser(fact["a"], "clean")
                return self.grammar_checker(emotion_prefix + result)
        
        # 7. Check knowledge modules (JSON)
        for module in self.knowledge_base:
            for pattern in module.get("patterns", []):
                if pattern.lower() in raw_input_lower:
                    result = random.choice(module["responses"])
                    result = self.rephraser(result, "clean")
                    return self.grammar_checker(emotion_prefix + result)
        
        # 8. Pure emotion response
        tokens = self.tokenize(raw_input_lower)
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return self.grammar_checker(emotion_prefix)
        
        # 9. Search local knowledge
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        if keywords:
            best_sentence = None
            max_overlap = 0
            for sentence in self.raw_data_chunks:
                overlap = sum(1 for kw in keywords if kw in sentence.lower())
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence
            
            if best_sentence and max_overlap >= 2 and len(best_sentence) > 100:
                best_sentence = self.rephraser(best_sentence, "clean")
                return self.grammar_checker(emotion_prefix + best_sentence)
        
        # 10. ALWAYS try internet search as final fallback
        search_result = self.google_search(raw_input)
        if search_result and len(search_result) > 100:
            search_result = self.rephraser(search_result, "clean")
            return emotion_prefix + self.grammar_checker(search_result)
        
        # 11. Absolute final fallback
        return "I'm not sure about that. Could you rephrase or ask a different question? You can also use 'learn about [topic]' to help me learn!"


# Web server handler
class ChatHandler(BaseHTTPRequestHandler):
    bot = None
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>KeyGen.ai - AI Assistant</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 20px;
                    }
                    .container {
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        max-width: 800px;
                        width: 100%;
                        overflow: hidden;
                    }
                    .header {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        text-align: center;
                    }
                    .header h1 { font-size: 24px; margin-bottom: 5px; }
                    .header p { font-size: 14px; opacity: 0.9; }
                    #chat {
                        height: 400px;
                        padding: 20px;
                        overflow-y: auto;
                        background: #f8f9fa;
                    }
                    .message {
                        margin-bottom: 15px;
                        padding: 10px 15px;
                        border-radius: 15px;
                        max-width: 80%;
                        animation: fadeIn 0.3s;
                    }
                    @keyframes fadeIn {
                        from { opacity: 0; transform: translateY(10px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    .user-message {
                        background: #667eea;
                        color: white;
                        margin-left: auto;
                    }
                    .ai-message {
                        background: #e9ecef;
                        color: #333;
                    }
                    .input-container {
                        padding: 20px;
                        background: white;
                        border-top: 1px solid #dee2e6;
                    }
                    #input {
                        width: 100%;
                        padding: 12px 20px;
                        border: 2px solid #dee2e6;
                        border-radius: 25px;
                        font-size: 16px;
                        outline: none;
                        transition: border-color 0.3s;
                    }
                    #input:focus {
                        border-color: #667eea;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🤖 KeyGen.ai Assistant</h1>
                        <p>Internet-connected AI - Ask me anything!</p>
                    </div>
                    <div id="chat"></div>
                    <div class="input-container">
                        <input type="text" id="input" placeholder="Type your message here..." autofocus>
                    </div>
                </div>
                <script>
                    const chat = document.getElementById('chat');
                    const input = document.getElementById('input');
                    
                    function addMessage(text, isUser) {
                        const div = document.createElement('div');
                        div.className = 'message ' + (isUser ? 'user-message' : 'ai-message');
                        div.textContent = text;
                        chat.appendChild(div);
                        chat.scrollTop = chat.scrollHeight;
                    }
                    
                    function addTypingIndicator() {
                        const div = document.createElement('div');
                        div.className = 'message ai-message';
                        div.id = 'typing';
                        div.textContent = '...';
                        chat.appendChild(div);
                        chat.scrollTop = chat.scrollHeight;
                    }
                    
                    function removeTypingIndicator() {
                        const typing = document.getElementById('typing');
                        if (typing) typing.remove();
                    }
                    
                    async function sendMessage() {
                        const message = input.value.trim();
                        if (!message) return;
                        
                        addMessage(message, true);
                        input.value = '';
                        addTypingIndicator();
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            });
                            const data = await response.json();
                            removeTypingIndicator();
                            addMessage(data.response, false);
                        } catch (error) {
                            removeTypingIndicator();
                            addMessage('Error connecting to server. Please try again.', false);
                        }
                    }
                    
                    input.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') sendMessage();
                    });
                </script>
            </body>
            </html>
            '''
            self.wfile.write(html.encode())
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy', 'name': 'KeyGen.ai'}).encode())

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                user_msg = data.get('message', '')
                response_text = self.bot.get_response(user_msg)
            except Exception as e:
                response_text = f"Error processing your request: {str(e)}"
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    port = int(os.environ.get("PORT", 10000))
    bot = KeyGenAI()
    ChatHandler.bot = bot
    
    server_address = ('0.0.0.0', port)
    server = HTTPServer(server_address, ChatHandler)
    
    print(f"\n{'='*50}")
    print(f"🤖 KeyGen.ai SYSTEM ONLINE")
    print(f"{'='*50}")
    print(f"🌐 Server: http://0.0.0.0:{port}")
    print(f"📚 Features: Internet Search | Wikipedia | Greetings | Emotions")
    print(f"💡 Tip: Ask any question and I'll search the internet!")
    print(f"{'='*50}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
