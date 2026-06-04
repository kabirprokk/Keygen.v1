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
    print("Warning: TextBlob not installed. Using basic grammar functions.")

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
            # Keep only last 100 entries
            if len(self.search_cache) > 100:
                keys = list(self.search_cache.keys())[-100:]
                self.search_cache = {k: self.search_cache[k] for k in keys}
            
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, indent=2)
        except Exception as e:
            print(f"Cache save error: {e}")

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
            if tokens[i] and tokens[i+1]:  # Skip empty tokens
                self.markov_graph[tokens[i]].append(tokens[i+1])

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
        """Enhanced grammar checker with fallback"""
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
        text = re.sub(r'\bi\b', 'I', text)  # Capitalize I
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Fix spacing before punctuation
        
        return text

    def rephraser(self, text, style="clean"):
        """Enhanced rephraser with TextBlob fallback"""
        if not text:
            return ""
        
        # If TextBlob is not available, use basic rephrasing
        if not TEXTBLOB_AVAILABLE:
            return self._basic_rephrase(text, style)
        
        try:
            blob = TextBlob(text)
            
            if style == "clean":
                # Remove redundant words and simplify
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
                result = result.replace("good", "excellent")
                
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
                
            else:  # creative
                result = text
                creative_phrases = ["Interestingly, ", "Notably, ", "Furthermore, ", "In addition, "]
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
            return ". ".join(sentences[:3])  # Return first 3 sentences
        return text

    def make_http_request(self, url, timeout=10):
        """Centralized HTTP request handler with SSL support"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as response:
                return response.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"HTTP request error for {url}: {e}")
            return None

    def google_search(self, query):
        """Enhanced multi-engine search with caching and fallbacks"""
        if not query:
            return None
            
        print(f"({self.name} is searching the internet for accurate information...)")
        
        # Check cache first
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 3600:  # 1 hour cache
                print("Using cached result")
                return cache_entry['data']
        
        clean_query = query.strip()
        
        # Multiple search engines with different parsing strategies
        search_engines = [
            {
                "name": "Google",
                "url": f"https://www.google.com/search?q={urllib.parse.quote(clean_query)}&num=10",
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
                print(f"Trying {engine['name']}...")
                html = self.make_http_request(engine['url'], timeout=8)
                
                if html:
                    results = engine['parser'](html)
                    if results:
                        best_result = self._select_best_result(results)
                        if best_result:
                            # Cache the result
                            self.search_cache[cache_key] = {
                                'data': best_result,
                                'timestamp': time.time()
                            }
                            self.save_search_cache()
                            
                            polished = self.polish_and_save_web_data(best_result)
                            return polished
                            
            except Exception as e:
                print(f"Search error with {engine['name']}: {e}")
                continue
        
        return None

    def _parse_google_results(self, html):
        """Parse Google search results"""
        results = []
        
        # Multiple regex patterns for different result formats
        patterns = [
            r'<div class="BNeawe s3v9rd AP7Wnd">(.*?)</div>',
            r'<span class="st">(.*?)</span>',
            r'<div[^>]*class="[^"]*BNeawe[^"]*"[^>]*>(.*?)</div>',
            r'<div class="VwiC3b[^"]*">(.*?)</div>'
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
            r'<li class="b_algo[^"]*">.*?<p[^>]*>(.*?)</p>'
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
            r'<div class="result__body[^"]*"[^>]*>(.*?)</div>',
            r'<td class="result-sn-abstract[^"]*">(.*?)</td>'
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
            
        # Score each result based on multiple factors
        scored_results = []
        for result in results:
            score = 0
            # Longer results are generally more informative
            score += len(result) / 100
            # Results with complete sentences are better
            score += len(re.findall(r'[.!?]', result)) * 5
            # Avoid results with too much markup/code
            score -= result.count('<') * 10
            # Prefer results with factual indicators
            factual_words = ['is', 'are', 'was', 'were', 'has', 'have', 'according', 'research', 'study']
            score += sum(2 for word in factual_words if word in result.lower())
            
            scored_results.append((score, result))
        
        # Return the highest-scoring result
        scored_results.sort(reverse=True)
        best_result = scored_results[0][1]
        
        # Truncate if too long
        if len(best_result) > 1500:
            best_result = best_result[:1500] + "..."
            
        return best_result

    def deep_research_engine(self, query):
        """Enhanced research with Wikipedia and other sources"""
        print(f"({self.name} performing deep research on: {query})")
        
        # Try Wikipedia API first
        wiki_result = self._search_wikipedia(query)
        if wiki_result:
            return wiki_result
        
        # Fall back to general search
        return self.google_search(query)

    def _search_wikipedia(self, query):
        """Search Wikipedia API"""
        try:
            # Wikipedia API endpoint
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=3"
            
            req = urllib.request.Request(api_url, headers={'User-Agent': 'KeyGenAI/1.0'})
            with urllib.request.urlopen(req, timeout=10, context=self.ssl_context) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('query', {}).get('search'):
                # Get the first result's page ID
                page_id = data['query']['search'][0]['pageid']
                
                # Get the full extract
                extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&pageids={page_id}&format=json"
                req = urllib.request.Request(extract_url, headers={'User-Agent': 'KeyGenAI/1.0'})
                with urllib.request.urlopen(req, timeout=10, context=self.ssl_context) as response:
                    extract_data = json.loads(response.read().decode('utf-8'))
                
                pages = extract_data.get('query', {}).get('pages', {})
                for pid, page_data in pages.items():
                    extract = page_data.get('extract', '')
                    if extract:
                        # Truncate if too long
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
            
        # Remove HTML tags
        clean = re.sub(r'<.*?>', '', text)
        
        # Remove common web noise
        noise_patterns = [
            r'(?i)click here', r'(?i)read more', r'(?i)cookies?', 
            r'(?i)privacy policy', r'(?i)subscribe', r'(?i)advertisement',
            r'(?i)accept cookies', r'(?i)terms of (use|service)',
            r'(?i)all rights reserved', r'(?i)copyright \d{4}'
        ]
        
        for pattern in noise_patterns:
            clean = re.sub(pattern, '', clean)
        
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Only save if it's substantial content
        if len(clean) > 50:
            try:
                with open(self.verified_web_file, 'a', encoding='utf-8') as f:
                    f.write(clean + "\n\n")
                self.raw_data_chunks.append(clean)
            except Exception as e:
                print(f"Error saving web data: {e}")
        
        return clean

    def get_detailed_answer(self, question, context=""):
        """Enhanced answer generation with better relevance scoring"""
        if not question:
            return None
            
        # Check knowledge base first
        best_match = None
        highest_score = 0
        
        # Search through knowledge base
        for sentence in self.raw_data_chunks:
            score = self.calculate_relevance_score(question, sentence)
            if score > highest_score:
                highest_score = score
                best_match = sentence
        
        # If good match found
        if best_match and highest_score > 0.3:
            answer = self.generate_detailed_response(question, best_match)
            return answer
        
        # Search online if no good match
        search_result = self.google_search(question)
        if search_result:
            return self.generate_detailed_response(question, search_result)
        
        return None

    def calculate_relevance_score(self, question, text):
        """Enhanced relevance scoring"""
        if not question or not text:
            return 0
            
        question_words = set(self.tokenize(question))
        text_words = set(self.tokenize(text))
        
        if not question_words:
            return 0
        
        # Calculate Jaccard similarity
        intersection = len(question_words.intersection(text_words))
        union = len(question_words.union(text_words))
        
        if union == 0:
            return 0
            
        return intersection / union

    def generate_detailed_response(self, question, content):
        """Generate well-structured responses"""
        if not content:
            return "I couldn't find specific information about that."
            
        question_lower = question.lower().strip()
        
        # Determine response format based on question type
        if question_lower.startswith(("what", "which")):
            response = f"Here's what I found:\n\n{content}"
        elif question_lower.startswith(("how", "why")):
            response = f"Let me explain:\n\n{content}"
        elif question_lower.startswith(("where", "when", "who")):
            response = f"Here are the details:\n\n{content}"
        elif question_lower.startswith("can"):
            response = f"Based on my research:\n\n{content}"
        else:
            response = f"Here's the information you requested:\n\n{content}"
        
        return response

    def learn_from_user(self, text):
        """Enhanced autonomous learning"""
        if not text:
            return False
            
        words = text.split()
        # Learn statements with factual indicators
        factual_patterns = [" is ", " was ", " are ", " were ", " has ", " have ", " will "]
        
        if len(words) > 8 and "?" not in text:
            if any(pattern in text.lower() for pattern in factual_patterns):
                try:
                    with open(self.user_mem_file, 'a', encoding='utf-8') as f:
                        f.write(text.strip() + ".\n")
                    self.raw_data_chunks.append(text.strip())
                    return True
                except Exception as e:
                    print(f"Learning error: {e}")
        return False

    def load_all_data(self):
        """Enhanced data loading with better error handling"""
        # Create directories if they don't exist
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
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
                    # Create empty files if they don't exist
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
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        
        if all_tokens:
            self.build_markov(all_tokens)
        
        print(f"--- {self.name} SYSTEM ONLINE (Enhanced Mode) ---")
        print(f"Knowledge base: {len(self.raw_data_chunks)} sentences loaded")

    def get_response(self, user_input):
        """Main response handler with enhanced processing"""
        if not user_input:
            return "Please type something and I'll help you!"
            
        raw_input = user_input.lower().strip()
        emotion_prefix = self.get_emotion_prefix(raw_input)
        tokens = self.tokenize(raw_input)
        
        # Autonomous learning
        self.learn_from_user(user_input)

        # Handle special commands
        if raw_input.startswith("learn about "):
            topic = raw_input.replace("learn about ", "").strip()
            result = self.deep_research_engine(topic)
            if result:
                result = self.rephraser(result, "clean")
                return self.grammar_checker(emotion_prefix + result)
            return "I couldn't find information about that topic."

        # Check if it's a question
        is_question = "?" in user_input or raw_input.startswith(
            ("what", "why", "how", "where", "when", "who", "which", "can", "is", "are", "do", "does")
        )
        
        if is_question:
            detailed_answer = self.get_detailed_answer(user_input)
            if detailed_answer:
                detailed_answer = self.rephraser(detailed_answer, "clean")
                return self.grammar_checker(emotion_prefix + detailed_answer)

        # Check fact engine
        for fact in self.gk_base:
            if fact.get("q", "").lower() in raw_input:
                result = self.rephraser(fact["a"], "clean")
                return self.grammar_checker(emotion_prefix + result)

        # Check knowledge modules
        for module in self.knowledge_base:
            for pattern in module.get("patterns", []):
                if pattern.lower() in raw_input:
                    result = random.choice(module["responses"])
                    result = self.rephraser(result, "clean")
                    return self.grammar_checker(emotion_prefix + result)

        # Handle pure emotional responses
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return self.grammar_checker(emotion_prefix)

        # Search local knowledge
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        if keywords:
            best_sentence = None
            max_overlap = 0
            for sentence in self.raw_data_chunks:
                overlap = sum(1 for kw in keywords if kw in sentence.lower())
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence
            
            if best_sentence and max_overlap >= 1:
                best_sentence = self.rephraser(best_sentence, "clean")
                return self.grammar_checker(emotion_prefix + best_sentence)

        # Internet search fallback
        if len(tokens) >= 2:
            search_result = self.google_search(user_input)
            if search_result:
                search_result = self.rephraser(search_result, "clean")
                return emotion_prefix + self.grammar_checker(search_result)

        # Final fallback
        return self.grammar_checker(
            emotion_prefix + 
            "I'm not sure about that. Try asking a specific question or use 'learn about [topic]' to help me learn!"
        )

# Web server handler
class ChatHandler(BaseHTTPRequestHandler):
    bot = None
    
    def do_GET(self):
        if self.path == '/':
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
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    #chat { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: scroll; margin-bottom: 10px; }
                    #input { width: 100%; padding: 10px; }
                </style>
            </head>
            <body>
                <h1>KeyGen.ai Assistant</h1>
                <div id="chat"></div>
                <input type="text" id="input" placeholder="Ask me anything...">
                <script>
                    document.getElementById('input').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            const message = this.value;
                            this.value = '';
                            document.getElementById('chat').innerHTML += '<p><b>You:</b> ' + message + '</p>';
                            fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            })
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('chat').innerHTML += '<p><b>AI:</b> ' + data.response + '</p>';
                                document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
                            });
                        }
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
    
    print(f"--- KeyGen.ai SYSTEM ONLINE (Enhanced Version) ---")
    print(f"Server listening on 0.0.0.0:{port}")
    print("Features: Multi-engine Search, Wikipedia API, Grammar Check, Rephraser, Cache")
    print("Access at: http://localhost:" + str(port))
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
