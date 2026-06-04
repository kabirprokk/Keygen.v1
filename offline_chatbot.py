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
                         "to", "at", "by", "for", "of", "with", "in", "on", "that", "this",
                         "it", "its", "be", "been", "being", "have", "has", "had", "do", "does",
                         "did", "will", "would", "could", "should", "may", "might", "can", "shall"}
        
        # Enhanced greetings database
        self.greetings = {
            "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                        "howdy", "greetings", "sup", "what's up", "yo", "hola", "bonjour",
                        "heya", "heyy", "hii", "helloo", "morning", "evening"],
            "responses": [
                "Hello! 👋 I'm {name}, your AI assistant. How can I help you today?",
                "Hi there! 😊 I'm {name}. What would you like to know?",
                "Hey! ✨ I'm {name}, ready to assist you. Ask me anything!",
                "Greetings! 🌟 I'm {name}. I can search the internet and answer questions. What can I do for you?",
                "Hello! 🚀 {name} at your service. How may I help you?",
                "Hi! Nice to meet you! 💫 I'm {name}. What can I help you with?",
                "Hey there! 🎯 {name} here. Feel free to ask me anything!",
                "Welcome! 🤖 I'm {name}, your intelligent assistant. How can I assist you today?"
            ],
            "follow_ups": [
                " What would you like to learn about?",
                " How can I assist you?",
                " Ask me anything!",
                " I can search the internet for you!",
                " Need help with something?",
                " What's on your mind today?"
            ]
        }
        
        self.emotions = {
            "happy": ["I'm delighted to see you're in a good mood! 😊", "That's wonderful news! 🎉", "I'm glad you're feeling positive! ✨"],
            "sad": ["I'm sorry you're feeling this way. I'm here to help. 💙", "I understand. Sometimes things are difficult. 🤗"],
            "angry": ["I hear you're frustrated. Let's try to resolve this together. 🤝", "I sense some tension. Let's work through this."],
            "lonely": ["I may be a program, but I am always here to talk. 💭", "You're not alone while I'm active. 🌟"]
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
        text_lower = text.lower().strip().rstrip('!.,? ')
        for pattern in self.greetings["patterns"]:
            if text_lower == pattern or text_lower.startswith(pattern):
                return True
        # Also check if it's just a short greeting-like message
        if len(text_lower.split()) <= 2 and any(g in text_lower for g in ["hi", "hey", "hello", "yo"]):
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
        if text[-1] not in ".!?:\"'" and not text.endswith("..."):
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
            r'<div class="kno-rdesc[^"]*">.*?<span[^>]*>(.*?)</span>',
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
        """Enhanced relevance scoring with multiple factors"""
        if not question or not text:
            return 0
            
        question_words = set(self.tokenize(question))
        text_words = set(self.tokenize(text))
        
        if not question_words:
            return 0
        
        # Jaccard similarity
        intersection = len(question_words.intersection(text_words))
        union = len(question_words.union(text_words))
        
        if union == 0:
            return 0
        
        jaccard_score = intersection / union
        
        # Bonus for exact phrase matches
        phrase_bonus = 0
        question_lower = question.lower()
        text_lower = text.lower()
        if question_lower in text_lower:
            phrase_bonus = 0.3
        
        # Bonus for keyword density
        keyword_density = intersection / len(text_words) if text_words else 0
        
        # Combined score
        final_score = jaccard_score * 0.5 + phrase_bonus * 0.3 + keyword_density * 0.2
        
        return final_score

    def search_local_knowledge(self, query):
        """Search local knowledge base and return best match with confidence score"""
        if not query:
            return None, 0
        
        tokens = self.tokenize(query)
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        if not keywords:
            return None, 0
        
        best_match = None
        highest_score = 0
        
        for sentence in self.raw_data_chunks:
            score = self.calculate_relevance_score(query, sentence)
            
            # Extra weight for sentences containing multiple keywords
            keyword_matches = sum(1 for kw in keywords if kw in sentence.lower())
            score += keyword_matches * 0.1
            
            if score > highest_score:
                highest_score = score
                best_match = sentence
        
        return best_match, highest_score

    def get_answer_with_fallback(self, question):
        """Smart answer retrieval: Check local first, then internet if needed"""
        if not question:
            return None, "unknown"
        
        # Step 1: Check GK Base (Fact Engine) - High priority
        for fact in self.gk_base:
            if fact.get("q", "").lower() in question.lower():
                return fact["a"], "knowledge_base"
        
        # Step 2: Check Knowledge Modules (JSON)
        for module in self.knowledge_base:
            for pattern in module.get("patterns", []):
                if pattern.lower() in question.lower():
                    return random.choice(module["responses"]), "knowledge_base"
        
        # Step 3: Search local text knowledge
        local_result, confidence = self.search_local_knowledge(question)
        
        # If local result is good enough (high confidence and substantial content)
        if local_result and confidence > 0.3 and len(local_result) > 100:
            print(f"✓ Found in local knowledge (confidence: {confidence:.2f})")
            return local_result, "local"
        
        # If local result exists but is poor quality
        if local_result and confidence <= 0.3:
            print(f"⚠ Local knowledge insufficient (confidence: {confidence:.2f}), searching internet...")
            web_result = self.google_search(question)
            if web_result:
                return web_result, "internet"
            # Fall back to poor local result if internet fails
            if len(local_result) > 50:
                return local_result + "\n\n(Note: This is from limited local knowledge. Internet search failed.)", "local_fallback"
        
        # Step 4: No good local result - Search internet
        print("📡 No local knowledge found, searching internet...")
        web_result = self.google_search(question)
        if web_result:
            return web_result, "internet"
        
        # Step 5: Try Wikipedia
        wiki_result = self._search_wikipedia(question)
        if wiki_result:
            return wiki_result, "wikipedia"
        
        return None, "unknown"

    def _format_answer(self, question, content, source):
        """Format answer based on question type and source"""
        question_lower = question.lower().strip()
        
        # Determine question type
        if question_lower.startswith("what"):
            prefix = "📚 Here's what I found"
        elif question_lower.startswith("why"):
            prefix = "💡 Here's the explanation"
        elif question_lower.startswith("how"):
            prefix = "🔧 Let me explain how this works"
        elif question_lower.startswith("where"):
            prefix = "📍 Here's the location information"
        elif question_lower.startswith("when"):
            prefix = "⏰ Here's the timeline"
        elif question_lower.startswith("who"):
            prefix = "👤 Here's who I found"
        elif question_lower.startswith(("which", "can", "is", "are", "do", "does")):
            prefix = "🔍 Based on my research"
        else:
            prefix = "📖 Here's what I found"
        
        # Format the response
        response = f"{prefix}:\n\n{content}"
        
        # Add source attribution
        if source == "internet":
            response += "\n\n🌐 (Source: Internet search)"
        elif source == "wikipedia":
            response += "\n\n📚 (Source: Wikipedia)"
        elif source == "local":
            response += "\n\n💾 (Source: Local knowledge base)"
        elif source == "knowledge_base":
            response += "\n\n📋 (Source: Knowledge base)"
        
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
        """Main response handler - OPTIMIZED for local-first, internet-fallback"""
        if not user_input or not user_input.strip():
            return "Please type something and I'll help you! 😊"
        
        raw_input = user_input.strip()
        raw_input_lower = raw_input.lower()
        
        # 1. CHECK FOR GREETINGS FIRST
        if self.is_greeting(raw_input):
            return self.get_greeting_response()
        
        # 2. Check for emotion
        emotion_prefix = self.get_emotion_prefix(raw_input_lower)
        
        # 3. Learn from user input
        self.learn_from_user(user_input)
        
        # 4. Handle "learn about" command
        if raw_input_lower.startswith("learn about "):
            topic = raw_input[12:].strip()
            result = self.google_search(topic)
            if result:
                result = self.rephraser(result, "clean")
                return self.grammar_checker(emotion_prefix + f"I learned about {topic}:\n\n{result}")
            return f"I couldn't find information about '{topic}'. Please try a different topic."
        
        # 5. Check if it's a question
        is_question = ("?" in raw_input or 
                      raw_input_lower.startswith(("what", "why", "how", "where", "when", "who", 
                                                   "which", "can", "is", "are", "do", "does",
                                                   "explain", "tell", "describe", "define")))
        
        if is_question:
            # SMART FALLBACK: Check local first, then internet
            answer, source = self.get_answer_with_fallback(raw_input)
            if answer:
                formatted = self._format_answer(raw_input, answer, source)
                formatted = self.rephraser(formatted, "clean")
                return self.grammar_checker(emotion_prefix + formatted)
            else:
                return "I searched everywhere but couldn't find a reliable answer. Could you rephrase your question? 🤔"
        
        # 6. For non-questions, try to find relevant information
        tokens = self.tokenize(raw_input_lower)
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        if keywords:
            # Try local knowledge first
            local_result, confidence = self.search_local_knowledge(raw_input)
            if local_result and confidence > 0.3 and len(local_result) > 100:
                local_result = self.rephraser(local_result, "clean")
                return self.grammar_checker(emotion_prefix + local_result)
            
            # If no good local result, try internet
            search_result = self.google_search(raw_input)
            if search_result and len(search_result) > 100:
                search_result = self.rephraser(search_result, "clean")
                return self.gram_checker(emotion_prefix + search_result)
        
        # 7. Pure emotion response
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return self.grammar_checker(emotion_prefix)
        
        # 8. Final fallback
        return "I'm not sure about that. Could you rephrase or ask a different question? You can also use 'learn about [topic]' to help me learn! 📚"


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
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>KeyGen.ai - AI Assistant</title>
                <style>
                    :root {
                        --primary: #6C63FF;
                        --primary-dark: #5A52D5;
                        --secondary: #FF6584;
                        --bg: #0f0f1a;
                        --surface: #1a1a2e;
                        --surface-light: #252540;
                        --text: #e0e0e0;
                        --text-secondary: #a0a0b0;
                        --border: #2a2a40;
                        --success: #4CAF50;
                        --gradient-1: linear-gradient(135deg, #6C63FF, #FF6584);
                        --gradient-2: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        --shadow: 0 10px 40px rgba(0,0,0,0.3);
                        --shadow-lg: 0 20px 60px rgba(0,0,0,0.4);
                    }
                    
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                        background: var(--bg);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 20px;
                        background-image: 
                            radial-gradient(ellipse at top left, rgba(108, 99, 255, 0.1), transparent 50%),
                            radial-gradient(ellipse at bottom right, rgba(255, 101, 132, 0.1), transparent 50%);
                    }
                    
                    .container {
                        background: var(--surface);
                        border-radius: 24px;
                        box-shadow: var(--shadow-lg);
                        max-width: 850px;
                        width: 100%;
                        overflow: hidden;
                        border: 1px solid var(--border);
                        backdrop-filter: blur(10px);
                    }
                    
                    .header {
                        background: var(--gradient-1);
                        padding: 24px 30px;
                        display: flex;
                        align-items: center;
                        gap: 16px;
                    }
                    
                    .header-icon {
                        width: 48px;
                        height: 48px;
                        background: rgba(255,255,255,0.2);
                        border-radius: 14px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 24px;
                        backdrop-filter: blur(10px);
                        animation: pulse 2s infinite;
                    }
                    
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); }
                        50% { transform: scale(1.05); }
                    }
                    
                    .header-text h1 {
                        color: white;
                        font-size: 22px;
                        font-weight: 700;
                        margin-bottom: 2px;
                    }
                    
                    .header-text p {
                        color: rgba(255,255,255,0.8);
                        font-size: 13px;
                    }
                    
                    .status-dot {
                        width: 8px;
                        height: 8px;
                        background: #4CAF50;
                        border-radius: 50%;
                        display: inline-block;
                        margin-right: 6px;
                        animation: glow 1.5s infinite;
                    }
                    
                    @keyframes glow {
                        0%, 100% { box-shadow: 0 0 5px #4CAF50; }
                        50% { box-shadow: 0 0 20px #4CAF50; }
                    }
                    
                    #chat-container {
                        height: 450px;
                        overflow-y: auto;
                        padding: 24px;
                        background: var(--surface);
                        scroll-behavior: smooth;
                    }
                    
                    #chat-container::-webkit-scrollbar {
                        width: 6px;
                    }
                    
                    #chat-container::-webkit-scrollbar-track {
                        background: transparent;
                    }
                    
                    #chat-container::-webkit-scrollbar-thumb {
                        background: var(--border);
                        border-radius: 3px;
                    }
                    
                    .message-wrapper {
                        display: flex;
                        margin-bottom: 20px;
                        animation: slideIn 0.3s ease-out;
                    }
                    
                    @keyframes slideIn {
                        from {
                            opacity: 0;
                            transform: translateY(10px);
                        }
                        to {
                            opacity: 1;
                            transform: translateY(0);
                        }
                    }
                    
                    .message-wrapper.user {
                        justify-content: flex-end;
                    }
                    
                    .message {
                        max-width: 75%;
                        padding: 14px 18px;
                        border-radius: 18px;
                        position: relative;
                        line-height: 1.5;
                        font-size: 15px;
                        word-wrap: break-word;
                        white-space: pre-wrap;
                    }
                    
                    .message-wrapper.user .message {
                        background: var(--gradient-1);
                        color: white;
                        border-bottom-right-radius: 4px;
                        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
                    }
                    
                    .message-wrapper.ai .message {
                        background: var(--surface-light);
                        color: var(--text);
                        border-bottom-left-radius: 4px;
                        border: 1px solid var(--border);
                    }
                    
                    .message-avatar {
                        width: 36px;
                        height: 36px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 18px;
                        flex-shrink: 0;
                        margin: 0 10px;
                    }
                    
                    .message-wrapper.ai .message-avatar {
                        background: var(--surface-light);
                    }
                    
                    .message-wrapper.user .message-avatar {
                        background: var(--primary);
                    }
                    
                    .typing-indicator {
                        display: flex;
                        align-items: center;
                        gap: 12px;
                        padding: 14px 18px;
                        background: var(--surface-light);
                        border-radius: 18px;
                        border-bottom-left-radius: 4px;
                        border: 1px solid var(--border);
                        max-width: 100px;
                    }
                    
                    .typing-dot {
                        width: 8px;
                        height: 8px;
                        background: var(--text-secondary);
                        border-radius: 50%;
                        animation: typing 1.4s infinite;
                    }
                    
                    .typing-dot:nth-child(2) {
                        animation-delay: 0.2s;
                    }
                    
                    .typing-dot:nth-child(3) {
                        animation-delay: 0.4s;
                    }
                    
                    @keyframes typing {
                        0%, 60%, 100% {
                            transform: translateY(0);
                            opacity: 0.4;
                        }
                        30% {
                            transform: translateY(-8px);
                            opacity: 1;
                        }
                    }
                    
                    .input-container {
                        padding: 20px 24px;
                        background: var(--surface);
                        border-top: 1px solid var(--border);
                        display: flex;
                        gap: 12px;
                        align-items: center;
                    }
                    
                    #input {
                        flex: 1;
                        padding: 14px 20px;
                        background: var(--surface-light);
                        border: 2px solid var(--border);
                        border-radius: 16px;
                        color: var(--text);
                        font-size: 15px;
                        outline: none;
                        transition: all 0.3s;
                    }
                    
                    #input:focus {
                        border-color: var(--primary);
                        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1);
                    }
                    
                    #input::placeholder {
                        color: var(--text-secondary);
                    }
                    
                    .send-btn {
                        width: 48px;
                        height: 48px;
                        background: var(--gradient-1);
                        border: none;
                        border-radius: 14px;
                        color: white;
                        font-size: 20px;
                        cursor: pointer;
                        transition: all 0.3s;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;
                    }
                    
                    .send-btn:hover {
                        transform: scale(1.05);
                        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
                    }
                    
                    .send-btn:active {
                        transform: scale(0.95);
                    }
                    
                    .suggestions {
                        display: flex;
                        gap: 8px;
                        padding: 0 24px 16px;
                        flex-wrap: wrap;
                    }
                    
                    .suggestion-chip {
                        padding: 8px 16px;
                        background: var(--surface-light);
                        border: 1px solid var(--border);
                        border-radius: 20px;
                        color: var(--text-secondary);
                        font-size: 13px;
                        cursor: pointer;
                        transition: all 0.2s;
                        white-space: nowrap;
                    }
                    
                    .suggestion-chip:hover {
                        background: var(--primary);
                        color: white;
                        border-color: var(--primary);
                    }
                    
                    .timestamp {
                        font-size: 11px;
                        color: var(--text-secondary);
                        margin-top: 4px;
                        padding: 0 10px;
                    }
                    
                    @media (max-width: 600px) {
                        .container {
                            border-radius: 0;
                            height: 100vh;
                        }
                        
                        #chat-container {
                            height: calc(100vh - 200px);
                        }
                        
                        .message {
                            max-width: 85%;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="header-icon">🤖</div>
                        <div class="header-text">
                            <h1>KeyGen.ai</h1>
                            <p><span class="status-dot"></span>Online - Ready to help</p>
                        </div>
                    </div>
                    
                    <div id="chat-container">
                        <div class="message-wrapper ai">
                            <div class="message-avatar">🤖</div>
                            <div>
                                <div class="message">
                                    Hello! 👋 I'm KeyGen.ai, your AI assistant. I can answer questions using my knowledge base and search the internet when needed. How can I help you today?
                                </div>
                                <div class="timestamp">Just now</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="suggestions">
                        <span class="suggestion-chip" onclick="useSuggestion(this)">What is artificial intelligence?</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">How does machine learning work?</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">Tell me about quantum computing</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">What is blockchain?</span>
                    </div>
                    
                    <div class="input-container">
                        <input type="text" id="input" placeholder="Type your message here..." autofocus>
                        <button class="send-btn" onclick="sendMessage()">➤</button>
                    </div>
                </div>
                
                <script>
                    const chatContainer = document.getElementById('chat-container');
                    const input = document.getElementById('input');
                    
                    function getTime() {
                        const now = new Date();
                        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    }
                    
                    function addMessage(text, isUser) {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'message-wrapper ' + (isUser ? 'user' : 'ai');
                        
                        const avatar = document.createElement('div');
                        avatar.className = 'message-avatar';
                        avatar.textContent = isUser ? '👤' : '🤖';
                        
                        const messageContainer = document.createElement('div');
                        const message = document.createElement('div');
                        message.className = 'message';
                        message.textContent = text;
                        
                        const timestamp = document.createElement('div');
                        timestamp.className = 'timestamp';
                        timestamp.textContent = getTime();
                        
                        messageContainer.appendChild(message);
                        messageContainer.appendChild(timestamp);
                        
                        if (isUser) {
                            wrapper.appendChild(messageContainer);
                            wrapper.appendChild(avatar);
                        } else {
                            wrapper.appendChild(avatar);
                            wrapper.appendChild(messageContainer);
                        }
                        
                        chatContainer.appendChild(wrapper);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                        
                        return message;
                    }
                    
                    function showTypingIndicator() {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'message-wrapper ai';
                        wrapper.id = 'typing-wrapper';
                        
                        const avatar = document.createElement('div');
                        avatar.className = 'message-avatar';
                        avatar.textContent = '🤖';
                        
                        const indicator = document.createElement('div');
                        indicator.className = 'typing-indicator';
                        indicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
                        
                        wrapper.appendChild(avatar);
                        wrapper.appendChild(indicator);
                        chatContainer.appendChild(wrapper);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                    
                    function removeTypingIndicator() {
                        const typing = document.getElementById('typing-wrapper');
                        if (typing) typing.remove();
                    }
                    
                    async function typeWriterEffect(element, text, speed = 20) {
                        element.textContent = '';
                        for (let i = 0; i < text.length; i++) {
                            element.textContent += text.charAt(i);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                            await new Promise(resolve => setTimeout(resolve, speed));
                        }
                    }
                    
                    async function sendMessage() {
                        const message = input.value.trim();
                        if (!message) return;
                        
                        // Add user message
                        addMessage(message, true);
                        input.value = '';
                        
                        // Show typing indicator
                        showTypingIndicator();
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message})
                            });
                            const data = await response.json();
                            
                            // Remove typing indicator
                            removeTypingIndicator();
                            
                            // Add AI message with typing animation
                            const aiMessage = addMessage('', false);
                            await typeWriterEffect(aiMessage, data.response, 15);
                            
                        } catch (error) {
                            removeTypingIndicator();
                            addMessage('⚠️ Error connecting to server. Please try again.', false);
                        }
                    }
                    
                    function useSuggestion(chip) {
                        input.value = chip.textContent;
                        sendMessage();
                    }
                    
                    // Handle Enter key
                    input.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });
                    
                    // Focus input on load
                    input.focus();
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
                response_text = f"⚠️ Error processing your request: {str(e)}"
            
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
    
    print(f"""
{'='*60}
🤖 KeyGen.ai SYSTEM ONLINE
{'='*60}
🌐 Server: http://0.0.0.0:{port}
📚 Features: 
   • Local Knowledge Base (Priority)
   • Internet Search (Google, Bing, DuckDuckGo)
   • Wikipedia API
   • Smart Fallback System
   • Typing Animation
   • Modern Dark UI
💡 How it works:
   1. Checks local knowledge first
   2. If confidence < 30%, searches internet
   3. Falls back to Wikipedia if needed
{'='*60}
    """)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
