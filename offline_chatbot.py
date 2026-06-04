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
        
        self.greetings = {
            "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                        "howdy", "greetings", "sup", "what's up", "yo", "hola", "bonjour",
                        "heya", "heyy", "hii", "helloo", "morning", "evening"],
            "responses": [
                "Hello! 👋 How can I help you today?",
                "Hi there! 😊 What would you like to know?",
                "Hey! ✨ Ask me anything!",
                "Greetings! 🌟 How can I assist you?",
                "Hello! 🚀 What can I help you with?",
                "Hi! 💫 What's on your mind?",
                "Hey there! 🎯 Feel free to ask me anything!",
                "Welcome! 🤖 How can I help?"
            ]
        }
        
        self.emotions = {
            "happy": ["Glad you're feeling good! 😊", "That's wonderful! 🎉"],
            "sad": ["I'm here to help. 💙", "I understand. 🤗"],
            "angry": ["Let's work through this together. 🤝", "I hear you."],
            "lonely": ["I'm always here to talk. 💭", "You're not alone. 🌟"]
        }
        
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        os.makedirs(self.knowledge_dir, exist_ok=True)
        self.load_all_data()
        self.load_search_cache()

    def load_search_cache(self):
        try:
            if os.path.exists(self.search_cache_file):
                with open(self.search_cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
        except:
            self.search_cache = {}

    def save_search_cache(self):
        try:
            if len(self.search_cache) > 100:
                keys = list(self.search_cache.keys())[-100:]
                self.search_cache = {k: self.search_cache[k] for k in keys}
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, indent=2)
        except:
            pass

    def tokenize(self, text):
        if not text:
            return []
        return re.findall(r'\b\w+\b', str(text).lower())

    def build_markov(self, tokens):
        if not tokens or len(tokens) < 2:
            return
        for i in range(len(tokens) - 1):
            if tokens[i] and tokens[i+1]:
                self.markov_graph[tokens[i]].append(tokens[i+1])

    def is_greeting(self, text):
        text_lower = text.lower().strip().rstrip('!.,? ')
        if len(text_lower.split()) <= 2 and any(g in text_lower for g in ["hi", "hey", "hello", "yo"]):
            return True
        for pattern in self.greetings["patterns"]:
            if text_lower == pattern or text_lower.startswith(pattern):
                return True
        return False

    def get_greeting_response(self):
        return random.choice(self.greetings["responses"])

    def get_emotion_prefix(self, text):
        if not text:
            return ""
        text_lower = text.lower()
        for emotion, responses in self.emotions.items():
            if emotion in text_lower:
                return random.choice(responses) + " "
        return ""

    def grammar_checker(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = text.strip()
        if not text:
            return ""
        if len(text) > 1:
            text = text[0].upper() + text[1:]
        if text[-1] not in ".!?:\"'":
            text += "."
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\bi\b(?![\'\.])', 'I', text)
        return text

    def summarize_text(self, text, max_sentences=3):
        """Summarize text to key points only"""
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # If already short, return as is
        if len(sentences) <= max_sentences:
            return text
        
        # Score sentences by relevance (keyword density)
        words = self.tokenize(text)
        keywords = [w for w in words if w not in self.stopwords and len(w) > 3]
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
            scored_sentences.append((score, sentence))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in scored_sentences[:max_sentences]]
        
        # Keep original order
        ordered = [s for s in sentences if s in top_sentences]
        
        return ' '.join(ordered) if ordered else ' '.join(sentences[:max_sentences])

    def truncate_answer(self, text, max_chars=500):
        """Truncate long answers"""
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        
        if last_period > max_chars * 0.5:
            return truncated[:last_period + 1]
        else:
            return truncated.rsplit(' ', 1)[0] + "..."

    def make_http_request(self, url, timeout=10):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=self.ssl_context) as response:
                return response.read().decode('utf-8', errors='ignore')
        except:
            return None

    def google_search(self, query):
        if not query:
            return None
        
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 3600:
                return cache_entry['data']
        
        engines = [
            {
                "name": "Google",
                "url": f"https://www.google.com/search?q={urllib.parse.quote(query)}",
                "parser": self._parse_generic
            },
            {
                "name": "DuckDuckGo",
                "url": f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}",
                "parser": self._parse_generic
            }
        ]
        
        for engine in engines:
            try:
                html = self.make_http_request(engine['url'], timeout=8)
                if html:
                    results = engine['parser'](html)
                    if results:
                        best = self._select_best_result(results)
                        if best and len(best) > 50:
                            self.search_cache[cache_key] = {'data': best, 'timestamp': time.time()}
                            self.save_search_cache()
                            return self.polish_and_save_web_data(best)
            except:
                continue
        
        return self._search_wikipedia(query)

    def _parse_generic(self, html):
        results = []
        patterns = [
            r'<p[^>]*>(.*?)</p>',
            r'<div[^>]*class="[^"]*(?:result|snippet|abstract)[^"]*"[^>]*>(.*?)</div>',
            r'<span[^>]*class="[^"]*(?:st|snippet)[^"]*"[^>]*>(.*?)</span>',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, html, re.DOTALL)
            for match in matches:
                clean = re.sub(r'<.*?>', '', match)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if 50 < len(clean) < 2000:
                    results.append(clean)
        return results

    def _select_best_result(self, results):
        if not results:
            return None
        scored = []
        for r in results:
            score = len(r) / 100 + len(re.findall(r'[.!?]', r)) * 3
            scored.append((score, r))
        scored.sort(reverse=True)
        return scored[0][1] if scored else None

    def _search_wikipedia(self, query):
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
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
                        return self.polish_and_save_web_data(extract[:1000])
        except:
            pass
        return None

    def polish_and_save_web_data(self, text):
        if not text:
            return text
        clean = re.sub(r'<.*?>', '', text)
        noise = [r'(?i)click here', r'(?i)read more', r'(?i)cookies?', r'(?i)privacy policy']
        for n in noise:
            clean = re.sub(n, '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) > 50:
            try:
                with open(self.verified_web_file, 'a', encoding='utf-8') as f:
                    f.write(clean + "\n\n")
                self.raw_data_chunks.append(clean)
            except:
                pass
        return clean

    def calculate_relevance_score(self, question, text):
        if not question or not text:
            return 0
        q_words = set(self.tokenize(question))
        t_words = set(self.tokenize(text))
        if not q_words:
            return 0
        intersection = len(q_words.intersection(t_words))
        union = len(q_words.union(t_words))
        return intersection / union if union > 0 else 0

    def search_local_knowledge(self, query):
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
            keyword_matches = sum(1 for kw in keywords if kw in sentence.lower())
            score += keyword_matches * 0.1
            if score > highest_score:
                highest_score = score
                best_match = sentence
        
        return best_match, highest_score

    def get_answer_with_fallback(self, question):
        if not question:
            return None, "unknown"
        
        # Check GK Base
        for fact in self.gk_base:
            if fact.get("q", "").lower() in question.lower():
                return self.truncate_answer(fact["a"]), "knowledge_base"
        
        # Check Knowledge Modules
        for module in self.knowledge_base:
            for pattern in module.get("patterns", []):
                if pattern.lower() in question.lower():
                    return self.truncate_answer(random.choice(module["responses"])), "knowledge_base"
        
        # Check local knowledge
        local_result, confidence = self.search_local_knowledge(question)
        if local_result and confidence > 0.3 and len(local_result) > 50:
            return self.truncate_answer(local_result), "local"
        
        # Search internet
        web_result = self.google_search(question)
        if web_result:
            # Summarize and truncate
            summarized = self.summarize_text(web_result, max_sentences=3)
            return self.truncate_answer(summarized), "internet"
        
        # Wikipedia fallback
        wiki_result = self._search_wikipedia(question)
        if wiki_result:
            summarized = self.summarize_text(wiki_result, max_sentences=3)
            return self.truncate_answer(summarized), "wikipedia"
        
        return None, "unknown"

    def _format_answer(self, question, content, source):
        question_lower = question.lower().strip()
        
        # Shorter, more direct prefixes
        if question_lower.startswith("what"):
            prefix = ""
        elif question_lower.startswith("why"):
            prefix = "Because "
        elif question_lower.startswith("how"):
            prefix = ""
        elif question_lower.startswith("where"):
            prefix = ""
        elif question_lower.startswith("when"):
            prefix = ""
        elif question_lower.startswith("who"):
            prefix = ""
        else:
            prefix = ""
        
        # Don't add source attribution to keep it concise
        return f"{prefix}{content}"

    def learn_from_user(self, text):
        if not text or len(text.split()) < 8 or "?" in text:
            return False
        factual_patterns = [" is ", " was ", " are ", " were ", " has ", " have "]
        if any(pattern in text.lower() for pattern in factual_patterns):
            try:
                with open(self.user_mem_file, 'a', encoding='utf-8') as f:
                    f.write(text.strip() + ".\n")
                self.raw_data_chunks.append(text.strip())
                return True
            except:
                pass
        return False

    def load_all_data(self):
        json_files = [(self.data_file, 'knowledge_base'), (self.gk_file, 'gk_base')]
        for file_path, attr_name in json_files:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        setattr(self, attr_name, json.load(f))
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump([], f)
                    setattr(self, attr_name, [])
            except:
                setattr(self, attr_name, [])
        
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
                    except:
                        pass
        if all_tokens:
            self.build_markov(all_tokens)

    def get_response(self, user_input):
        if not user_input or not user_input.strip():
            return "Please ask me something! 😊"
        
        raw_input = user_input.strip()
        raw_input_lower = raw_input.lower()
        
        # Greetings
        if self.is_greeting(raw_input):
            return self.get_greeting_response()
        
        # Emotion
        emotion_prefix = self.get_emotion_prefix(raw_input_lower)
        
        # Learn
        self.learn_from_user(user_input)
        
        # Learn about command
        if raw_input_lower.startswith("learn about "):
            topic = raw_input[12:].strip()
            result = self.google_search(topic)
            if result:
                summarized = self.summarize_text(result, max_sentences=2)
                return self.truncate_answer(f"Learned about {topic}: {summarized}", 400)
            return f"Couldn't find information about '{topic}'."
        
        # Questions
        is_question = ("?" in raw_input or 
                      raw_input_lower.startswith(("what", "why", "how", "where", "when", "who", 
                                                   "which", "can", "is", "are", "do", "does",
                                                   "explain", "tell", "describe", "define")))
        
        if is_question:
            answer, source = self.get_answer_with_fallback(raw_input)
            if answer:
                formatted = self._format_answer(raw_input, answer, source)
                return self.truncate_answer(formatted, 500)
            return "I couldn't find a reliable answer. Try rephrasing your question."
        
        # Non-questions
        tokens = self.tokenize(raw_input_lower)
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        if keywords:
            local_result, confidence = self.search_local_knowledge(raw_input)
            if local_result and confidence > 0.3 and len(local_result) > 50:
                return self.truncate_answer(local_result, 400)
            
            search_result = self.google_search(raw_input)
            if search_result:
                summarized = self.summarize_text(search_result, max_sentences=2)
                return self.truncate_answer(summarized, 400)
        
        # Emotion only
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return emotion_prefix
        
        return "I'm not sure about that. Could you rephrase your question?"


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
                <title>KeyGen.ai</title>
                <style>
                    :root {
                        --bg: #000000;
                        --surface: #0a0a0a;
                        --surface2: #111111;
                        --border: #1a1a1a;
                        --text: #ffffff;
                        --text-secondary: #888888;
                        --glow: #ffffff;
                        --accent: #ffffff;
                    }
                    
                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }
                    
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background: var(--bg);
                        min-height: 100vh;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 16px;
                    }
                    
                    .container {
                        background: var(--surface);
                        border-radius: 20px;
                        max-width: 750px;
                        width: 100%;
                        overflow: hidden;
                        border: 1px solid var(--border);
                        box-shadow: 0 0 30px rgba(255,255,255,0.03), 0 0 60px rgba(255,255,255,0.01);
                    }
                    
                    .header {
                        padding: 20px 24px;
                        display: flex;
                        align-items: center;
                        gap: 14px;
                        border-bottom: 1px solid var(--border);
                        background: var(--surface2);
                    }
                    
                    .header-icon {
                        width: 42px;
                        height: 42px;
                        background: var(--bg);
                        border-radius: 12px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 22px;
                        border: 1px solid var(--border);
                        box-shadow: 0 0 15px rgba(255,255,255,0.05);
                    }
                    
                    .header-text h1 {
                        color: var(--text);
                        font-size: 18px;
                        font-weight: 600;
                        letter-spacing: -0.3px;
                    }
                    
                    .header-text p {
                        color: var(--text-secondary);
                        font-size: 12px;
                    }
                    
                    .status-dot {
                        width: 6px;
                        height: 6px;
                        background: var(--glow);
                        border-radius: 50%;
                        display: inline-block;
                        margin-right: 6px;
                        box-shadow: 0 0 8px var(--glow);
                        animation: glow 2s infinite;
                    }
                    
                    @keyframes glow {
                        0%, 100% { box-shadow: 0 0 8px var(--glow); }
                        50% { box-shadow: 0 0 16px var(--glow); }
                    }
                    
                    #chat-container {
                        height: 420px;
                        overflow-y: auto;
                        padding: 20px;
                        background: var(--surface);
                        scroll-behavior: smooth;
                    }
                    
                    #chat-container::-webkit-scrollbar {
                        width: 4px;
                    }
                    
                    #chat-container::-webkit-scrollbar-track {
                        background: transparent;
                    }
                    
                    #chat-container::-webkit-scrollbar-thumb {
                        background: var(--border);
                        border-radius: 2px;
                    }
                    
                    .message-wrapper {
                        display: flex;
                        margin-bottom: 16px;
                        animation: slideIn 0.25s ease-out;
                    }
                    
                    @keyframes slideIn {
                        from { opacity: 0; transform: translateY(8px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                    
                    .message-wrapper.user {
                        justify-content: flex-end;
                    }
                    
                    .message {
                        max-width: 78%;
                        padding: 12px 16px;
                        border-radius: 16px;
                        position: relative;
                        line-height: 1.45;
                        font-size: 14px;
                        word-wrap: break-word;
                        white-space: pre-wrap;
                    }
                    
                    .message-wrapper.user .message {
                        background: var(--text);
                        color: var(--bg);
                        border-bottom-right-radius: 4px;
                        font-weight: 500;
                    }
                    
                    .message-wrapper.ai .message {
                        background: var(--surface2);
                        color: var(--text);
                        border-bottom-left-radius: 4px;
                        border: 1px solid var(--border);
                    }
                    
                    .message-avatar {
                        width: 32px;
                        height: 32px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 16px;
                        flex-shrink: 0;
                        margin: 0 8px;
                    }
                    
                    .message-wrapper.ai .message-avatar {
                        background: var(--surface2);
                        border: 1px solid var(--border);
                    }
                    
                    .message-wrapper.user .message-avatar {
                        background: var(--text);
                        color: var(--bg);
                    }
                    
                    .typing-indicator {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding: 12px 16px;
                        background: var(--surface2);
                        border-radius: 16px;
                        border-bottom-left-radius: 4px;
                        border: 1px solid var(--border);
                        max-width: 80px;
                    }
                    
                    .typing-dot {
                        width: 6px;
                        height: 6px;
                        background: var(--text-secondary);
                        border-radius: 50%;
                        animation: typing 1.4s infinite;
                    }
                    
                    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
                    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
                    
                    @keyframes typing {
                        0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
                        30% { transform: translateY(-6px); opacity: 1; }
                    }
                    
                    .input-container {
                        padding: 16px 20px;
                        background: var(--surface2);
                        border-top: 1px solid var(--border);
                        display: flex;
                        gap: 10px;
                        align-items: center;
                    }
                    
                    #input {
                        flex: 1;
                        padding: 12px 16px;
                        background: var(--surface);
                        border: 1px solid var(--border);
                        border-radius: 14px;
                        color: var(--text);
                        font-size: 14px;
                        outline: none;
                        transition: all 0.2s;
                    }
                    
                    #input:focus {
                        border-color: var(--text);
                        box-shadow: 0 0 0 2px rgba(255,255,255,0.05);
                    }
                    
                    #input::placeholder {
                        color: #444;
                    }
                    
                    .btn {
                        height: 42px;
                        border: 1px solid var(--border);
                        border-radius: 12px;
                        color: var(--text);
                        font-size: 14px;
                        cursor: pointer;
                        transition: all 0.2s;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        flex-shrink: 0;
                        background: var(--surface);
                    }
                    
                    .send-btn {
                        width: 42px;
                        font-size: 18px;
                    }
                    
                    .send-btn:hover {
                        background: var(--text);
                        color: var(--bg);
                        border-color: var(--text);
                    }
                    
                    .stop-btn {
                        width: 42px;
                        font-size: 16px;
                        display: none;
                    }
                    
                    .stop-btn:hover {
                        background: #ff3333;
                        border-color: #ff3333;
                        color: white;
                    }
                    
                    .stop-btn.active {
                        display: flex;
                    }
                    
                    .send-btn.hidden {
                        display: none;
                    }
                    
                    .suggestions {
                        display: flex;
                        gap: 8px;
                        padding: 12px 20px;
                        flex-wrap: wrap;
                        background: var(--surface);
                    }
                    
                    .suggestion-chip {
                        padding: 7px 14px;
                        background: var(--surface2);
                        border: 1px solid var(--border);
                        border-radius: 20px;
                        color: var(--text-secondary);
                        font-size: 12px;
                        cursor: pointer;
                        transition: all 0.2s;
                        white-space: nowrap;
                    }
                    
                    .suggestion-chip:hover {
                        background: var(--text);
                        color: var(--bg);
                        border-color: var(--text);
                    }
                    
                    .timestamp {
                        font-size: 10px;
                        color: #444;
                        margin-top: 4px;
                        padding: 0 8px;
                    }
                    
                    @media (max-width: 600px) {
                        body { padding: 0; }
                        .container { border-radius: 0; height: 100vh; display: flex; flex-direction: column; }
                        #chat-container { flex: 1; height: auto; }
                        .message { max-width: 85%; }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div class="header-icon">🤖</div>
                        <div class="header-text">
                            <h1>KeyGen.ai</h1>
                            <p><span class="status-dot"></span>Online</p>
                        </div>
                    </div>
                    
                    <div id="chat-container">
                        <div class="message-wrapper ai">
                            <div class="message-avatar">🤖</div>
                            <div>
                                <div class="message">Hello! 👋 I'm KeyGen.ai. Ask me anything!</div>
                                <div class="timestamp">Just now</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="suggestions">
                        <span class="suggestion-chip" onclick="useSuggestion(this)">What is AI?</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">How does ML work?</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">Quantum computing</span>
                        <span class="suggestion-chip" onclick="useSuggestion(this)">What is blockchain?</span>
                    </div>
                    
                    <div class="input-container">
                        <input type="text" id="input" placeholder="Ask anything..." autofocus>
                        <button class="btn send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
                        <button class="btn stop-btn" id="stopBtn" onclick="stopGeneration()">■</button>
                    </div>
                </div>
                
                <script>
                    const chatContainer = document.getElementById('chat-container');
                    const input = document.getElementById('input');
                    const sendBtn = document.getElementById('sendBtn');
                    const stopBtn = document.getElementById('stopBtn');
                    
                    let isGenerating = false;
                    let abortController = null;
                    
                    function getTime() {
                        return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    }
                    
                    function addMessage(text, isUser) {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'message-wrapper ' + (isUser ? 'user' : 'ai');
                        
                        const avatar = document.createElement('div');
                        avatar.className = 'message-avatar';
                        avatar.textContent = isUser ? '👤' : '🤖';
                        
                        const container = document.createElement('div');
                        const message = document.createElement('div');
                        message.className = 'message';
                        message.textContent = text;
                        
                        const timestamp = document.createElement('div');
                        timestamp.className = 'timestamp';
                        timestamp.textContent = getTime();
                        
                        container.appendChild(message);
                        container.appendChild(timestamp);
                        
                        if (isUser) {
                            wrapper.appendChild(container);
                            wrapper.appendChild(avatar);
                        } else {
                            wrapper.appendChild(avatar);
                            wrapper.appendChild(container);
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
                    
                    function setGeneratingState(generating) {
                        isGenerating = generating;
                        if (generating) {
                            sendBtn.classList.add('hidden');
                            stopBtn.classList.add('active');
                            input.disabled = true;
                        } else {
                            sendBtn.classList.remove('hidden');
                            stopBtn.classList.remove('active');
                            input.disabled = false;
                            input.focus();
                        }
                    }
                    
                    function stopGeneration() {
                        if (abortController) {
                            abortController.abort();
                            abortController = null;
                        }
                        isGenerating = false;
                        removeTypingIndicator();
                        setGeneratingState(false);
                    }
                    
                    async function typeWriterEffect(element, text, speed = 12) {
                        element.textContent = '';
                        for (let i = 0; i < text.length; i++) {
                            if (!isGenerating) break;
                            element.textContent += text.charAt(i);
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                            await new Promise(resolve => setTimeout(resolve, speed));
                        }
                    }
                    
                    async function sendMessage() {
                        const message = input.value.trim();
                        if (!message || isGenerating) return;
                        
                        addMessage(message, true);
                        input.value = '';
                        showTypingIndicator();
                        setGeneratingState(true);
                        
                        abortController = new AbortController();
                        
                        try {
                            const response = await fetch('/chat', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message: message}),
                                signal: abortController.signal
                            });
                            const data = await response.json();
                            
                            removeTypingIndicator();
                            
                            if (isGenerating) {
                                const aiMessage = addMessage('', false);
                                await typeWriterEffect(aiMessage, data.response, 12);
                            }
                        } catch (error) {
                            if (error.name !== 'AbortError') {
                                removeTypingIndicator();
                                addMessage('⚠️ Connection error. Try again.', false);
                            }
                        }
                        
                        setGeneratingState(false);
                        abortController = null;
                    }
                    
                    function useSuggestion(chip) {
                        input.value = chip.textContent;
                        sendMessage();
                    }
                    
                    input.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            sendMessage();
                        }
                    });
                    
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
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data)
                user_msg = data.get('message', '')
                response_text = self.bot.get_response(user_msg)
            except Exception as e:
                response_text = f"Error: {str(e)}"
            
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
╔══════════════════════════════════════╗
║       🤖 KeyGen.ai ONLINE           ║
║   http://0.0.0.0:{port}              ║
║   Deep Black + White Glow Theme     ║
║   Concise Answers + Stop Button     ║
╚══════════════════════════════════════╝
    """)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
