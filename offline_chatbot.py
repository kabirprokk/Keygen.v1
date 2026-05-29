import re
import random
import json
import os
import urllib.request
import urllib.parse
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from textblob import TextBlob

class KeyGenAI:
    def __init__(self, knowledge_dir="knowledge", data_file="data.json", gk_file="gk_knowledge.json"):
        self.name = "KeyGen.ai"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.knowledge_dir = os.path.join(self.script_dir, knowledge_dir)
        self.data_file = os.path.join(self.script_dir, data_file)
        self.gk_file = os.path.join(self.script_dir, gk_file)
        
        # New Persistent Memory Files
        self.user_mem_file = os.path.join(self.knowledge_dir, "user_mem.txt")
        self.verified_web_file = os.path.join(self.knowledge_dir, "verified_web.txt")
        
        self.raw_data_chunks = []
        self.markov_graph = defaultdict(list)
        self.knowledge_base = []
        self.gk_base = []
        self.stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "to", "at", "by", "for", "of", "with"}
        
        self.emotions = {
            "happy": ["I'm delighted to see you're in a good mood!", "That's wonderful news!", "I'm glad you're feeling positive!"],
            "sad": ["I'm sorry you're feeling this way. I'm here to help.", "I understand. Sometimes things are difficult."],
            "angry": ["I hear you're frustrated. Let's try to resolve this together.", "I sense some tension."],
            "lonely": ["I may be a program, but I am always here to talk.", "You're not alone while I'm active."]
        }
        
        self.load_all_data()

    def tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())

    def build_markov(self, tokens):
        for i in range(len(tokens) - 1):
            self.markov_graph[tokens[i]].append(tokens[i+1])

    def get_emotion_prefix(self, text):
        for emotion, responses in self.emotions.items():
            if emotion in text.lower():
                return random.choice(responses) + " "
        return ""

    def grammar_checker(self, text):
        """Basic grammar checking without external dependencies"""
        if not text:
            return ""
        
        text = text.strip()
        if not text:
            return ""
        
        # Basic punctuation and capitalization
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Ensure proper ending punctuation
        if text[-1] not in ".!?":
            text += "."
        
        return text

    def rephraser(self, text, style="clean"):
        """Rephrases text in different styles: clean, professional, simple, or creative"""
        if not text:
            return ""
        
        blob = TextBlob(text)
        
        if style == "clean":
            # Remove redundant words and simplify
            words = text.split()
            cleaned = []
            for i, word in enumerate(words):
                if word.lower() not in self.stopwords or (i > 0 and i < len(words)-1):
                    cleaned.append(word)
            result = " ".join(cleaned)
            
        elif style == "professional":
            # Make it more formal
            result = text.replace("I think", "Based on analysis")
            result = result.replace("maybe", "potentially")
            result = result.replace("a lot", "significantly")
            
        elif style == "simple":
            # Make it easier to understand
            sentences = re.split(r'(?<=[.!?])\s+', text)
            simple_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 20:
                    words = sentence.split()
                    chunk_size = 15
                    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
                    simple_sentences.extend(chunks)
                else:
                    simple_sentences.append(sentence)
            result = ". ".join(simple_sentences)
            
        else:  # creative
            # Add variety and flair
            result = text
            creative_phrases = ["Interestingly, ", "Notably, ", "Furthermore, ", "In addition, "]
            if len(result.split()) > 5:
                insert_pos = len(result) // 3
                result = result[:insert_pos] + random.choice(creative_phrases) + result[insert_pos:]
        
        return self.grammar_checker(result)

    def google_search(self, query):
        """Advanced Google search with fallback mechanisms"""
        print(f"({self.name} is searching Google for accurate information...)")
        
        # Clean the query
        clean_query = query.strip()
        
        # List of search URLs with fallbacks
        search_urls = [
            f"https://www.google.com/search?q={urllib.parse.quote(clean_query)}",
            f"https://www.bing.com/search?q={urllib.parse.quote(clean_query)}",
            f"https://duckduckgo.com/html/?q={urllib.parse.quote(clean_query)}"
        ]
        
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        
        for url in search_urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': user_agent})
                with urllib.request.urlopen(req, timeout=5) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                
                # Extract meaningful content
                # Look for paragraphs and text blocks
                text_blocks = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
                text_blocks.extend(re.findall(r'<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL))
                
                # Clean the extracted text
                cleaned_blocks = []
                for block in text_blocks:
                    # Remove HTML tags
                    clean = re.sub(r'<.*?>', '', block)
                    # Remove extra whitespace
                    clean = re.sub(r'\s+', ' ', clean)
                    clean = clean.strip()
                    
                    # Filter out short or irrelevant blocks
                    if len(clean) > 100 and "cookie" not in clean.lower() and "privacy" not in clean.lower():
                        cleaned_blocks.append(clean)
                
                if cleaned_blocks:
                    # Get the longest, most relevant block
                    best_result = max(cleaned_blocks, key=len)
                    
                    # Limit length for reasonable response
                    if len(best_result) > 1000:
                        best_result = best_result[:1000] + "..."
                    
                    # Polish and save to memory
                    polished = self.polish_and_save_web_data(best_result)
                    return polished
                    
            except Exception as e:
                print(f"Search error with {url}: {e}")
                continue
        
        return None

    def get_detailed_answer(self, question, context=""):
        """Generates detailed, accurate answers based on the question"""
        
        # Check knowledge base first
        best_match = None
        highest_score = 0
        
        # Search through knowledge base for relevant content
        for sentence in self.raw_data_chunks:
            score = self.calculate_relevance_score(question, sentence)
            if score > highest_score:
                highest_score = score
                best_match = sentence
        
        # If good match found in knowledge base
        if best_match and highest_score > 0.3:
            # Expand the answer with context
            words = best_match.split()
            if len(words) < 30:
                # Find surrounding context
                for s in self.raw_data_chunks:
                    if best_match in s and len(s.split()) > len(words):
                        best_match = s
                        break
            
            answer = self.generate_detailed_response(question, best_match)
            return answer
        
        # Search online if no good match
        search_result = self.google_search(question)
        if search_result:
            return self.generate_detailed_response(question, search_result)
        
        return None

    def calculate_relevance_score(self, question, text):
        """Calculate how relevant a text is to the question"""
        question_words = set(self.tokenize(question))
        text_words = set(self.tokenize(text))
        
        if not question_words:
            return 0
        
        overlap = len(question_words.intersection(text_words))
        return overlap / len(question_words)

    def generate_detailed_response(self, question, content):
        """Generate a detailed, well-structured response"""
        
        # Tokenize question to understand what's being asked
        question_lower = question.lower()
        
        # Determine question type
        if question_lower.startswith(("what", "which")):
            response = f"Based on available information, here's what I found:\n\n{content}"
        elif question_lower.startswith(("how", "why")):
            response = f"Let me explain this in detail:\n\n{content}"
        elif question_lower.startswith(("where", "when", "who")):
            response = f"Here are the specific details:\n\n{content}"
        else:
            response = f"Here's the information you requested:\n\n{content}"
        
        # Add additional context if available
        if len(content.split()) < 100:
            additional = self.find_additional_context(content)
            if additional:
                response += f"\n\nAdditionally: {additional}"
        
        return response

    def find_additional_context(self, content):
        """Find additional context related to the content"""
        keywords = self.tokenize(content)[:5]
        for sentence in self.raw_data_chunks:
            if any(kw in sentence.lower() for kw in keywords) and sentence != content:
                return sentence[:200]
        return None

    def apply_grammar(self, text):
        """Legacy method - now uses enhanced grammar checker"""
        return self.grammar_checker(text)

    def wikipedia_learning(self, topic):
        return self.deep_research_engine(f"wikipedia {topic}") or f"I tried to learn about {topic} but couldn't find anything."

    def generate_hallucination(self, tokens):
        seed_words = [t for t in tokens if t in self.markov_graph]
        word = random.choice(seed_words) if seed_words else random.choice(list(self.markov_graph.keys()))
        
        result = [word]
        for _ in range(15):
            if word in self.markov_graph:
                word = random.choice(self.markov_graph[word])
                result.append(word)
            else: break
        return " ".join(result)

    def load_all_data(self):
        # 1. Load Logic Modules (JSON)
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        
        # 2. Load GK Modules (Fact Engine)
        if os.path.exists(self.gk_file):
            with open(self.gk_file, 'r', encoding='utf-8') as f:
                self.gk_base = json.load(f)
        
        # 3. Load Knowledge Directory
        if not os.path.exists(self.knowledge_dir): 
            os.makedirs(self.knowledge_dir)
            
        all_tokens = []
        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".txt"):
                path = os.path.join(self.knowledge_dir, filename)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        self.raw_data_chunks.extend([s.strip() for s in sentences if len(s) > 10])
                        all_tokens.extend(self.tokenize(text))
                except: continue
        
        if all_tokens: 
            self.build_markov(all_tokens)
        print(f"--- {self.name} SYSTEM ONLINE (Autonomous Mode) ---")

    def learn_from_user(self, text):
        """Autonomous Learning: Analyzes user input for factual patterns and saves them."""
        words = text.split()
        if len(words) > 8 and any(x in text.lower() for x in [" is ", " was ", " are ", " were "]) and "?" not in text:
            try:
                with open(self.user_mem_file, 'a', encoding='utf-8') as f:
                    f.write(text.strip() + ".\n")
                self.raw_data_chunks.append(text.strip())
                return True
            except: return False
        return False

    def polish_and_save_web_data(self, text):
        """Cleans web data of boilerplate and saves it to the offline brain."""
        clean = re.sub(r'<.*?>', '', text)
        noise = ["click here", "read more", "cookies", "privacy policy", "subscribe", "advertisement"]
        for n in noise: 
            clean = clean.replace(n, "")
        
        clean = clean.strip()
        if len(clean) > 50:
            try:
                with open(self.verified_web_file, 'a', encoding='utf-8') as f:
                    f.write(clean + "\n\n")
                self.raw_data_chunks.append(clean)
                return clean
            except: return clean
        return clean

    def deep_research_engine(self, query):
        """Polished Multi-Engine Search with Autonomous Storage."""
        print(f"({self.name} is performing deep-search and auto-polishing...)")
        engines = [
            {"name": "Google", "url": "https://www.google.com/search?q="},
            {"name": "DuckDuckGo", "url": "https://duckduckgo.com/html/?q="}
        ]
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        
        for engine in engines:
            try:
                url = engine["url"] + urllib.parse.quote(query)
                req = urllib.request.Request(url, headers={'User-Agent': user_agent})
                with urllib.request.urlopen(req, timeout=3) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                
                potential = re.findall(r'>(.*?)<', html)
                cleaned = [s.strip() for s in potential if len(s.strip()) > 60 and "{" not in s]
                
                if cleaned:
                    best = max(cleaned, key=len)
                    polished = self.polish_and_save_web_data(best)
                    return polished
            except: continue
        return None

    def get_response(self, user_input):
        """Main response handler with enhanced processing pipeline"""
        raw_input = user_input.lower().strip()
        emotion_prefix = self.get_emotion_prefix(raw_input)
        tokens = self.tokenize(raw_input)
        
        # Autonomous Conversation Learning
        self.learn_from_user(user_input)

        # Self-Learning Command
        if raw_input.startswith("learn about "):
            topic = raw_input.replace("learn about ", "").strip()
            result = self.wikipedia_learning(topic)
            # Apply rephraser and grammar checker
            result = self.rephraser(result, "clean")
            return self.grammar_checker(result)

        # Check if this is a question requiring detailed answer
        if "?" in user_input or raw_input.startswith(("what", "why", "how", "where", "when", "who", "which")):
            detailed_answer = self.get_detailed_answer(user_input)
            if detailed_answer:
                # Apply rephraser for clarity
                detailed_answer = self.rephraser(detailed_answer, "clean")
                # Final grammar check
                detailed_answer = self.grammar_checker(detailed_answer)
                return self.apply_grammar(emotion_prefix + detailed_answer)

        # Fact Engine (GK Priority)
        for fact in self.gk_base:
            if fact["q"] in raw_input:
                result = self.rephraser(fact["a"], "clean")
                return self.apply_grammar(emotion_prefix + self.grammar_checker(result))

        # Logic Modules (JSON)
        for module in self.knowledge_base:
            for pattern in module["patterns"]:
                if pattern.lower() in raw_input:
                    result = random.choice(module["responses"])
                    result = self.rephraser(result, "clean")
                    return self.apply_grammar(emotion_prefix + self.grammar_checker(result))

        # Pure Emotion Shield
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return self.apply_grammar(emotion_prefix)

        # Precision Search (.txt files)
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
                # Rephrase for better readability
                best_sentence = self.rephraser(best_sentence, "clean")
                return self.apply_grammar(emotion_prefix + best_sentence)

        # Google Search (if no answer found in data)
        if len(tokens) >= 2:
            print(f"Searching Google for: {user_input}")
            search_result = self.google_search(user_input)
            if search_result:
                # Rephrase the search result
                search_result = self.rephraser(search_result, "clean")
                return emotion_prefix + self.grammar_checker(search_result)

        # Fallback
        fallback_msg = "My verification systems could not find a definitive answer. Type 'Learn about [topic]' to help me study!"
        fallback_msg = self.rephraser(fallback_msg, "clean")
        return self.apply_grammar(emotion_prefix + fallback_msg)

# --- WEB SERVER LOGIC ---
class ChatHandler(BaseHTTPRequestHandler):
    bot = None
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            index_path = os.path.join(os.path.dirname(__file__), 'public', 'index.html')
            try:
                with open(index_path, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"Error: public/index.html not found.")
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            user_msg = data.get('message', '')
            
            response_text = self.bot.get_response(user_msg)
            
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
    print("Features enabled: Rephraser, Grammar Checker, Google Search")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
