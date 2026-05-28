import re
import random
import json
import os
import urllib.request
import urllib.parse
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

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

    def apply_grammar(self, text):
        if not text: return ""
        text = text.strip()
        if not text: return ""
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if text[-1] not in ".!?": text += "."
        return text

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
        
        # 3. Load Knowledge Directory (WITHOUT listing full contents for efficiency)
        if not os.path.exists(self.knowledge_dir): os.makedirs(self.knowledge_dir)
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
        
        if all_tokens: self.build_markov(all_tokens)
        print(f"--- {self.name} SYSTEM ONLINE (Autonomous Mode) ---")

    def learn_from_user(self, text):
        """Autonomous Learning: Analyzes user input for factual patterns and saves them."""
        words = text.split()
        # Heuristic: If it's a long sentence containing 'is/was' and no question mark, it's likely a fact.
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
        # 1. Strip HTML tags
        clean = re.sub(r'<.*?>', '', text)
        # 2. Remove common web noise
        noise = ["click here", "read more", "cookies", "privacy policy", "subscribe", "advertisement"]
        for n in noise: clean = clean.replace(n, "")
        
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
                
                # Extract text blocks
                potential = re.findall(r'>(.*?)<', html)
                cleaned = [s.strip() for s in potential if len(s.strip()) > 60 and "{" not in s]
                
                if cleaned:
                    best = max(cleaned, key=len)
                    polished = self.polish_and_save_web_data(best)
                    return polished
            except: continue
        return None

    def get_response(self, user_input):
        raw_input = user_input.lower().strip()
        emotion_prefix = self.get_emotion_prefix(raw_input)
        tokens = self.tokenize(raw_input)
        
        # 0. Autonomous Conversation Learning
        self.learn_from_user(user_input)

        # 1. Self-Learning Command
        if raw_input.startswith("learn about "):
            topic = raw_input.replace("learn about ", "").strip()
            return self.apply_grammar(self.wikipedia_learning(topic))

        # 1. Pure Emotion Shield
        subject_keywords = [t for t in tokens if t not in self.stopwords and t not in self.emotions and len(t) > 3]
        if emotion_prefix and not subject_keywords and len(tokens) <= 4:
            return self.apply_grammar(emotion_prefix)

        # 2. Fact Engine (GK Priority)
        for fact in self.gk_base:
            if fact["q"] in raw_input:
                return self.apply_grammar(emotion_prefix + fact["a"])

        # 3. Logic Modules (JSON)
        for module in self.knowledge_base:
            for pattern in module["patterns"]:
                if pattern.lower() in raw_input:
                    return self.apply_grammar(emotion_prefix + random.choice(module["responses"]))

        # 4. Precision Search (.txt files)
        keywords = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        if keywords:
            best_sentence = None
            max_overlap = 0
            for sentence in self.raw_data_chunks:
                overlap = sum(1 for kw in keywords if kw in sentence.lower())
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence
            if best_sentence and max_overlap >= 2:
                return self.apply_grammar(emotion_prefix + best_sentence)

        # 5. Multi-Engine Research
        if len(tokens) >= 2:
            research_result = self.deep_research_engine(user_input)
            if research_result:
                return emotion_prefix + research_result

        # 6. Fallback Hallucination
        if len(self.markov_graph) > 20:
            return self.apply_grammar(emotion_prefix + self.generate_hallucination(tokens))

        return self.apply_grammar(emotion_prefix + "My verification systems could not find a definitive answer. Type 'Learn about [topic]' to help me study!")

# --- WEB SERVER LOGIC ---
class ChatHandler(BaseHTTPRequestHandler):
    bot = None
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            # FIX: Ensure we look in the 'public' directory
            index_path = os.path.join(os.path.dirname(__file__), 'public', 'index.html')
            try:
                with open(index_path, 'rb') as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.wfile.write(b"Error: public/index.html not found.")

    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            user_msg = data.get('message', '')
            
            response_text = self.bot.get_response(user_msg)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            # ALLOW CROSS-ORIGIN (CORS)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps({'response': response_text}).encode())

    def do_OPTIONS(self):
        """Handle pre-flight requests from browsers."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server():
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    bot = KeyGenAI()
    ChatHandler.bot = bot
    
    # Explicitly bind to 0.0.0.0 for Render
    server_address = ('0.0.0.0', port)
    server = HTTPServer(server_address, ChatHandler)
    
    print(f"--- KeyGen.ai SYSTEM ONLINE ---")
    print(f"Server listening on 0.0.0.0:{port}")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
