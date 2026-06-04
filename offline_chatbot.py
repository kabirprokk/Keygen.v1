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
import math
import ast
import operator as op
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# ==========================================
# SAFE MATHEMATICAL EVALUATOR
# ==========================================
class SafeMathEvaluator:
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: lambda x: x
    }
    
    functions = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'pow': pow,
        'pi': math.pi,
        'e': math.e
    }

    def eval_node(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](self.eval_node(node.left), self.eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](self.eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.functions:
                args = [self.eval_node(arg) for arg in node.args]
                return self.functions[func_name](*args)
            raise ValueError(f"Function {func_name} is not supported")
        elif isinstance(node, ast.Name):
            if node.id in self.functions:
                return self.functions[node.id]
            raise ValueError(f"Constant {node.id} is not supported")
        else:
            raise TypeError(f"Unsupported AST node: {type(node).__name__}")

    def evaluate(self, expr_str):
        expr_str = expr_str.replace('^', '**')
        try:
            tree = ast.parse(expr_str, mode='eval')
            return self.eval_node(tree.body)
        except:
            return None

# ==========================================
# TF-IDF & COSINE SIMILARITY ENGINE (PURE PYTHON)
# ==========================================
class SimpleTFIDF:
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or set()
        self.vocab = set()
        self.idf = {}
        self.docs = []

    def tokenize(self, text):
        if not text:
            return []
        # Strip punctuation and keep alphanumeric words
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords]

    def fit_docs(self, docs_input):
        self.vocab = set()
        self.docs = []
        self.idf = {}
        
        # Count Document Frequency (DF)
        df_counts = defaultdict(int)
        temp_docs = []
        
        for text, payload in docs_input:
            tokens = self.tokenize(text)
            if not tokens:
                continue
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] += 1
            temp_docs.append({'text': text, 'tokens': tokens, 'payload': payload})
            
        total_docs = len(temp_docs)
        if total_docs == 0:
            return
            
        # Compute Inverse Document Frequency (IDF) with smoothing
        for term, df in df_counts.items():
            self.idf[term] = math.log((1 + total_docs) / (1 + df)) + 1
            self.vocab.add(term)
            
        # Compute normalized TF-IDF vectors
        for doc in temp_docs:
            tokens = doc['tokens']
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
                
            vector = {}
            for t, count in tf.items():
                vector[t] = count * self.idf.get(t, 0.0)
                
            sq_sum = sum(w * w for w in vector.values())
            norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
            normalized_vector = {t: w / norm for t, w in vector.items()}
            
            self.docs.append({
                'text': doc['text'],
                'vector': normalized_vector,
                'payload': doc['payload']
            })

    def query(self, query_text):
        query_tokens = self.tokenize(query_text)
        if not query_tokens:
            return None, 0.0
            
        query_tf = defaultdict(int)
        for t in query_tokens:
            query_tf[t] += 1
            
        query_vector = {}
        for t, count in query_tf.items():
            if t in self.idf:
                query_vector[t] = count * self.idf[t]
                
        sq_sum = sum(w * w for w in query_vector.values())
        if sq_sum == 0:
            return None, 0.0
        query_norm = math.sqrt(sq_sum)
        normalized_query = {t: w / query_norm for t, w in query_vector.items()}
        
        best_doc = None
        best_score = 0.0
        
        for doc in self.docs:
            doc_vector = doc['vector']
            score = sum(normalized_query[t] * doc_vector[t] for t in normalized_query if t in doc_vector)
            if score > best_score:
                best_score = score
                best_doc = doc
                
        return best_doc, best_score

# ==========================================
# COGNITIVE ENGINE (MAIN LOGIC)
# ==========================================
class KeyGenAI:
    def __init__(self, data_file="data.json", gk_file="gk_knowledge.json"):
        self.name = "KeyGen.ai"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(self.script_dir, data_file)
        self.gk_file = os.path.join(self.script_dir, gk_file)
        
        # Persistent context directory
        self.knowledge_dir = os.path.join(self.script_dir, "knowledge")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        self.user_mem_file = os.path.join(self.knowledge_dir, "user_mem.txt")
        self.verified_web_file = os.path.join(self.knowledge_dir, "verified_web.txt")
        self.search_cache_file = os.path.join(self.knowledge_dir, "search_cache.json")
        
        self.stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                         "to", "at", "by", "for", "of", "with", "in", "on", "that", "this",
                         "it", "its", "be", "been", "being", "have", "has", "had", "do", "does",
                         "did", "will", "would", "could", "should", "may", "might", "can", "shall"}
        
        self.greetings = {
            "patterns": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", 
                        "howdy", "greetings", "sup", "what's up", "yo", "hola", "heya"],
            "responses": [
                "Hello! 👋 How can I help you today?",
                "Hi there! 😊 What would you like to know?",
                "Hey! ✨ Ask me anything!",
                "Greetings! 🌟 How can I assist you?"
            ]
        }
        
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Engines and memory initialization
        self.math_evaluator = SafeMathEvaluator()
        self.tfidf_engine = SimpleTFIDF(self.stopwords)
        
        self.search_cache = {}
        self.active_entity = None  # Conversation context memory
        self.chat_history = []
        
        self.load_all_data()
        self.load_search_cache()

    def tokenize(self, text):
        if not text:
            return []
        return re.findall(r'\b\w+\b', str(text).lower())

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

    def load_all_data(self):
        # 1. Load data.json
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
            else:
                self.knowledge_base = []
        except Exception as e:
            print(f"Error loading {self.data_file}: {e}")
            self.knowledge_base = []
            
        # 2. Load gk_knowledge.json
        try:
            if os.path.exists(self.gk_file):
                with open(self.gk_file, 'r', encoding='utf-8') as f:
                    self.gk_base = json.load(f)
            else:
                self.gk_base = []
        except Exception as e:
            print(f"Error loading {self.gk_file}: {e}")
            self.gk_base = []

        # 3. Load rules.json and merge into knowledge_base
        rules_path = os.path.join(self.script_dir, "rules.json")
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                    for r in rules_data:
                        if not any(k.get('id') == r.get('id') for k in self.knowledge_base):
                            self.knowledge_base.append(r)
        except Exception as e:
            print(f"Error loading rules.json: {e}")

        # Fit documents into TF-IDF index
        docs_input = []
        
        # Ingest General Knowledge
        for fact in self.gk_base:
            q = fact.get("q", "")
            if q:
                docs_input.append((q, {
                    "type": "gk",
                    "answer": fact.get("a", ""),
                    "subject": q
                }))
                
        # Ingest Rules/Patterns
        for module in self.knowledge_base:
            patterns = module.get("patterns", [])
            responses = module.get("responses", [])
            module_id = module.get("id", "generic")
            if patterns and responses:
                for pattern in patterns:
                    docs_input.append((pattern, {
                        "type": "rule",
                        "responses": responses,
                        "subject": module_id
                    }))

        # Ingest Long-Term Memories
        if os.path.exists(self.user_mem_file):
            try:
                with open(self.user_mem_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 10:
                            docs_input.append((line, {
                                "type": "memory",
                                "answer": line,
                                "subject": line
                            }))
            except Exception as e:
                print(f"Error loading memory: {e}")
                
        self.tfidf_engine.fit_docs(docs_input)
        print(f"Index successfully prepared with {len(docs_input)} items.")

    # Fuzzy String match correction (Levenshtein Distance)
    def levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    def autocorrect_query(self, query):
        tokens = self.tokenize(query)
        corrected_tokens = []
        for token in tokens:
            if token in self.stopwords or token in self.tfidf_engine.vocab:
                corrected_tokens.append(token)
            else:
                # Attempt to find close matches in the vocabulary
                best_match = None
                best_dist = 999
                for word in self.tfidf_engine.vocab:
                    if abs(len(token) - len(word)) <= 1:
                        dist = self.levenshtein_distance(token, word)
                        if dist < best_dist and dist <= 1:  # 1-char spelling error correction
                            best_dist = dist
                            best_match = word
                corrected_tokens.append(best_match if best_match else token)
        return " ".join(corrected_tokens)

    # Conversational memory resolution
    def resolve_pronouns(self, query):
        if not self.active_entity:
            return query
        
        pronouns = [
            (r'\bhe\b', self.active_entity),
            (r'\bshe\b', self.active_entity),
            (r'\bit\b', self.active_entity),
            (r'\bits\b', f"{self.active_entity}'s"),
            (r'\bthey\b', self.active_entity),
            (r'\bhim\b', self.active_entity),
            (r'\bher\b', self.active_entity),
            (r'\bthem\b', self.active_entity)
        ]
        
        resolved_query = query
        for pattern, replacement in pronouns:
            resolved_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)
        return resolved_query

    def extract_active_entity(self, query):
        # Noun phrase parsing with TextBlob if available
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(query)
                nps = blob.noun_phrases
                if nps:
                    self.active_entity = nps[-1]
                    return
            except:
                pass
        
        # Fallback proper noun / vocabulary matcher
        words = query.split()
        if len(words) > 1:
            for w in words[1:]:
                clean_w = w.strip('?,.!').lower()
                if w[0].isupper() and clean_w not in self.stopwords and clean_w in self.tfidf_engine.vocab:
                    self.active_entity = clean_w
                    return

    # Tools: Math, Unit Converter, Date-Time
    def try_eval_math(self, query):
        clean_query = query.lower().strip().rstrip('?').replace('x', '*')
        math_prefixes = ["calculate", "evaluate", "what is", "solve"]
        for p in math_prefixes:
            if clean_query.startswith(p):
                clean_query = clean_query[len(p):].strip()
                break
                
        if any(c in clean_query for c in ['+', '-', '*', '/', '^', '%']) or any(f in clean_query for f in ['sqrt', 'sin', 'cos', 'log']):
            expr = re.sub(r'[^0-9a-zA-Z\+\-\*\/\(\)\.\^\s\,\%]', '', clean_query)
            val = self.math_evaluator.evaluate(expr)
            if val is not None:
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                return f"Math Calculation:\n```math\n{expr} = {val}\n```"
        return None

    def try_unit_conversion(self, query):
        query = query.lower().strip()
        pattern = r'(?:convert\s+)?([\d\.]+)\s*([a-zA-Z_°]+)\s+(?:to|in)\s+([a-zA-Z_°]+)'
        match = re.search(pattern, query)
        if not match:
            return None
            
        value_str, from_unit, to_unit = match.groups()
        try:
            value = float(value_str)
        except ValueError:
            return None
            
        from_unit = from_unit.strip().lower()
        to_unit = to_unit.strip().lower()
        
        conversions = {
            # Temperature
            ('c', 'f'): lambda v: (v * 9/5) + 32,
            ('f', 'c'): lambda v: (v - 32) * 5/9,
            ('c', 'k'): lambda v: v + 273.15,
            ('k', 'c'): lambda v: v - 273.15,
            
            # Length
            ('km', 'miles'): lambda v: v * 0.621371,
            ('miles', 'km'): lambda v: v / 0.621371,
            ('m', 'feet'): lambda v: v * 3.28084,
            ('feet', 'm'): lambda v: v / 3.28084,
            ('cm', 'inches'): lambda v: v * 0.393701,
            ('inches', 'cm'): lambda v: v / 0.393701,
            
            # Weight
            ('kg', 'lbs'): lambda v: v * 2.20462,
            ('lbs', 'kg'): lambda v: v / 2.20462,
            ('g', 'oz'): lambda v: v * 0.035274,
            ('oz', 'g'): lambda v: v / 0.035274
        }
        
        aliases = {
            'celsius': 'c', 'centigrade': 'c', '°c': 'c', 'c': 'c',
            'fahrenheit': 'f', '°f': 'f', 'f': 'f',
            'kelvin': 'k', 'k': 'k',
            'kilometer': 'km', 'kilometers': 'km', 'km': 'km',
            'mile': 'miles', 'miles': 'miles',
            'meter': 'm', 'meters': 'm', 'm': 'm',
            'foot': 'feet', 'feet': 'feet', 'ft': 'feet',
            'centimeter': 'cm', 'centimeters': 'cm', 'cm': 'cm',
            'inch': 'inches', 'inches': 'inches', 'in': 'inches',
            'kilogram': 'kg', 'kilograms': 'kg', 'kg': 'kg',
            'pound': 'lbs', 'pounds': 'lbs', 'lb': 'lbs', 'lbs': 'lbs',
            'gram': 'g', 'grams': 'g', 'g': 'g',
            'ounce': 'oz', 'ounces': 'oz', 'oz': 'oz'
        }
        
        norm_from = aliases.get(from_unit, from_unit)
        norm_to = aliases.get(to_unit, to_unit)
        
        if (norm_from, norm_to) in conversions:
            res = conversions[(norm_from, norm_to)](value)
            return f"Unit Conversion:\n```text\n{value} {from_unit} = {res:.4f} {to_unit}\n```"
        return None

    def try_date_time(self, query):
        query = query.lower().strip()
        time_keywords = ["what time is it", "current time", "what's the time", "tell me the time", "time now"]
        date_keywords = ["what is the date", "what's the date", "today's date", "current date", "what date is it", "tell me the date"]
        day_keywords = ["what day is it", "what day of the week", "today's day", "day is today"]
        year_keywords = ["what year is it", "current year"]
        
        import datetime
        now = datetime.datetime.now()
        
        if any(kw in query for kw in time_keywords):
            return f"The current time is **{now.strftime('%I:%M %p')}**."
        if any(kw in query for kw in date_keywords):
            return f"Today's date is **{now.strftime('%A, %B %d, %Y')}**."
        if any(kw in query for kw in day_keywords):
            return f"Today is **{now.strftime('%A')}**."
        if any(kw in query for kw in year_keywords):
            return f"The current year is **{now.strftime('%Y')}**."
        return None

    # Internet scrapers
    def make_http_request(self, url, timeout=8):
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
            if time.time() - cache_entry['timestamp'] < 86400:  # Cache for 1 day
                return cache_entry['data']
        
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        try:
            html = self.make_http_request(url, timeout=8)
            if html:
                # Extraction matching standard classes
                matches = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
                results = []
                for m in matches:
                    clean = re.sub(r'<.*?>', '', m)
                    clean = re.sub(r'\s+', ' ', clean).strip()
                    if len(clean) > 40:
                        results.append(clean)
                
                if results:
                    best = results[0]
                    self.search_cache[cache_key] = {'data': best, 'timestamp': time.time()}
                    self.save_search_cache()
                    self.polish_and_save_web_data(best)
                    return best
        except:
            pass
        return self._search_wikipedia(query)

    def _search_wikipedia(self, query):
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'KeyGenAI/2.0'})
            with urllib.request.urlopen(req, timeout=8, context=self.ssl_context) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('query', {}).get('search'):
                page_id = data['query']['search'][0]['pageid']
                extract_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1&explaintext=1&pageids={page_id}&format=json"
                req = urllib.request.Request(extract_url, headers={'User-Agent': 'KeyGenAI/2.0'})
                with urllib.request.urlopen(req, timeout=8, context=self.ssl_context) as response:
                    extract_data = json.loads(response.read().decode('utf-8'))
                
                pages = extract_data.get('query', {}).get('pages', {})
                for pid, page_data in pages.items():
                    extract = page_data.get('extract', '')
                    if extract:
                        return self.polish_and_save_web_data(extract[:800])
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
            except:
                pass
        return clean

    def learn_from_user(self, text):
        if not text or len(text.split()) < 7 or "?" in text:
            return False
        factual_patterns = [" is ", " was ", " are ", " were ", " has ", " have "]
        if any(pattern in text.lower() for pattern in factual_patterns):
            try:
                with open(self.user_mem_file, 'a', encoding='utf-8') as f:
                    f.write(text.strip() + ".\n")
                # Reload to put it in TF-IDF index
                self.load_all_data()
                return True
            except:
                pass
        return False

    def get_response(self, user_input):
        if not user_input or not user_input.strip():
            return "Please ask me something! 😊"
            
        raw_input = user_input.strip()
        
        # 1. Handle Greetings
        clean_input = raw_input.lower().rstrip('!.,? ')
        if len(clean_input.split()) <= 2 and any(g == clean_input for g in self.greetings["patterns"]):
            return random.choice(self.greetings["responses"])
            
        # 2. Conversational Memory Pronoun Resolution
        resolved_input = self.resolve_pronouns(raw_input)
        
        # 3. Check for smart tools (use resolved_input so symbols and operators remain intact)
        math_res = self.try_eval_math(resolved_input)
        if math_res: return math_res
        
        unit_res = self.try_unit_conversion(resolved_input)
        if unit_res: return unit_res
        
        datetime_res = self.try_date_time(resolved_input)
        if datetime_res: return datetime_res

        # 4. Spelling correction for text queries
        corrected_input = self.autocorrect_query(resolved_input)

        # 5. Autonomous learning triggers
        self.learn_from_user(raw_input)

        # 6. Check Semantic Match via TF-IDF (Rules, GK, and Memories)
        best_doc, score = self.tfidf_engine.query(corrected_input)
        if best_doc and score > 0.25:
            payload = best_doc['payload']
            self.active_entity = payload.get("subject", self.active_entity)
            
            if payload["type"] == "gk" or payload["type"] == "memory":
                return payload["answer"]
            elif payload["type"] == "rule":
                return random.choice(payload["responses"])

        # 7. Internet search fallback
        web_res = self.google_search(corrected_input)
        if web_res:
            self.extract_active_entity(corrected_input)
            return web_res

        return "I'm not sure about that. Could you rephrase your question? Or try checking my calculations!"

# ==========================================
# WEB SERVER CONTROLLER
# ==========================================
class ChatHandler(BaseHTTPRequestHandler):
    bot = None
    
    def do_GET(self):
        path = self.path
        # Route requests
        if path == '/' or path == '/index.html':
            filepath = os.path.join(os.path.dirname(__file__), 'public', 'index.html')
            content_type = 'text/html'
        elif path == '/index.css':
            filepath = os.path.join(os.path.dirname(__file__), 'public', 'index.css')
            content_type = 'text/css'
        elif path == '/index.js':
            filepath = os.path.join(os.path.dirname(__file__), 'public', 'index.js')
            content_type = 'application/javascript'
        elif path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
            return
        else:
            self.send_error(404, "File Not Found")
            return
            
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Server Error: {str(e)}")

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
+----------------------------------------------+
|                 KeyGen.ai                    |
|         UPGRADED COGNITIVE SERVER            |
|         Listening on http://0.0.0.0:{port}   |
+----------------------------------------------+
    """)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
