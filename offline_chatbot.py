import re
import random
import json
import os
import ssl
import urllib.parse
import hashlib
import time
import math
import ast
import operator as op
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler

# --------------------------------------------------------------------
# Robust dependencies (fall back gracefully)
# --------------------------------------------------------------------
try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Modern DuckDuckGo search library (recommended)
try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# --------------------------------------------------------------------
# SAFE MATHEMATICAL EVALUATOR
# --------------------------------------------------------------------
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
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
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

# --------------------------------------------------------------------
# TF-IDF & COSINE SIMILARITY ENGINE
# --------------------------------------------------------------------
class SimpleTFIDF:
    def __init__(self, stopwords=None):
        self.stopwords = stopwords or set()
        self.vocab = set()
        self.idf = {}
        self.docs = []

    def tokenize(self, text):
        if not text: return []
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords]

    def fit_docs(self, docs_input):
        self.vocab = set()
        self.docs = []
        df_counts = defaultdict(int)
        temp_docs = []
        
        for text, payload in docs_input:
            tokens = self.tokenize(text)
            if not tokens: continue
            for t in set(tokens): df_counts[t] += 1
            temp_docs.append({'tokens': tokens, 'payload': payload})
            
        total_docs = len(temp_docs)
        if total_docs == 0: return
            
        for term, df in df_counts.items():
            self.idf[term] = math.log((1 + total_docs) / (1 + df)) + 1
            self.vocab.add(term)
            
        for doc in temp_docs:
            tf = defaultdict(int)
            for t in doc['tokens']: tf[t] += 1
            vector = {t: count * self.idf.get(t, 0.0) for t, count in tf.items()}
            norm = math.sqrt(sum(w * w for w in vector.values())) or 1.0
            doc['vector'] = {t: w / norm for t, w in vector.items()}
            self.docs.append(doc)

    def query(self, query_text):
        query_tokens = self.tokenize(query_text)
        if not query_tokens: return None, 0.0
        query_tf = defaultdict(int)
        for t in query_tokens: query_tf[t] += 1
        query_vector = {t: count * self.idf[t] for t, count in query_tf.items() if t in self.idf}
        norm = math.sqrt(sum(w * w for w in query_vector.values()))
        if norm == 0: return None, 0.0
        normalized_query = {t: w / norm for t, w in query_vector.items()}
        
        best_doc, best_score = None, 0.0
        for doc in self.docs:
            score = sum(normalized_query[t] * doc['vector'][t] for t in normalized_query if t in doc['vector'])
            if score > best_score:
                best_score = score
                best_doc = doc
        return best_doc, best_score

# --------------------------------------------------------------------
# COGNITIVE ENGINE (MAIN LOGIC)
# --------------------------------------------------------------------
class KeyGenAI:
    def __init__(self, data_file="data.json", gk_file="gk_knowledge.json"):
        self.name = "KeyGen.v1"
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(self.script_dir, data_file)
        self.gk_file = os.path.join(self.script_dir, gk_file)
        self.knowledge_dir = os.path.join(self.script_dir, "knowledge")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        self.user_mem_file = os.path.join(self.knowledge_dir, "user_mem.txt")
        self.search_cache_file = os.path.join(self.knowledge_dir, "search_cache.json")
        
        self.stopwords = {"a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
                         "to", "at", "by", "for", "of", "with", "in", "on", "that", "this",
                         "it", "its", "be", "been", "being", "have", "has", "had", "do", "does",
                         "did", "will", "would", "could", "should", "may", "might", "can", "shall"}
        
        self.greetings = {
            "patterns": ["hi", "hello", "hey", "sup", "yo", "greetings"],
            "responses": ["Hello! How can I help you today?", "Hi there! What's on your mind?", "Greetings. I am ready."]
        }
        
        self.math_evaluator = SafeMathEvaluator()
        self.tfidf_engine = SimpleTFIDF(self.stopwords)
        self.search_cache = {}
        self.active_entity = None
        
        self.load_all_data()
        self.load_search_cache()

    # -----------------------------------------------------------------
    # Improved, multi‑layered web search
    # -----------------------------------------------------------------
    def web_search(self, query):
        """
        Perform a web search using the best available method.
        Returns a string with the top result, or None if everything fails.
        """
        if not query.strip():
            return None

        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]

        # ---------------------------------------------------------------
        # 1. Use duckduckgo_search library (most reliable)
        # ---------------------------------------------------------------
        if DDGS_AVAILABLE:
            try:
                with DDGS() as ddgs:
                    # First try an instant answer (e.g., calculator, wiki abstract)
                    instant = list(ddgs.answers(query))
                    if instant:
                        ans = instant[0].get('text') or instant[0].get('abstract')
                        if ans:
                            self.search_cache[cache_key] = ans
                            self.save_search_cache()
                            return ans

                    # Otherwise, take the first organic result
                    results = list(ddgs.text(query, max_results=3))
                    if results:
                        # Prefer a snippet that looks like a full answer
                        for r in results:
                            body = r.get('body')
                            if body and len(body) > 40:
                                ans = body
                                break
                        if not ans:
                            ans = results[0].get('body', '')
                        if ans:
                            self.search_cache[cache_key] = ans
                            self.save_search_cache()
                            return ans
            except Exception as e:
                # Library call failed – fall through to next method
                pass

        # ---------------------------------------------------------------
        # 2. Fallback: DuckDuckGo Instant Answer API + HTML scraping
        #    (only if requests/bs4 are available)
        # ---------------------------------------------------------------
        if REQUESTS_AVAILABLE:
            # Use a session with a realistic user-agent and SSL context
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36'
            }
            # Attempt to handle SSL certificate issues gracefully
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            try:
                # Instant Answer API
                api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
                resp = requests.get(api_url, headers=headers, timeout=8,
                                    verify=False)  # verify=False for problematic envs
                data = resp.json()
                if data.get('AbstractText'):
                    ans = data['AbstractText']
                    self.search_cache[cache_key] = ans
                    self.save_search_cache()
                    return ans

                # HTML scraping fallback
                scrape_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                resp2 = requests.get(scrape_url, headers=headers, timeout=8,
                                     verify=False)
                soup = BeautifulSoup(resp2.text, 'html.parser')
                snippets = soup.find_all('a', class_='result__snippet')
                if snippets:
                    ans = snippets[0].get_text(strip=True)
                    if ans:
                        self.search_cache[cache_key] = ans
                        self.save_search_cache()
                        return ans
            except Exception:
                pass

        # ---------------------------------------------------------------
        # 3. Last resort: bare‑minimum urllib (no external deps)
        # ---------------------------------------------------------------
        try:
            import urllib.request
            api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            with urllib.request.urlopen(api_url, timeout=5,
                                        context=ssl._create_unverified_context()) as u:
                raw = u.read().decode('utf-8')
                data = json.loads(raw)
                if data.get('AbstractText'):
                    ans = data['AbstractText']
                    self.search_cache[cache_key] = ans
                    self.save_search_cache()
                    return ans
        except Exception:
            pass

        return None

    def save_search_cache(self):
        try:
            with open(self.search_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, indent=2)
        except:
            pass

    # -----------------------------------------------------------------
    # Local knowledge loading (unchanged, only pasted for completeness)
    # -----------------------------------------------------------------
    def load_all_data(self):
        docs_input = []
        for file in [self.data_file, self.gk_file, os.path.join(self.script_dir, "rules.json")]:
            if os.path.exists(file):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if "q" in item:
                                    docs_input.append((item["q"], {"type": "gk", "answer": item["a"]}))
                                elif "patterns" in item:
                                    for p in item["patterns"]:
                                        docs_input.append((p, {"type": "rule", "responses": item["responses"]}))
                except:
                    pass

        if os.path.exists(self.user_mem_file):
            try:
                with open(self.user_mem_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(line.strip()) > 5:
                            docs_input.append((line.strip(), {"type": "memory", "answer": line.strip()}))
            except:
                pass

        self.tfidf_engine.fit_docs(docs_input)

    def try_eval_math(self, query):
        expr = re.sub(r'[^0-9\+\-\*\/\(\)\.\^\s]', '', query.replace('x', '*'))
        if any(c in expr for c in '+-*/^'):
            res = self.math_evaluator.evaluate(expr)
            if res is not None:
                return f"Result: **{res}**"
        return None

    def try_unit_conversion(self, query):
        match = re.search(r'([\d\.]+)\s*([a-z]+)\s+(?:to|in)\s+([a-z]+)', query.lower())
        if not match:
            return None
        val, f, t = float(match.group(1)), match.group(2), match.group(3)
        conv = {
            ('kg', 'lbs'): val * 2.20462,
            ('lbs', 'kg'): val / 2.20462,
            ('c', 'f'): (val * 9/5) + 32,
            ('f', 'c'): (val - 32) * 5/9
        }
        res = conv.get((f, t))
        return f"**{val} {f}** = **{res:.2f} {t}**" if res else None

    def get_response(self, user_input):
        if not user_input.strip():
            return "I'm listening."
        inp = user_input.strip().lower()

        # 1. Greetings
        if inp in self.greetings["patterns"]:
            return random.choice(self.greetings["responses"])

        # 2. Math & Units
        math_res = self.try_eval_math(inp)
        if math_res:
            return math_res
        unit_res = self.try_unit_conversion(inp)
        if unit_res:
            return unit_res

        # 3. Local Knowledge
        best_doc, score = self.tfidf_engine.query(inp)
        if best_doc and score > 0.4:
            payload = best_doc['payload']
            if "answer" in payload:
                return payload["answer"]
            if "responses" in payload:
                return random.choice(payload["responses"])

        # 4. Web Search
        web_res = self.web_search(user_input)
        if web_res:
            return web_res

        return "I don't have enough information to answer that. Could you try rephrasing?"

# --------------------------------------------------------------------
# WEB SERVER (unchanged)
# --------------------------------------------------------------------
class ChatHandler(BaseHTTPRequestHandler):
    bot = None

    def do_GET(self):
        path = self.path if self.path != '/' else '/index.html'
        ext = path.split('.')[-1]
        ctype = {'html': 'text/html', 'css': 'text/css', 'js': 'application/javascript'}.get(ext, 'text/plain')
        try:
            with open(os.path.join(os.path.dirname(__file__), 'public', path.lstrip('/')), 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', ctype)
                self.end_headers()
                self.wfile.write(f.read())
        except:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/chat':
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(length))
            resp = self.bot.get_response(data.get('message', ''))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'response': resp}).encode())

def run_server():
    port = int(os.environ.get("PORT", 10000))
    bot = KeyGenAI()
    ChatHandler.bot = bot
    server = HTTPServer(('0.0.0.0', port), ChatHandler)
    print(f"KeyGen.v1 | Deepblack Engine Running on port {port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

if __name__ == "__main__":
    run_server()
