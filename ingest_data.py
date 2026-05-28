import json
import os

def ingest_structured_data(input_file, output_name):
    """
    Utility to convert large JSON/CSV datasets (like Wiki or Reddit dumps) 
    into clean, sentence-based .txt files for KeyGen.ai.
    """
    knowledge_path = os.path.join("knowledge", f"{output_name}.txt")
    
    # Ensure directory exists
    if not os.path.exists("knowledge"):
        os.makedirs("knowledge")
        
    print(f"Ingesting {input_file}...")
    
    try:
        # This is a template for how you would process a large file
        # For Wikipedia/Reddit, you'd usually parse line-by-line
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(knowledge_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    # Logic to extract 'text' or 'body' fields from JSON lines
                    try:
                        data = json.loads(line)
                        text = data.get('text') or data.get('body') or ""
                        if len(text) > 50:
                            f_out.write(text.replace('\n', ' ') + "\n\n")
                    except:
                        continue
        print(f"Successfully created {knowledge_path}")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    print("KeyGen.ai Data Ingestor")
    # Example usage: ingest_structured_data("reddit_dump.json", "reddit_brain")
