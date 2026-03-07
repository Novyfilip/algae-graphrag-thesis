import json
import re
from pathlib import Path
import ollama

PROJECT_ROOT = Path("C:/Users/filip/Desktop/Thesis/project")
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Figure caption pattern ---
fig_pattern = re.compile(
    r'((?:Figs?|Figures?)\s*\.?\s*\d+(?:[-–—]\d+)?\..*?)(?=\n\n|\Z)',
    re.DOTALL
)

# --- Reference/acknowledgement detection ---
def find_references(text: str) -> int:
    """Returns the character index to cut at, or len(text) if not found."""
    
    # Tryies explicit headers first (English and common variants)
    header_pattern = re.compile(
        r'\n\s*(References|Bibliography|Acknowledgements?|감사의\s*글|참고\s*문헌)\s*\n',
        re.IGNORECASE
    )
    match = header_pattern.search(text)
    if match:
        return match.start()
    
    # Fallback: finds where citation-dense lines start
    lines = text.split('\n')
    citation_pattern = re.compile(r'\b(19|20)\d{2}[a-z]?\b')
    #This is a sliding window of 5 lines to search in. If any window has over 4 lines containing a year, it is most likely the list of references.
    window_size = 5
    for i in range(len(lines) - window_size):
        window = lines[i:i + window_size]
        hits = sum(1 for line in window if citation_pattern.search(line))
        if hits >= 4:
            return text.index(lines[i])
    
    return len(text)

# --- Main preprocessing function ---
def preprocess_document(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    
    text = doc["text"]
    filename = doc["filename"]
    
    # Extracts metadata via gemma. 1b is fast and works well enough for this job
    response = ollama.chat(
        model="gemma3:1b",
        messages=[{
            "role": "user",
            "content": f"""Extract the title and authors from this paper header.
Return ONLY valid JSON: {{"title": "...", "authors": ["..."]}}

Text: {text[:500]}"""
        }]
    )
    raw = response["message"]["content"]
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        metadata = {"title": None, "authors": None}
    
    # summary of document
    summary_response = ollama.chat(
        model="gemma3:1b",
        messages=[{
            "role": "user",
            "content": f"""You are a marine biology research librarian cataloguing scientific papers.
Summarize what this paper covers in 2-3 sentences. Focus on:
- Which species, organisms, or biological processes are discussed
- Key findings or data reported
- The geographic or ecological context

Be specific with species names and factual claims. This summary will be used
to match the paper to future research queries. Do not include any questions, headers, or bullet points. Return only the summary sentences.

Text: {text[:3000]}"""
        }]
    )
    summary = summary_response["message"]["content"]
    
    # Strip figure captions
    text = fig_pattern.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip references and acknowledgements
    cutpoint = find_references(text)
    text = text[:cutpoint]
    
    processed = {
        "filename": filename,
        "title": metadata.get("title"),
        "authors": metadata.get("authors"),
        "year": filename.split("-")[1],
        "summary": summary,
        "text": text.strip()
    }
    
    # Saves to data/processed
    output_path = PROCESSED_DIR / f"{Path(filename).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    
    return processed
    # 

# --- Batch loop ---
if __name__ == "__main__":
    json_files = sorted(EXTRACTED_DIR.glob("*.json"))
    failed = []

    for i, json_path in enumerate(json_files):
        try:
            preprocess_document(json_path)
        except Exception as e:
            failed.append({"file": json_path.name, "error": str(e)})

        if (i + 1) % 50 == 0 or (i + 1) == len(json_files):
            print(f"[{i + 1}/{len(json_files)}] processed")

    print(f"\nDone. {len(json_files) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        with open(PROCESSED_DIR / "failed.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failed docs saved to {PROCESSED_DIR / 'failed.json'}")