#imports
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import instructor
from openai import OpenAI
from typing import Literal, Optional
from pydantic import BaseModel, Field
import time
from neo4j import GraphDatabase

#GLOBAL VARIABLES
CHUNKS_DIR = Path("../../data/chunks/recursive_1000")
CACHE_DIR = Path("data/kg_extractions")
REPORT_DIR = Path("../../outputs/graph")
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphrag"
SYSTEM_PROMPT = """You are a precise scientific information extractor for phycology (algae research).
For each entity you extract, place it in the correct category.
For each relationship, include a confidence score (0.0-1.0) based on how explicitly it is stated.
- 1.0 = directly stated ("Ulva pertusa was found in the Yellow Sea")
- 0.5 = implied but not explicit
- Below 0.3 = weak inference

Only extract what is explicitly stated. Return empty lists if nothing relevant found."""

#SCHEMA
class AlgalTaxon(BaseModel):
    """A species, genus, or taxonomic group of algae."""
    species_name: str = Field(description="Scientific name, e.g., 'Ulva pertusa'")
    taxonomic_rank: Optional[str] = Field(default=None, description="e.g., species, genus, family")

class ChemicalCompound(BaseModel):
    """A metabolite, pigment, toxin, or polysaccharide."""
    compound_name: str = Field(description="Chemical name, e.g., 'fucoxanthin'")

class Method(BaseModel):
    """An experimental technique or analytical procedure."""
    method_name: str = Field(description="Technique name, e.g., 'HPLC', 'rbcL sequencing'")

class Environment(BaseModel):
    """A habitat, biome, or geographic location."""
    environment_name: str = Field(description="Location or habitat, e.g., 'Yellow Sea', 'intertidal zone'")

class GeneticMarker(BaseModel):
    """A molecular marker used for identification or phylogenetics."""
    marker_name: str = Field(description="Marker name, e.g., 'rbcL', 'ITS', 'SSU'")

class Application(BaseModel):
    """An industrial or practical use of algae."""
    application_name: str = Field(description="Use case, e.g., 'biofuel production'")

RelationType = Literal[
    "FOUND_IN", "PRODUCES", "STUDIED_WITH", 
    "IDENTIFIED_BY", "BELONGS_TO", "AFFECTS", "CONTAINS"
]

class Relationship(BaseModel):
    """A relationship between two entities, with confidence score."""
    subject: str = Field(description="Source entity name (exact as extracted)")
    predicate: RelationType = Field(description="Relationship type")
    object: str = Field(description="Target entity name (exact as extracted)")
    confidence: float = Field(
        description="Confidence score 0.0-1.0 based on how explicitly stated in text",
        ge=0.0, le=1.0
    )

class ExtractionResult(BaseModel):
    """Complete extraction output for one chunk."""
    taxa: list[AlgalTaxon] = Field(default_factory=list)
    compounds: list[ChemicalCompound] = Field(default_factory=list)
    methods: list[Method] = Field(default_factory=list)
    environments: list[Environment] = Field(default_factory=list)
    markers: list[GeneticMarker] = Field(default_factory=list)
    applications: list[Application] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

#FUNCTIONS
def filename_to_doi(filename):
    stem = filename.replace(".pdf", "").replace(".json", "")
    parts = stem.split("-")
    return f"10.4490/{parts[0]}.{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"

def extract_from_chunk(chunk_text: str) -> ExtractionResult:
    """Extract entities and relationships with confidence scores."""
    return client.chat.completions.create(
        model="deepseek-chat",
        response_model=ExtractionResult,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract entities and relationships from:\n\n{chunk_text}"}
        ],
        temperature=0.1,
        max_tokens=4096  # Cap output length, had trouble with big tables
    )

def extract_with_cache(chunk_id, chunk_text, cache_dir=CACHE_DIR):
    cache_path = cache_dir / f"{chunk_id}.json"
    
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    result = extract_from_chunk(chunk_text)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f)
    
    return result.model_dump()

def run_full_extraction(json_files: list, max_chunks: int = None, cache_dir: Path = CACHE_DIR):
    """Extract entities and relationships from all chunks with caching."""
    processed = 0
    skipped = 0
    errors = 0
    chunk_count = 0
    error_details = []
    
    for json_file in json_files:
        if max_chunks and chunk_count >= max_chunks:
            break
        
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)
        
        doc_id = doc["filename"].replace(".pdf", "")
        
        for chunk in doc["chunks"]:
            if max_chunks and chunk_count >= max_chunks:
                break
            
            chunk_id = f"{doc_id}_chunk_{chunk['chunk_id']:03d}"
            cache_path = cache_dir / f"{chunk_id}.json"
            
            already_cached = cache_path.exists()
            
            try:
                extract_with_cache(chunk_id, chunk["text"], cache_dir)
                
                if already_cached:
                    skipped += 1
                else:
                    processed += 1
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Error on {chunk_id}: {e}")
                error_details.append({"chunk_id": chunk_id, "error": str(e)})
                errors += 1
            
            chunk_count += 1
            
            if processed % 50 == 0 and processed > 0:
                print(f"Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    
    print("\n~Extraction Complete~")
    print(f"Processed: {processed}")
    print(f"Skipped (cached): {skipped}")
    print(f"Errors: {errors}")
    
    return {
        "processed": processed, 
        "skipped": skipped, 
        "errors": errors,
        "error_details": error_details
    }

def load_extractions(cache_dir: Path = CACHE_DIR):
    """Load all cached extractions into a dict."""
    extractions = {}
    for cache_file in cache_dir.glob("*.json"):
        chunk_id = cache_file.stem
        with open(cache_file, encoding="utf-8") as f:
            extractions[chunk_id] = json.load(f)
    return extractions

def create_lexical_subgraph(session, json_files, extractions):
    """Create Document and Chunk nodes, plus PART_OF and NEXT_CHUNK relationships."""
    docs_created = 0
    chunks_created = 0
    
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            doc = json.load(f)
        
        doc_id = doc["filename"].replace(".pdf", "")
        doi = filename_to_doi(doc["filename"])
        
        doc_chunk_ids = [f"{doc_id}_chunk_{c['chunk_id']:03d}" for c in doc["chunks"]]
        extracted_chunk_ids = [cid for cid in doc_chunk_ids if cid in extractions]
        
        if not extracted_chunk_ids:
            continue
        
        session.run("""
            MERGE (d:Document {doi: $doi})
            SET d.title = $title,
                d.authors = $authors,
                d.year = $year,
                d.filename = $filename
        """, doi=doi, title=doc["title"], authors=doc["authors"], 
            year=doc["year"], filename=doc["filename"])
        docs_created += 1
        
        prev_chunk_id = None
        for chunk in doc["chunks"]:
            chunk_id = f"{doc_id}_chunk_{chunk['chunk_id']:03d}"
            
            if chunk_id not in extractions:
                continue
            
            session.run("""
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text
                WITH c
                MATCH (d:Document {doi: $doi})
                MERGE (c)-[:PART_OF]->(d)
            """, chunk_id=chunk_id, text=chunk["text"], doi=doi)
            chunks_created += 1
            
            if prev_chunk_id and prev_chunk_id in extractions:
                session.run("""
                    MATCH (c1:Chunk {chunk_id: $prev_id})
                    MATCH (c2:Chunk {chunk_id: $curr_id})
                    MERGE (c1)-[:NEXT_CHUNK]->(c2)
                """, prev_id=prev_chunk_id, curr_id=chunk_id)
            
            prev_chunk_id = chunk_id
        
        if docs_created % 20 == 0 and docs_created > 0:
            print(f"Documents: {docs_created}, Chunks: {chunks_created}")
    
    return docs_created, chunks_created

def create_domain_subgraph(session, extractions):
    """Create domain entity nodes, MENTIONS edges, and domain relationships."""
    entities_created = 0
    mentions_created = 0
    rels_created = 0
    
    category_config = {
        "taxa": ("AlgalTaxon", "species_name"),
        "compounds": ("ChemicalCompound", "compound_name"),
        "methods": ("Method", "method_name"),
        "environments": ("Environment", "environment_name"),
        "markers": ("GeneticMarker", "marker_name"),
        "applications": ("Application", "application_name")
    }
    
    for chunk_id, extraction in extractions.items():
        chunk_entity_names = []
        
        for category, (label, name_field) in category_config.items():
            for entity in extraction.get(category, []):
                name = entity.get(name_field)
                if not name:
                    continue
                
                chunk_entity_names.append((label, name))
                
                session.run(f"""
                    MERGE (e:{label} {{name: $name}})
                """, name=name)
                entities_created += 1
        
        for label, name in chunk_entity_names:
            session.run(f"""
                MATCH (c:Chunk {{chunk_id: $chunk_id}})
                MATCH (e:{label} {{name: $name}})
                MERGE (c)-[:MENTIONS]->(e)
            """, chunk_id=chunk_id, name=name)
            mentions_created += 1
        
        for rel in extraction.get("relationships", []):
            if rel.get("confidence", 0) < 0.5:
                continue
            
            session.run("""
                MATCH (s {name: $subject})
                MATCH (o {name: $object})
                MERGE (s)-[r:""" + rel["predicate"] + """]->(o)
                SET r.confidence = $confidence
            """, subject=rel["subject"], object=rel["object"], 
                confidence=rel["confidence"])
            rels_created += 1
        
        if len(extractions) > 50 and entities_created % 500 == 0 and entities_created > 0:
            print(f"Entities: {entities_created}, Mentions: {mentions_created}, Rels: {rels_created}")
    
    return entities_created, mentions_created, rels_created

def compute_extraction_stats(extractions):
    """Compute detailed statistics from extractions."""
    stats = {
        "total_chunks": len(extractions),
        "taxa": 0,
        "compounds": 0,
        "methods": 0,
        "environments": 0,
        "markers": 0,
        "applications": 0,
        "relationships": 0,
        "high_confidence_rels": 0,
        "medium_confidence_rels": 0,
        "relationship_types": {}
    }
    
    for extraction in extractions.values():
        stats["taxa"] += len(extraction.get("taxa", []))
        stats["compounds"] += len(extraction.get("compounds", []))
        stats["methods"] += len(extraction.get("methods", []))
        stats["environments"] += len(extraction.get("environments", []))
        stats["markers"] += len(extraction.get("markers", []))
        stats["applications"] += len(extraction.get("applications", []))
        
        for rel in extraction.get("relationships", []):
            stats["relationships"] += 1
            conf = rel.get("confidence", 0)
            if conf >= 0.8:
                stats["high_confidence_rels"] += 1
            elif conf >= 0.5:
                stats["medium_confidence_rels"] += 1
            
            pred = rel.get("predicate", "UNKNOWN")
            stats["relationship_types"][pred] = stats["relationship_types"].get(pred, 0) + 1
    
    stats["total_entities"] = (
        stats["taxa"] + stats["compounds"] + stats["methods"] + 
        stats["environments"] + stats["markers"] + stats["applications"]
    )
    
    return stats

def generate_report(extraction_stats, ingestion_stats, run_stats, output_dir: Path = REPORT_DIR):
    """Generate detailed and summary reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed JSON report
    detailed_report = {
        "timestamp": timestamp,
        "extraction": {
            "processed": run_stats["processed"],
            "skipped": run_stats["skipped"],
            "errors": run_stats["errors"],
            "error_details": run_stats.get("error_details", [])
        },
        "corpus": extraction_stats,
        "ingestion": {
            "documents": ingestion_stats["documents"],
            "chunks": ingestion_stats["chunks"],
            "entities": ingestion_stats["entities"],
            "mentions": ingestion_stats["mentions"],
            "relationships": ingestion_stats["relationships"]
        }
    }
    
    detailed_path = output_dir / f"report_detailed_{timestamp}.json"
    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(detailed_report, f, indent=2)
    
    # readable summary for thesis
    summary = f"""
================================================================================
KNOWLEDGE GRAPH CONSTRUCTION REPORT
Generated: {timestamp}
================================================================================

EXTRACTION PHASE
----------------
Chunks processed (new):  {run_stats['processed']}
Chunks skipped (cached): {run_stats['skipped']}
Errors:                  {run_stats['errors']}

CORPUS STATISTICS
-----------------
Total chunks extracted:  {extraction_stats['total_chunks']}
Total entities:          {extraction_stats['total_entities']}
  - Taxa:                {extraction_stats['taxa']}
  - Compounds:           {extraction_stats['compounds']}
  - Methods:             {extraction_stats['methods']}
  - Environments:        {extraction_stats['environments']}
  - Genetic markers:     {extraction_stats['markers']}
  - Applications:        {extraction_stats['applications']}

Total relationships:     {extraction_stats['relationships']}
  - High confidence:     {extraction_stats['high_confidence_rels']}
  - Medium confidence:   {extraction_stats['medium_confidence_rels']}

Relationship types:
{chr(10).join(f'  - {k}: {v}' for k, v in extraction_stats['relationship_types'].items())}

NEO4J INGESTION
---------------
Documents created:       {ingestion_stats['documents']}
Chunks created:          {ingestion_stats['chunks']}
Entity nodes created:    {ingestion_stats['entities']}
MENTIONS edges created:  {ingestion_stats['mentions']}
Domain relationships:    {ingestion_stats['relationships']}

================================================================================
"""
    
    summary_path = output_dir / f"report_summary_{timestamp}.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(summary)
    print(f"Reports saved to: {output_dir}")
    
    return detailed_path, summary_path


#CLIENT
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    ),
    mode=instructor.Mode.JSON
)

#NEO4J
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

#MAIN EXECUTION
if __name__ == "__main__":
    print("Starting knowledge graph construction...\n")
    
    # Load source files
    json_files = sorted(CHUNKS_DIR.glob("*.json"), reverse=True)
    print(f"Found {len(json_files)} source documents")
    
    # Run extraction
    run_stats = run_full_extraction(
        json_files, 
        max_chunks=32000,
        cache_dir=CACHE_DIR
    )
    
    # Load all extractions (including newly created ones)
    extractions = load_extractions(CACHE_DIR)
    print(f"\nLoaded {len(extractions)} total extractions")
    
    # Compute extraction statistics
    extraction_stats = compute_extraction_stats(extractions)
    
    # Ingest into Neo4j
    with driver.session() as session:
        print("\nCreating lexical subgraph...")
        docs, chunks = create_lexical_subgraph(session, json_files, extractions)
        print(f"Created {docs} documents, {chunks} chunks")
        
        print("\nCreating domain subgraph...")
        entities, mentions, rels = create_domain_subgraph(session, extractions)
        print(f"Created {entities} entities, {mentions} mentions, {rels} relationships")
    
    ingestion_stats = {
        "documents": docs,
        "chunks": chunks,
        "entities": entities,
        "mentions": mentions,
        "relationships": rels
    }
    
    # Generate reports
    generate_report(extraction_stats, ingestion_stats, run_stats)
    
    driver.close()
    print("\nDone!")