import os
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
#folder paths
raw = '../../data/raw/'
extracted = "../../data/extracted/"

#EXTRACTING TEXT FROM PDFs, using OCR if necessary
for filename in sorted(os.listdir(raw), reverse=True):#reverse order because old pdfs are usually scans and take long.
    output_path = os.path.join(extracted, filename.replace(".pdf", ".json"))
    
    if os.path.exists(output_path):
        print(f"Skipping {filename} (already extracted)")
        continue
    
    print(f"Processing {filename}...")
    #normal extraction of text first
    try:
        loader = UnstructuredPDFLoader(os.path.join(raw,filename),
        mode='single',#the loader returns the entire PDF as one document object
        strategy='fast')#uses pdfminer under the hood to directly read the text layer embedded in the PDF. Works on all digitally created pdfs
        docs = loader.load()
        text = docs[0].page_content.strip()
    #=====================================================================================    
    #If the text is very short, it probably failed to find characters, so it must be a scanned PDF.
        if len(text) < 100:                                         #arbitrarily chosen
            print('Scanned PDF detected, using OCR.')
            loader = UnstructuredPDFLoader(os.path.join(raw,filename),
            mode = 'single',
            strategy='ocr_only')                  #only difference

            docs = loader.load()
            text = docs[0].page_content.strip()
            

        # saves the extracted text
        extracted_data = {"filename": filename,  "text": text}
        with open(output_path, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, ensure_ascii=False)       
    except Exception as e:
        print(f"  -> FAILED: {e}")