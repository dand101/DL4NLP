import os
import glob
import pandas as pd
import uuid
import time
import json
from tqdm import tqdm
from google import genai
from pydantic import BaseModel, Field

API_KEY = "API_KEY_HERE" 

INPUT_DIR = "/home/teo/semeval9/data/dev_phase/subtask1/train"
OUTPUT_DIR = "/home/teo/semeval9/data/dev_phase/subtask1/train_augmented"

MODEL_NAME = "gemini-2.5-flash-lite" 
MAX_NEW_SAMPLES_PER_LANG = 300  
BATCH_SIZE = 20                 

client = genai.Client(api_key=API_KEY)

class AugmentedEntry(BaseModel):
    text: str = Field(description="The generated social media text")

class AugmentedBatch(BaseModel):
    entries: list[AugmentedEntry]

def generate_samples(language_code, label_int, count, example_texts):

    label_desc = "highly polarized, emotional, and subjective" if label_int == 1 else "neutral, factual, or objective"
    
    examples_json = json.dumps(example_texts, ensure_ascii=False)
    prompt = f"""
    You are a data augmentation system for a SemEval NLP task.
    
    TASK:
    Generate {count} unique {language_code} social media comments that are {label_desc}.
    
    OUTPUT FORMAT:
    Return a strict JSON object with a list of entries.
    
    STYLE EXAMPLES (Mimic tone/length, do not translate):
    {examples_json}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': AugmentedBatch,
            }
        )
        
        if response.parsed:
             return [{"text": entry.text} for entry in response.parsed.entries]
        elif response.text:
             return json.loads(response.text).get("entries", [])
        return []
        
    except Exception as e:
        return []

def augment_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    print(f"Starting Augmentation with {MODEL_NAME}")
    print(f"Found {len(files)} files.")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        if "_augumented" in filename:
            continue
            
        lang_code = filename.split('.')[0] 
        
        df = pd.read_csv(file_path)
        
        counts = df['polarization'].value_counts()
        if 0 not in counts: counts[0] = 0
        if 1 not in counts: counts[1] = 0
        
        major_class = counts.idxmax()
        minor_class = counts.idxmin()
        diff = counts[major_class] - counts[minor_class]
        
        if diff <= 0:
            print(f"{lang_code}: Balanced/Skipped {counts.to_dict()}")
            df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            continue
            
        to_generate = min(diff, MAX_NEW_SAMPLES_PER_LANG)
        print(f"{lang_code}: Generating {to_generate} samples for class {minor_class}")
        
        existing_minority = df[df['polarization'] == minor_class]['text'].tolist()
        few_shot_examples = existing_minority[:5] if len(existing_minority) >= 5 else existing_minority
        
        new_rows = []
        pbar = tqdm(total=to_generate, desc="   Generating")
        
        while len(new_rows) < to_generate:
            current_batch_size = min(BATCH_SIZE, to_generate - len(new_rows))
            
            batch_results = generate_samples(
                lang_code, 
                minor_class, 
                current_batch_size, 
                few_shot_examples
            )
            
            if not batch_results:
                time.sleep(1)
                continue

            for entry in batch_results:
                text = entry['text']
                new_id = f"{lang_code}_{uuid.uuid4().hex}"
                
                new_rows.append({
                    "id": new_id,
                    "text": text,
                    "polarization": minor_class
                })
                
            pbar.update(len(batch_results))
            time.sleep(0.5)
            
        pbar.close()
        
        new_df = pd.DataFrame(new_rows)
        augmented_df = pd.concat([df, new_df], ignore_index=True)
        augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        out_path = os.path.join(OUTPUT_DIR, f"{lang_code}_augumented.csv")
        augmented_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} (Total: {len(augmented_df)})")

if __name__ == "__main__":
    augment_directory()