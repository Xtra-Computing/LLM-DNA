import requests
import json
from pathlib import Path


def main():
    # Set up output path relative to script location
    ROOT = Path(__file__).resolve().parent.parent
    output_path = ROOT / "configs" / "openrouter_llm_list.jsonl"

    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    def is_generative_text_model(model):
        """
        Determine if the model is a generative text model (decoder-only or encoder-decoder)
        with text input and text output.
        
        Conditions:
        - Output modalities must include "text".
        - Input modalities must include "text".
        """
        architecture = model.get("architecture", {})
        input_mods = architecture.get("input_modalities", [])
        output_mods = architecture.get("output_modalities", [])
        
        # Must output text
        if "text" not in output_mods:
            return False
        
        # Input must include text
        if "text" not in input_mods:
            return False
        
        return True

    models = []
    for m in data["data"]:
        model_id = m["id"].lower()
        if not is_generative_text_model(m):
            continue

        # Construct modality from input and output modalities
        architecture = m.get("architecture", {})
        input_mods = architecture.get("input_modalities", [])
        output_mods = architecture.get("output_modalities", [])
        modality = f"{'|'.join(input_mods)} -> {'|'.join(output_mods)}"

        # Collect model info
        models.append({
            "model_id": m["id"],
            "provider": "openrouter",
            "architecture": "unknown",
            "context_length": m.get("context_length"),
            "modality": modality
        })

    # Write to JSONL file
    with open(output_path, "w") as f:
        for m in models:
            f.write(json.dumps(m) + "\n")

    print(f"Number of eligible generative text models: {len(models)}")
    print("Sample models:")
    for m in models[:5]:
        print(f"  {m['model_id']} (modality: {m['modality']})")

if __name__ == "__main__":
    main()