from huggingface_hub import HfApi
import json
from pathlib import Path

TARGET = 12000


def main():
    # Set up output path relative to script location
    ROOT = Path(__file__).resolve().parent.parent
    output_path = ROOT / "configs" / "huggingface_llm_list.jsonl"

    api = HfApi()

    def is_valid_model(m):
        """Filter unwanted models"""
        # keep only generation models
        if m.pipeline_tag not in ["text-generation", "text2text-generation"]:
            return False

        return True

    print("Fetching models from HuggingFace...")

    models = api.list_models(
        sort="downloads"
    )

    selected = []

    for m in models:
        if not is_valid_model(m):
            continue

        entry = {
            "model_id": m.modelId,
            "provider": "huggingface",
            "task": m.pipeline_tag,
            "downloads": m.downloads,
        }

        selected.append(entry)

        if len(selected) >= TARGET:
            break

    print(f"Collected {len(selected)} models")

    with open(output_path, "w") as f:
        for model in selected:
            f.write(json.dumps(model) + "\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
