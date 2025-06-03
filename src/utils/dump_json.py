import os
import json


def dump_json(data: dict, output_path: str="../data/reviews.json") -> None:
  """
    Store RAG results in a JSON file for reviewing
  """
  if os.path.exists(output_path):
    with open(output_path, 'r+', encoding='utf-8') as file:
      existing_data = json.load(file)
      existing_data.append(data)
      file.seek(0)
      json.dump(existing_data, file, ensure_ascii=False, indent=2)
  else:
    with open(output_path, 'w', encoding='utf-8') as file:
      json.dump([data], file, ensure_ascii=False, indent=2)
      