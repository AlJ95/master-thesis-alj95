from pathlib import Path
import json

class SchemaException(BaseException):
    pass

ALLOWED_KEYS = {"text", "metadata", "uuid"}

def load_data(file_path: Path):
    """
    Base loading function for JSON files in given format:
    {
        "data": {
            [
                {
                   "uuid": <uuid>,
                   "text": <text that gets embedded>, 
                   "metadata": {
                        "author": <author>,
                        ...
                   }
                },
                ...
            ]
        }
    }
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
            
            if data is None or "data" not in data:
                raise SchemaException("data")
            
            data = data["data"]

            for doc in data:
                
                if "text" not in doc or doc["text"] is None:
                    raise SchemaException(doc)

                if len(set(doc.keys()).difference(ALLOWED_KEYS)) > 0:
                    Warning(f"There are more keys than expected in the JSON file. Please only use following keys: {', '.join(ALLOWED_KEYS)}")

                if "metadata" not in data:
                    doc["metadata"] = {}
            
            return data
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in {file_path}: {e}")
