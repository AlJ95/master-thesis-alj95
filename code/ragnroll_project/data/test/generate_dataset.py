import os
import json
from pathlib import Path

def generate_dataset(input_dir, output_file):
    dataset = []
    
    # Durchlaufe alle Dateien im Eingabeverzeichnis
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = Path(root) / file
                
                # Extrahiere Label und CVE-ID aus dem Dateinamen
                parts = file.split('_')
                label = parts[3]  # VULN oder PATCHED
                cve_id = f"{parts[1]}-{parts[2]}"
                
                # Lese den Dateiinhalt
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Füge den Eintrag zum Dataset hinzu
                dataset.append({
                    'text': content,
                    'label': label,
                    'cve_id': cve_id
                })
    
    # Schreibe das Dataset in die Ausgabedatei
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

if __name__ == '__main__':
    input_dir = 'code/ragnroll_project/data/se_vulnerabilities_test/test/'
    output_file = 'code/ragnroll_project/data/vulnerability_dataset.json'
    generate_dataset(input_dir, output_file)
