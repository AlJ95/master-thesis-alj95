import re
import os
import glob

# List of patterns to remove from HTML files
patterns = [
    # Bestehende Patterns
    r'<script[\s\S]*?</script>',
    r'<li>[\s\S]*?<a[\s\S]*?</a>[\s\S]*?</li>',
    r'<div class="(topNav|bottomNav)">[\s\S]*?</div>',
    r'<meta[\s\S]*?>',
    r'<!--[\s\S]*?-->',
    r'<style[\s\S]*?</style>',
    r'<link[\s\S]*?>',
    
    # Neue Patterns
    # Entferne Seitenleisten/Asides
    r'<aside[\s\S]*?</aside>',
    
    # Entferne Navigationsleisten
    r'<nav[\s\S]*?</nav>',
    
    # Entferne Header
    r'<header[\s\S]*?</header>',
    
    # Entferne alternative Sprachlinks
    r'<link rel="alternate"[\s\S]*?>',
    
    # Entferne kanonische Links
    r'<link rel="canonical"[\s\S]*?>',
    
    # Entferne Dropdown-Menüs
    r'<div class="dropdown">[\s\S]*?</div>',
    
    # Entferne Suchformulare
    r'<form class="td-sidebar__search[\s\S]*?</form>'

    # Entferne leere Zeilen
    r'^[\s\S]*?$',
]

def clean_html_file(file_path):
    """
    Clean a single HTML file by removing all specified patterns
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Apply each pattern to remove unwanted content
    cleaned_content = content
    for pattern in patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content)
    
    # Write the cleaned content back to the file
    from pathlib import Path
    file_location = Path(__file__)
    html_directory = file_location.parent
    with open(html_directory / "clean_html" / file_path.name, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    # Calculate reduction in size
    original_size = len(content)
    new_size = len(cleaned_content)
    reduction = (original_size - new_size) / original_size * 100 if original_size > 0 else 0
    
    return original_size, new_size, reduction

def process_html_files(directory):
    """
    Process all HTML files in the given directory and its subdirectories
    """
    html_files = glob.glob(f"{directory}/html/**/*.html", recursive=True)
    total_files = len(html_files)
    total_reduction = 0
    total_original_size = 0
    total_new_size = 0
    
    print(f"Found {total_files} HTML files to process")
    
    for i, file_path in enumerate(html_files):
        try:
            original_size, new_size, reduction = clean_html_file(file_path)
            total_original_size += original_size
            total_new_size += new_size
            
            total_reduction += reduction

            if i % 100 == 0:
                print(f"[{i+1}/{total_files}] Average reduction: {total_reduction/i:.2f}%")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    avg_reduction = total_reduction / total_files if total_files > 0 else 0
    overall_reduction = (total_original_size - total_new_size) / total_original_size * 100 if total_original_size > 0 else 0
    
    print(f"\nProcessing completed:")
    print(f"Total files processed: {total_files}")
    print(f"Average reduction per file: {avg_reduction:.2f}%")
    print(f"Overall size reduction: {overall_reduction:.2f}%")
    print(f"Original total size: {total_original_size/1024/1024:.2f} MB")
    print(f"New total size: {total_new_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    # Directory containing HTML files (adjust as needed)
    from pathlib import Path
    file_location = Path(__file__)
    html_directory = file_location.parent
    
    # Check if directory exists
    if not os.path.exists(html_directory):
        print(f"Directory '{html_directory}' does not exist.")
        exit(1)
    
    process_html_files(html_directory)