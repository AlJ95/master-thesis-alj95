import re
import os
import glob
from pathlib import Path
import concurrent.futures
import threading

# List of patterns to remove from HTML files
patterns = [
    # Bestehende Patterns
    r'<script[\s\S]*?</script>',
    r'<li[^>]*>\s*<a[\s\S]*?</a>\s*</li>', # Nur Listen-Elemente mit Links entfernen
    r'<div class="(topNav|bottomNav)">[\s\S]*?</div>',
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
    r'<form class="td-sidebar__search[\s\S]*?</form>',

    # Neue Patterns für Hadoop Directory Listing
    r'<img[^>]*>', # Entferne nur Bilder/Icons
    r' alt="[^"]*"', # Entferne nur alt Attribute

    # remove svg
    r'<svg[^>]*>',

    # remove path
    r'<path[^>]*>',

    # remove footer
    r'<footer[\s\S]*?</footer>',
    
    # Entferne leere Zeilen und Zeilen die nur Leerzeichen enthalten
    r'^\s*$',
]

def format_size(size_in_bytes):
    """Convert bytes to human readable format (KB or MB)"""
    if size_in_bytes < 1024 * 1024:  # Less than 1MB
        return f"{size_in_bytes/1024:.2f}KB"
    return f"{size_in_bytes/(1024*1024):.2f}MB"

def clean_html_file(file_path, output_dir):
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
    
    # Maintain the same directory structure in clean_html
    rel_path = os.path.relpath(file_path, start=html_directory / "html")
    output_file = output_dir / rel_path
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the cleaned content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    # Calculate reduction in size
    original_size = len(content)
    new_size = len(cleaned_content)
    reduction = (original_size - new_size) / original_size * 100 if original_size > 0 else 0
    
    return original_size, new_size, reduction

def process_files_batch(files, output_dir):
    """
    Process a batch of files and return their statistics
    """
    batch_stats = {
        'total_original_size': 0,
        'total_new_size': 0,
        'processed_files': 0
    }
    
    for file_path in files:
        try:
            original_size, new_size, _ = clean_html_file(file_path, output_dir)
            batch_stats['total_original_size'] += original_size
            batch_stats['total_new_size'] += new_size
            batch_stats['processed_files'] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return batch_stats

def process_html_directory(directory):
    """
    Process HTML files in a specific directory using multiple threads
    """
    html_files = list(Path(directory).rglob("*.html"))
    total_files = len(html_files)
    
    if total_files == 0:
        return None
        
    print(f"\nProcessing directory: {directory}")
    print(f"Found {total_files} HTML files")
    
    output_dir = html_directory / "clean_html"
    output_dir.mkdir(exist_ok=True)
    
    # Determine number of threads based on CPU cores
    num_threads = os.cpu_count()
    
    # Split files into batches for each thread
    batch_size = max(1, total_files // num_threads)
    file_batches = [html_files[i:i + batch_size] for i in range(0, len(html_files), batch_size)]
    
    total_stats = {
        'total_original_size': 0,
        'total_new_size': 0,
        'processed_files': 0
    }
    
    # Process batches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {
            executor.submit(process_files_batch, batch, output_dir): batch 
            for batch in file_batches
        }
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_stats = future.result()
            total_stats['total_original_size'] += batch_stats['total_original_size']
            total_stats['total_new_size'] += batch_stats['total_new_size']
            total_stats['processed_files'] += batch_stats['processed_files']
            
            completed += len(future_to_batch[future])
            print(f"Processed {completed}/{total_files} files...")
    
    overall_reduction = ((total_stats['total_original_size'] - total_stats['total_new_size']) / 
                        total_stats['total_original_size'] * 100 
                        if total_stats['total_original_size'] > 0 else 0)
    
    return {
        "directory": directory.name,
        "total_files": total_files,
        "total_original_size": total_stats['total_original_size'],
        "total_new_size": total_stats['total_new_size'],
        "reduction_percent": overall_reduction
    }

if __name__ == "__main__":
    # Get script location
    file_location = Path(__file__)
    html_directory = file_location.parent
    
    # Check if directory exists
    if not os.path.exists(html_directory / "html"):
        print(f"Directory '{html_directory / 'html'}' does not exist.")
        exit(1)
        
    # Process each subdirectory
    all_stats = []
    html_subdirs = [d for d in (html_directory / "html").iterdir() if d.is_dir()]
    
    print("Starting HTML cleanup process...")
    print("=" * 70)
    
    for subdir in html_subdirs:
        stats = process_html_directory(subdir)
        if stats:
            all_stats.append(stats)
            
    print("\nFinal Statistics:")
    print("=" * 70)
    print(f"{'System':<20} {'Files':<10} {'Original Size':<15} {'New Size':<15} {'Reduction':<12}")
    print("-" * 70)
    
    for stat in all_stats:
        print(f"{stat['directory']:<20} {stat['total_files']:<10} "
              f"{format_size(stat['total_original_size']):<15} "
              f"{format_size(stat['total_new_size']):<15} "
              f"{stat['reduction_percent']:.2f}%")
    
    # Calculate overall totals
    total_files = sum(stat['total_files'] for stat in all_stats)
    total_original_size = sum(stat['total_original_size'] for stat in all_stats)
    total_new_size = sum(stat['total_new_size'] for stat in all_stats)
    avg_reduction = sum(stat['reduction_percent'] for stat in all_stats) / len(all_stats)
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_files:<10} "
          f"{format_size(total_original_size):<15} "
          f"{format_size(total_new_size):<15} "
          f"{avg_reduction:.2f}%")