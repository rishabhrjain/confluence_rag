from pathlib import Path
import json
import re
import html


def save_json(dir, filename,  data):

    # create dir if doesn't exists
    dir = Path(dir)

    if not dir.exists():
        dir.mkdir(parents=True)

    filepath = dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)


def clean_html_text(input_text):
    """
    Clean HTML/XML text using regex, preserving important content including tables.
    Ensures URLs don't contain unwanted spaces.
    
    Args:
        input_text (str): The input HTML/XML text to clean
        
    Returns:
        str: The cleaned text with unnecessary HTML tags removed
    """
    # Unescape HTML entities
    text = html.unescape(input_text)
    
    # Extract and preserve URLs first to prevent spaces being added
    urls = {}
    url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
    
    def save_url(match):
        url = match.group(0)
        placeholder = f"__URL_{len(urls)}__"
        urls[placeholder] = url
        return placeholder
    
    text = re.sub(url_pattern, save_url, text)
    
    # Extract links and save text + URL
    links = {}
    link_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1[^>]*>(.*?)<\/a>'
    
    def save_link(match):
        url = match.group(2)
        link_text = re.sub(r'<[^>]*>', '', match.group(3))
        
        # Check if URL is already a placeholder
        if not url.startswith('__URL_'):
            # Save the URL to prevent spaces
            url_placeholder = f"__URL_{len(urls)}__"
            urls[url_placeholder] = url
            url = url_placeholder
            
        if link_text and url != link_text:
            placeholder = f"__LINK_{len(links)}__"
            links[placeholder] = f"{link_text} ({url})"
            return placeholder
        return link_text
    
    text = re.sub(link_pattern, save_link, text, flags=re.DOTALL)
    
    # Process HTML tables
    tables = []
    table_pattern = r'<table[^>]*>(.*?)<\/table>'
    
    def process_table(match):
        table_content = match.group(1)
        placeholder = f"__TABLE_{len(tables)}__"
        
        # Process rows
        rows = re.findall(r'<tr[^>]*>(.*?)<\/tr>', table_content, flags=re.DOTALL)
        processed_rows = []
        
        for row in rows:
            # Process header cells
            headers = re.findall(r'<th[^>]*>(.*?)<\/th>', row, flags=re.DOTALL)
            
            # Process data cells if no headers found
            if not headers:
                cells = re.findall(r'<td[^>]*>(.*?)<\/td>', row, flags=re.DOTALL)
                if cells:
                    # Clean inner tags from cells
                    cells = [re.sub(r'<[^>]*>', ' ', cell).strip() for cell in cells]
                    processed_rows.append(cells)
            else:
                # Clean inner tags from headers
                headers = [re.sub(r'<[^>]*>', ' ', header).strip() for header in headers]
                processed_rows.append(headers)
        
        # Build table string
        if processed_rows:
            formatted_table = []
            
            # First row might be headers
            if len(processed_rows) > 1:
                formatted_table.append(" | ".join(processed_rows[0]))
                formatted_table.append("-" * len(" | ".join(processed_rows[0])))
                
                for row in processed_rows[1:]:
                    formatted_table.append(" | ".join(row))
            else:
                # Just one row
                formatted_table.append(" | ".join(processed_rows[0]))
            
            tables.append("\n".join(formatted_table))
        else:
            tables.append("")
            
        return placeholder
    
    text = re.sub(table_pattern, process_table, text, flags=re.DOTALL)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove scripts and styles
    text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    
    # Extract content from heading tags and preserve their significance
    for i in range(1, 7):
        heading_pattern = f'<h{i}[^>]*>(.*?)</h{i}>'
        text = re.sub(heading_pattern, r'\n\n\1\n\n', text, flags=re.DOTALL)
    
    # Handle structured macros and other Confluence-specific elements
    text = re.sub(r'<ac:structured-macro.*?</ac:structured-macro>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ac:inline-comment-marker.*?</ac:inline-comment-marker>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ri:.*?</ri:.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ac:.*?</ac:.*?>', '', text, flags=re.DOTALL)
    
    # Handle basic formatting
    text = re.sub(r'<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'<(?:i|em)[^>]*>(.*?)</(?:i|em)>', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'<code[^>]*>(.*?)</code>', r'\1', text, flags=re.DOTALL)
    
    # Handle paragraphs
    text = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', text, flags=re.DOTALL)
    
    # Handle lists
    text = re.sub(r'<li[^>]*>(.*?)</li>', r'• \1\n', text, flags=re.DOTALL)
    text = re.sub(r'<(?:ul|ol)[^>]*>|</(?:ul|ol)>', '', text, flags=re.DOTALL)
    
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Clean up whitespace and formatting
    text = re.sub(r' +', ' ', text)  # Normalize spaces
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)  # Trim lines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize paragraph breaks
    
    # Restore URLs first (no spaces)
    for placeholder, url in urls.items():
        text = text.replace(placeholder, url)
    
    # Restore links (uses the clean URLs)
    for placeholder, link in links.items():
        # Fix any placeholders in the link text
        for url_placeholder, url in urls.items():
            link = link.replace(url_placeholder, url)
        text = text.replace(placeholder, link)
    
    # Restore tables
    for i, table in enumerate(tables):
        table_marker = f"__TABLE_{i}__"
        if table and table_marker in text:
            # Fix any URL placeholders in the table
            for placeholder, url in urls.items():
                table = table.replace(placeholder, url)
            text = text.replace(table_marker, f"\n\n{table}\n\n")
    
    # Detect potential tabular data that wasn't in HTML tables
    lines = text.split('\n')
    processed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if line looks like a column header (3+ capitalized words)
        header_match = re.findall(r'\b[A-Z][a-zA-Z]*\b', line)
        if len(header_match) >= 3 and i + 1 < len(lines):
            # Possible table header found, check next lines for potential data
            potential_headers = header_match[:3]  # Take first 3 detected headers
            
            # Try to find data rows with similar structure
            potential_rows = []
            j = i + 1
            while j < len(lines) and lines[j].strip() and len(re.findall(r'\S+\s+\S+\s+\S+', lines[j].strip())) > 0:
                potential_rows.append(lines[j].strip())
                j += 1
                
            # If we found potential rows, format as a table
            if len(potential_rows) > 0:
                processed_lines.append(" | ".join(potential_headers))
                for row in potential_rows:
                    # Simple splitting by whitespace - can be improved
                    parts = re.split(r'\s{2,}', row, maxsplit=2)
                    if len(parts) >= 3:
                        processed_lines.append(" | ".join(parts[:3]))
                    else:
                        processed_lines.append(row)
                
                i = j - 1  # Skip processed rows
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
        
        i += 1
    
    text = "\n".join(processed_lines)
    
    # Final cleanup of common issues
    text = re.sub(r'\.(?=\S)', '. ', text)  # Ensure space after periods
    text = re.sub(r'\n([a-z])', r' \1', text)  # Join broken sentences
    
    return text.strip()


def extract_formatted_content_from_atlas_doc(atlas_doc_json):
    doc = json.loads(atlas_doc_json)
    
    content_fragments = []
    
    # Recursive function to extract content from nodes
    def process_node(node, depth=0):
        if isinstance(node, dict):
            node_type = node.get('type')
            
            # Extract URLs from attributes
            attrs = node.get('attrs', {})
            if 'url' in attrs:
                url = attrs['url']
                content_fragments.append(f" [URL: {url}]")
            
            # Also check for href in attributes (common for links)
            if 'href' in attrs:
                href = attrs['href']
                content_fragments.append(f" [URL: {href}]")
            
            # Handle text nodes with possible marks (like links)
            if node_type == 'text':
                text = node.get('text', '')
                marks = node.get('marks', [])
                
                # Check for links in marks
                for mark in marks:
                    if mark.get('type') == 'link':
                        mark_attrs = mark.get('attrs', {})
                        href = mark_attrs.get('href')
                        if href:
                            # Format as text with URL in parentheses
                            text = f"{text} [URL: {href}]"
                
                content_fragments.append(text)
            
            # Handle structural elements
            elif node_type in ['paragraph', 'heading']:
                # Process content first
                if 'content' in node and isinstance(node['content'], list):
                    for child in node['content']:
                        process_node(child, depth + 1)
                
                # Add newline after paragraphs and headings
                content_fragments.append('\n')
                
                # Add an extra newline after headings
                if node_type == 'heading':
                    content_fragments.append('\n')
            
            # Handle list items
            elif node_type == 'listItem':
                content_fragments.append('  ' * depth + '• ')
                if 'content' in node and isinstance(node['content'], list):
                    for child in node['content']:
                        process_node(child, depth + 1)
            
            # Process other container nodes
            elif 'content' in node and isinstance(node['content'], list):
                for child in node['content']:
                    process_node(child, depth)
        
        # If it's a list, process each item
        elif isinstance(node, list):
            for item in node:
                process_node(item, depth)
    
    # Start processing from the root
    process_node(doc)
    
    # Join all fragments and clean up multiple consecutive newlines
    joined_content = ''.join(content_fragments)
    
    # Clean up multiple consecutive newlines
    while '\n\n\n' in joined_content:
        joined_content = joined_content.replace('\n\n\n', '\n\n')
    
    return joined_content