#!/usr/bin/env python3
"""
Convert BFS_DFS_Study_Guide.md to PDF

This script tries multiple methods to convert Markdown to PDF.
"""

import os
import sys
import subprocess

def try_pandoc(md_file, pdf_file):
    """Try using pandoc to convert MD to PDF"""
    try:
        cmd = ['pandoc', md_file, '-o', pdf_file, '--pdf-engine=xelatex', 
               '-V', 'geometry:margin=1in', '-V', 'fontsize=11pt']
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def try_markdown_pdf(md_file, pdf_file):
    """Try using markdown-pdf npm package"""
    try:
        cmd = ['markdown-pdf', md_file, '-o', pdf_file]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def try_weasyprint(md_file, pdf_file):
    """Try using markdown2 + weasyprint"""
    try:
        import markdown
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'codehilite'])
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'DejaVu Sans', Arial, sans-serif; margin: 1in; line-height: 1.6; }}
                code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                h1, h2, h3 {{ color: #333; page-break-after: avoid; }}
                @page {{ margin: 1in; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        from weasyprint import HTML
        HTML(string=full_html).write_pdf(pdf_file)
        return True
    except ImportError:
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, 'BFS_DFS_Study_Guide.md')
    pdf_file = os.path.join(script_dir, 'BFS_DFS_Study_Guide.pdf')
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found!")
        sys.exit(1)
    
    print(f"Converting {md_file} to {pdf_file}...")
    
    # Try different methods
    if try_pandoc(md_file, pdf_file):
        print(f"✅ Successfully created {pdf_file} using pandoc!")
        return
    
    if try_weasyprint(md_file, pdf_file):
        print(f"✅ Successfully created {pdf_file} using weasyprint!")
        return
    
    if try_markdown_pdf(md_file, pdf_file):
        print(f"✅ Successfully created {pdf_file} using markdown-pdf!")
        return
    
    print("\n❌ Could not convert to PDF automatically.")
    print("\n📝 Manual conversion options:")
    print("   1. Install pandoc: sudo apt install pandoc texlive-xetex (Linux) or brew install pandoc (Mac)")
    print("      Then run: pandoc BFS_DFS_Study_Guide.md -o BFS_DFS_Study_Guide.pdf")
    print("\n   2. Use online converter: https://www.markdowntopdf.com/")
    print("\n   3. Open in VS Code with Markdown PDF extension")
    print("\n   4. Use Python with weasyprint:")
    print("      pip install markdown weasyprint")
    print("      python convert_to_pdf.py")

if __name__ == '__main__':
    main()
