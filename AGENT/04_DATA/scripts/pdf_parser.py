"""
PDF Parser for NMR Spectrum Data
==================================
Extracts numeric data from the competition PDF file.

NOTE: This script may need adjustment once the actual dataset format is known.
The current implementation attempts multiple extraction strategies.

Usage:
    python pdf_parser.py --input ../../nmr-pattern/Domain_1_processed_NMR_spectrum.pdf --output ../parsed/
"""

import os
import argparse
import warnings

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False

import numpy as np


def extract_with_pdfplumber(pdf_path):
    """
    Attempt extraction using pdfplumber (pure Python, no Java dependency).
    
    Strategies:
    1. Extract tables directly
    2. Extract text and parse numbers
    3. Extract images (if spectra are embedded as images)
    """
    if not HAS_PDFPLUMBER:
        print("  [SKIP] pdfplumber not installed. Run: pip install pdfplumber")
        return None
    
    results = {
        'tables': [],
        'text': [],
        'images': [],
        'pages': 0,
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        results['pages'] = len(pdf.pages)
        print(f"  PDF has {len(pdf.pages)} page(s)")
        
        for i, page in enumerate(pdf.pages):
            # Strategy 1: Extract tables
            tables = page.extract_tables()
            if tables:
                print(f"  Page {i+1}: Found {len(tables)} table(s)")
                for table in tables:
                    results['tables'].append(table)
            
            # Strategy 2: Extract text
            text = page.extract_text()
            if text:
                results['text'].append(text)
                # Count numeric values in text
                nums = [w for w in text.split() if _is_number(w)]
                print(f"  Page {i+1}: Found {len(nums)} numeric values in text")
            
            # Strategy 3: Check for images
            if page.images:
                print(f"  Page {i+1}: Found {len(page.images)} embedded image(s)")
                results['images'].extend(page.images)
    
    return results


def extract_with_tabula(pdf_path):
    """
    Attempt extraction using tabula-py (requires Java).
    Good for structured tables in PDFs.
    """
    if not HAS_TABULA:
        print("  [SKIP] tabula-py not installed. Run: pip install tabula-py")
        return None
    
    try:
        dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        if dfs:
            print(f"  tabula found {len(dfs)} table(s)")
            for i, df in enumerate(dfs):
                print(f"    Table {i+1}: shape {df.shape}")
            return dfs
    except Exception as e:
        print(f"  tabula extraction failed: {e}")
    
    return None


def _is_number(s):
    """Check if a string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_text_to_matrix(text_blocks):
    """
    Parse extracted text into numeric matrix.
    Assumes numbers are space or newline separated.
    """
    all_numbers = []
    
    for text in text_blocks:
        lines = text.strip().split('\n')
        for line in lines:
            row_nums = []
            for token in line.split():
                token = token.strip(',;')
                if _is_number(token):
                    row_nums.append(float(token))
            if row_nums:
                all_numbers.append(row_nums)
    
    if not all_numbers:
        return None
    
    # Try to form a consistent matrix
    row_lengths = [len(row) for row in all_numbers]
    most_common_length = max(set(row_lengths), key=row_lengths.count)
    
    consistent_rows = [row for row in all_numbers if len(row) == most_common_length]
    
    if consistent_rows:
        return np.array(consistent_rows)
    
    return None


def analyze_pdf(pdf_path, output_dir):
    """
    Analyze the PDF and attempt data extraction using all available methods.
    Outputs a report of findings.
    """
    print(f"Analyzing: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path)} bytes")
    print()
    
    report = []
    report.append(f"# PDF Analysis Report\n")
    report.append(f"**File:** {pdf_path}\n")
    report.append(f"**Size:** {os.path.getsize(pdf_path)} bytes\n\n")
    
    # Method 1: pdfplumber
    print("Method 1: pdfplumber extraction")
    plumber_results = extract_with_pdfplumber(pdf_path)
    
    if plumber_results:
        report.append(f"## pdfplumber Results\n")
        report.append(f"- Pages: {plumber_results['pages']}\n")
        report.append(f"- Tables found: {len(plumber_results['tables'])}\n")
        report.append(f"- Text blocks: {len(plumber_results['text'])}\n")
        report.append(f"- Images: {len(plumber_results['images'])}\n\n")
        
        # Try to parse text into matrix
        if plumber_results['text']:
            matrix = parse_text_to_matrix(plumber_results['text'])
            if matrix is not None:
                report.append(f"**Extracted matrix shape:** {matrix.shape}\n")
                np.savetxt(os.path.join(output_dir, 'extracted_from_pdf.csv'),
                           matrix, delimiter=',')
                print(f"  Extracted matrix: {matrix.shape}")
            else:
                report.append("**Could not form a consistent numeric matrix from text**\n")
        
        if plumber_results['images']:
            report.append("\n**NOTE:** PDF contains images. The spectra may be stored as \n")
            report.append("image plots rather than numeric tables. In this case, OCR or \n")
            report.append("image digitization tools (e.g., WebPlotDigitizer) would be needed.\n\n")
    
    # Method 2: tabula-py
    print("\nMethod 2: tabula-py extraction")
    tabula_results = extract_with_tabula(pdf_path)
    
    if tabula_results:
        report.append(f"## tabula-py Results\n")
        for i, df in enumerate(tabula_results):
            report.append(f"- Table {i+1}: {df.shape[0]} rows × {df.shape[1]} columns\n")
            df.to_csv(os.path.join(output_dir, f'tabula_table_{i+1}.csv'), index=False)
    
    # Save report
    report_path = os.path.join(output_dir, 'pdf_analysis_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"\nReport saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='Extract data from NMR spectrum PDF')
    parser.add_argument('--input', type=str,
                        default='../../nmr-pattern/Domain_1_processed_NMR_spectrum.pdf',
                        help='Path to PDF file')
    parser.add_argument('--output', type=str, default='../parsed/',
                        help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.exists(args.input):
        print(f"ERROR: PDF not found at {args.input}")
        print("Please check the path and try again.")
        return
    
    analyze_pdf(args.input, args.output)


if __name__ == '__main__':
    main()
