#!/usr/bin/env python3
"""
Script to auto-tag notebook cells for nbconvert export.
- Tags code cells with no output as "hide-input"
- Tags specific markdown cells as "hide"
"""
import json
import sys

def tag_notebook(notebook_path, output_path=None):
    if output_path is None:
        output_path = notebook_path

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    # Markdown content to hide (partial match)
    hide_markdown_patterns = [
        "Deep Dive: Understanding BatchNorm",
        "how it works:\n1. center:",  # The follow-up cell
    ]

    tagged_count = 0
    hidden_count = 0

    for i, cell in enumerate(nb['cells']):
        # Initialize tags if not present
        if 'metadata' not in cell:
            cell['metadata'] = {}
        if 'tags' not in cell['metadata']:
            cell['metadata']['tags'] = []

        tags = cell['metadata']['tags']

        # Handle code cells - hide input if no meaningful output
        if cell['cell_type'] == 'code':
            outputs = cell.get('outputs', [])
            source = ''.join(cell.get('source', []))

            # Check if output is empty or just has "Outputs are too large" message
            has_meaningful_output = False
            for out in outputs:
                if out.get('output_type') == 'stream' and out.get('text'):
                    has_meaningful_output = True
                elif out.get('output_type') == 'execute_result':
                    has_meaningful_output = True
                elif out.get('output_type') == 'display_data':
                    has_meaningful_output = True

            # Skip empty cells entirely
            if not source.strip():
                if 'hide' not in tags:
                    tags.append('hide')
                    hidden_count += 1
            # Tag code-only cells (no output)
            elif not has_meaningful_output and 'hide-input' not in tags:
                tags.append('hide-input')
                tagged_count += 1

        # Handle markdown cells - hide specific ones
        elif cell['cell_type'] == 'markdown':
            source = ''.join(cell.get('source', []))
            for pattern in hide_markdown_patterns:
                if pattern in source and 'hide' not in tags:
                    tags.append('hide')
                    hidden_count += 1
                    print(f"  Hiding markdown cell {i}: {source[:50]}...")
                    break

    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\nTagged {tagged_count} code cells with 'hide-input'")
    print(f"Tagged {hidden_count} cells with 'hide' (empty or specific markdown)")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    notebook = sys.argv[1] if len(sys.argv) > 1 else "dev_ex4.ipynb"
    tag_notebook(notebook)
