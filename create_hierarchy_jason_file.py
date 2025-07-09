#https://www.youtube.com/watch?v=tcqEUSNCn8I
import os
import json
import pickle
import pandas as pd

HIERARCHY_PATH = "hierarchy"
DATA_PROCESSING_PATH = os.path.join("data", "Excel")

def build_tree(file_path):
    """
    Parse the Excel file into a parent-child hierarchy.
    Skips the first 4 rows of metadata; expects variable blank columns.
    """
    # Load data, real content starts on Excel row 5
    # Handle both .xls and .xlsx formats
    try:
        df = pd.read_excel(file_path, header=None, skiprows=4, engine="xlrd")
    except Exception as e:
        print(f"Failed to read with xlrd engine, trying openpyxl: {e}")
        df = pd.read_excel(file_path, header=None, skiprows=4, engine="openpyxl")

    last_seen = {}     # maps column_idx -> last seen ID at that level
    nodes = {}         # maps ID -> {'desc':…, 'parent':…}
    children = {}      # maps ID -> [child_id, …]

    for idx, row in df.iterrows():
        # scan left→right for any technical place ID
        for col_idx, cell in row.items():
            if pd.isna(cell):
                continue

            cid = None
            # detect integer-like IDs in strings or numeric cells
            if isinstance(cell, str) and cell.strip().isdigit():
                cid = cell.strip()
            elif isinstance(cell, (int, float)) and float(cell).is_integer():
                cid = str(int(cell))
            else:
                continue

            # found an ID at (excel_row, col_idx)
            # grab description: first non-empty string to the right
            desc = None
            for d in row[col_idx+1:]:
                if isinstance(d, str) and d.strip():
                    desc = d.strip()
                    break

            # find parent: nearest last_seen in any column to the left
            parent = None
            left_cols = [c for c in last_seen.keys() if c < col_idx]
            if left_cols:
                nearest = max(left_cols)
                parent = last_seen[nearest]

            # record node
            nodes[cid] = {'desc': desc, 'parent': parent}
            children.setdefault(cid, [])

            # link to parent
            if parent:
                children.setdefault(parent, []).append(cid)

            # update last_seen for this column, clear deeper levels
            last_seen[col_idx] = cid
            for deeper in [c for c in list(last_seen) if c > col_idx]:
                last_seen.pop(deeper, None)

    return nodes, children

HIERARCHY_PATH = "hierarchy"
DATA_PROCESSING_PATH = os.path.join("data", "data processing")

def main():
    # Build production hierarchy database
    generate_hierarchy_database()

def generate_hierarchy_database():
    print("Building technical location hierarchy database from production data...")
    
    # Find Excel files
    excel_files = find_excel_files()
    if not excel_files:
        print("No Excel files found!")
        return
    
    # Build hierarchy for each file
    for file_path, file_type in excel_files:
        print(f"Processing {file_type} file: {file_path}")
        build_and_save_hierarchy(file_path, file_type)

def find_excel_files():
    """Find Excel files for hierarchy building"""
    files = []
    
    # Main production file
    main_file = os.path.join(DATA_PROCESSING_PATH, "teknisk_lokasjon_struktur_hydro.xlsx")
    if os.path.exists(main_file):
        files.append((main_file, "main"))
        print(f"Found production file: {main_file}")
    else:
        print(f"Error: Production Excel file not found: {main_file}")
        print(f"Looking in directory: {DATA_PROCESSING_PATH}")
        
        # List available files for debugging
        if os.path.exists(DATA_PROCESSING_PATH):
            available_files = [f for f in os.listdir(DATA_PROCESSING_PATH) if f.endswith(('.xls', '.xlsx'))]
            print(f"Available Excel files in directory: {available_files}")
    
    return files

def build_and_save_hierarchy(excel_path, file_type):
    """Build hierarchy from Excel file and save to disk"""
    try:
        print(f"Building hierarchy from {excel_path}...")
        
        # Build the hierarchy
        nodes, children = build_tree(excel_path)
        
        print(f"Built hierarchy with {len(nodes)} nodes and {len(children)} parent-child relationships")
        
        # Create hierarchy directory if it doesn't exist
        os.makedirs(HIERARCHY_PATH, exist_ok=True)
        
        # Save as both JSON (human readable) and pickle (fast loading)
        hierarchy_data = {
            'nodes': nodes,
            'children': children,
            'source_file': excel_path,
            'file_type': file_type,
            'total_nodes': len(nodes),
            'total_relationships': len(children)
        }
        
        # Save as JSON
        json_path = os.path.join(HIERARCHY_PATH, f"hierarchy_{file_type}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON hierarchy to {json_path}")
        
        # Save as pickle (faster loading)
        pickle_path = os.path.join(HIERARCHY_PATH, f"hierarchy_{file_type}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(hierarchy_data, f)
        print(f"Saved pickle hierarchy to {pickle_path}")
        
        # Print some statistics
        print_hierarchy_stats(nodes, children, file_type)
        
    except Exception as e:
        print(f"Error building hierarchy for {file_type}: {e}")
        import traceback
        traceback.print_exc()

def print_hierarchy_stats(nodes, children, file_type):
    """Print statistics about the hierarchy"""
    print(f"\n--- Hierarchy Statistics ({file_type}) ---")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total parent-child relationships: {len(children)}")
    
    # Find root nodes (nodes with no parent)
    root_nodes = [node_id for node_id, node_info in nodes.items() if not node_info['parent']]
    print(f"Root nodes: {len(root_nodes)}")
    
    # Find leaf nodes (nodes with no children)
    leaf_nodes = [node_id for node_id in nodes.keys() if node_id not in children]
    print(f"Leaf nodes: {len(leaf_nodes)}")
    
    # Find nodes with most children
    if children:
        max_children = max(len(child_list) for child_list in children.values())
        nodes_with_max_children = [node_id for node_id, child_list in children.items() if len(child_list) == max_children]
        print(f"Max children per node: {max_children}")
        print(f"Nodes with most children: {nodes_with_max_children[:3]}")  # Show first 3
    
    print("=" * 50)
    print("Hierarchy database loaded successfully!")

if __name__ == "__main__":
    main()
