from xml.etree import ElementTree as ET
import json
import argparse
import os
from copy import deepcopy

def extract_valid_functions_with_library(signature_file):
    # Read and parse the file content
    with open(signature_file, 'r') as file:
        content = file.read()

    # Wrap content in a root tag for XML parsing and parse it
    content = f'<root>{content}</root>'
    root = ET.fromstring(content)

    # List to store valid function details
    valid_functions = []
    library_name = None

    # Find the <exe> element with a <repository>
    exe_with_repo = root.find('.//exe[repository]')
    if exe_with_repo is not None:
        library_name = exe_with_repo.find('name').text
    else:
        library_name = None

    # Iterate over each <fdesc> element
    for fdesc in root.findall('.//fdesc'):
        func_name = fdesc.get('name')
        func_addr = fdesc.get('addr')
        lshcosine_elem = fdesc.find('./lshcosine')

        # Skip if any of the required attributes are missing
        if func_name is None or func_addr is None or lshcosine_elem is None:
            continue

        # Collect LSH cosine hashes
        lshcosine_hashes = [
            hash_elem.text for hash_elem in lshcosine_elem.findall('./hash')
        ]

        # Only include if there are hashes under <lshcosine>
        if not lshcosine_hashes:
            continue

        # Append valid function details with the library name
        valid_functions.append(func_name)

    return valid_functions, library_name


def get_ground_truth_from_signature_file(signature_dir, output_file):
    files = os.listdir(signature_dir)
    ground_truth = {}  # the key is the full library+function name, the value is details info.
    for file in files:
        signature_file_path = os.path.join(signature_dir, file)
        valid_functions, full_library_name = extract_valid_functions_with_library(signature_file_path)

        pre_GroundTruth = {}
        # Process library_name
        if full_library_name is not None:
            temp = full_library_name.split('.so')[0]
            contents = temp.split('-')
            library_name = contents[0]
            if len(contents) > 1:
                library_version = contents[1] 
            else:
                library_version = 'NA'
            compiler_option = 'O2'
            pre_GroundTruth['library_name'] = library_name
            pre_GroundTruth['library_version'] = library_version
            pre_GroundTruth['compiler_option'] = compiler_option
            pre_GroundTruth['compiler_unit'] = 'Unknown'


        for func_name in valid_functions:
            full_key = f'{full_library_name}____{func_name}'
            pre_GroundTruth['function_name'] = func_name
            ground_truth[full_key] = deepcopy(pre_GroundTruth)
    
    with open(output_file, 'w') as f:
        json.dump(ground_truth, f)
            

def main():
    parser = argparse.ArgumentParser(description='Get Ground Truth from signature file for a corpus.')
    parser.add_argument("--signature_dir", help="Directory containing the signature files for the corpus.")
    parser.add_argument("--output_file", help="Output file to save the library scores.")
    args = parser.parse_args()

    get_ground_truth_from_signature_file(args.signature_dir, args.output_file)


if __name__ == '__main__':
    main()
