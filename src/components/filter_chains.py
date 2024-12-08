from kfp.v2 import dsl
from typing import NamedTuple, List, Dict

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-storage']
)
def filter_chains(
    chain_info_list: list,
    msa_path_info: str,
    project: str
) -> NamedTuple('Outputs', [
    ('chains_to_process', list),
    ('chains_with_precomputed', list)
]):
    """Filters chains based on presence of MSAs in Google Cloud Storage."""
    from google.cloud import storage
    from collections import namedtuple
    import os, json
    
    # Initialize GCS client
    client = storage.Client(project=project)
    
    # Parse the MSA path info
    try:
        msa_paths = json.loads(msa_path_info)
    except json.JSONDecodeError as e:
        print(f"Error parsing msa_path_info: {msa_path_info}")
        raise e
    
    chains_to_process = []
    chains_with_precomputed = []
    
    for chain in chain_info_list:
        processed_chain = {
            'chain_id': str(chain['chain_id']),
            'sequence_path': str(chain['sequence_path']),
            'description': str(chain['description'])
        }
        
        # Add MSA path to the processed chain info
        chain_id = processed_chain['chain_id']
        if chain_id in msa_paths.get('chains', {}):
            msa_path = msa_paths['chains'][chain_id]
            
            # Check if MSA exists
            bucket_name = msa_path.split('/')[2]
            msa_prefix = '/'.join(msa_path.split('/')[3:])
            print('Bucket name: {bucket_name}, msa_prefix: {msa_prefix}')
            bucket = client.bucket(bucket_name)
            marker_blob = bucket.blob(f"{msa_prefix}/features.pkl")
            
            if marker_blob.exists():
                chains_with_precomputed.append(processed_chain)
            else:
                chains_to_process.append(processed_chain)
        else:
            print(f"Warning: No MSA path found for chain {chain_id}")
            chains_to_process.append(processed_chain)
    
    print(f"Found {len(chains_with_precomputed)} chains with precomputed MSAs")
    print(f"Need to process {len(chains_to_process)} chains")
    
    Outputs = namedtuple('Outputs', ['chains_to_process', 'chains_with_precomputed'])
    print("Chains to process: ", chains_to_process)
    print("Found MSAs: ", chains_with_precomputed)
    return Outputs(chains_to_process, chains_with_precomputed)