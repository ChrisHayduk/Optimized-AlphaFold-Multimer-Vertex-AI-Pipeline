from kfp.v2 import dsl
from typing import NamedTuple, List
from dataclasses import dataclass

@dataclass
class ChainInfo:
    chain_id: str
    sequence_path: str
    description: str

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['google-cloud-storage']
)
def filter_chains(
    chain_info_list: list,
    per_chain_features_dir: str,
    project: str
) -> NamedTuple('Outputs', [
    ('chains_to_process', List[dict]),
    ('chains_with_precomputed', List[dict])
]):
    """Filters chains based on presence in Google Cloud Storage."""
    from google.cloud import storage
    from collections import namedtuple
    import os
    
    # Initialize GCS client
    client = storage.Client(project=project)
    
    # Parse GCS path
    if per_chain_features_dir.startswith('gs://'):
        bucket_name = per_chain_features_dir.split('/')[2]
        prefix = '/'.join(per_chain_features_dir.split('/')[3:])
        bucket = client.bucket(bucket_name)
    else:
        raise ValueError("per_chain_features_dir must be a GCS path")
    
    chains_to_process = []
    chains_with_precomputed = []
    
    for chain in chain_info_list:
        # Create a structured dictionary with explicit keys
        processed_chain = {
            'chain_id': str(chain['chain_id']),
            'sequence_path': str(chain['sequence_path']),
            'description': str(chain['description'])
        }
        
        chain_path = f"{prefix}/chain_{processed_chain['chain_id']}_features.pkl"
        marker_blob = bucket.blob(chain_path)
        
        if marker_blob.exists():
            chains_with_precomputed.append(processed_chain)
        else:
            chains_to_process.append(processed_chain)
    
    print(f"Found {len(chains_with_precomputed)} chains with precomputed features")
    print(f"Need to process {len(chains_to_process)} chains")
    
    Outputs = namedtuple('Outputs', ['chains_to_process', 'chains_with_precomputed'])
    return Outputs(chains_to_process, chains_with_precomputed)