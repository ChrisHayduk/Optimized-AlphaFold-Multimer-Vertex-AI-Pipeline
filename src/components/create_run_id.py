from kfp.v2 import dsl
from kfp.v2.dsl import Input, Artifact

import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def create_run_id(
    sequence_path: str,
    use_small_bfd: str,
    max_template_date: str,
    uniref_max_hits: int,
    mgnify_max_hits: int,
    uniprot_max_hits: int,
    project: str
) -> str:
    """Creates a unique run ID based on sequence content and parameters."""
    import hashlib
    import json
    import tempfile
    from google.cloud import storage
    from typing import Dict, List
    from alphafold.data import parsers
    from alphafold.data import pipeline_multimer  # Add this import
    
    # Parse the GCS path
    if not sequence_path.startswith('gs://'):
        raise ValueError(f"Expected gs:// path, got {sequence_path}")
    
    bucket_name = sequence_path.split('/')[2]
    blob_path = '/'.join(sequence_path.split('/')[3:])
    
    # Download the sequence file from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Create a temporary file to store the sequence
    with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
        # Download the content to the temporary file
        blob.download_to_filename(temp_file.name)
        
        # Read sequence content
        with open(temp_file.name, 'r') as f:
            sequence_content = f.read()
    
    # Parse the sequences using AlphaFold's parser
    seqs, seq_descs = parsers.parse_fasta(sequence_content)
    chain_id_map = pipeline_multimer._make_chain_id_map(
        sequences=seqs,
        descriptions=seq_descs
    )
    
    # Base parameters that affect the run
    base_params = {
        'use_small_bfd': use_small_bfd,
        'max_template_date': max_template_date,
        'uniref_max_hits': uniref_max_hits,
        'mgnify_max_hits': mgnify_max_hits,
        'uniprot_max_hits': uniprot_max_hits
    }

    storage_client = storage.Client()
    output_bucket = storage_client.bucket(project)
    
    # Create bucket if it doesn't exist
    if not output_bucket.exists():
        try:
            output_bucket = storage_client.create_bucket(
                project,
                location="us-central1"
            )
            print(f"Bucket {project} created")
        except Exception as e:
            print(f"Error creating bucket: {str(e)}")
            pass

    # Create paths for individual chains using the chain_id_map
    chain_paths = {}
    for chain_id, fasta_chain in chain_id_map.items():
        chain_params = base_params.copy()
        chain_params['sequence_content'] = fasta_chain.sequence

        print(f"Chain params for chain {chain_id}: {chain_params}")
        
        params_str = json.dumps(chain_params, sort_keys=True)
        hash_object = hashlib.sha256(params_str.encode())
        chain_hash = hash_object.hexdigest()
        
        chain_paths[chain_id] = f"gs://{project}/chain_msas/{chain_hash}"

    # Create path for full protein
    full_params = base_params.copy()
    full_params['sequence_content'] = sequence_content
    
    params_str = json.dumps(full_params, sort_keys=True)
    hash_object = hashlib.sha256(params_str.encode())
    full_hash = hash_object.hexdigest()
    
    full_protein_path = f"gs://{project}/full_protein_msas/{full_hash}"

    # Return all paths as a JSON string
    result_paths = {
        'full_protein': full_protein_path,
        'chains': chain_paths
    }

    print("Results paths: ", result_paths)
    
    return json.dumps(result_paths, sort_keys=True)