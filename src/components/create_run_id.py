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
    skip_msa: str,
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
    base_params_no_skip = {
        'use_small_bfd': use_small_bfd,
        'max_template_date': max_template_date,
        'uniref_max_hits': uniref_max_hits,
        'mgnify_max_hits': mgnify_max_hits,
        'uniprot_max_hits': uniprot_max_hits
        # Notice we intentionally omit skip_msa here
    }

    # Function to compute hash and check blob existence
    def compute_hash_and_check(params: dict, sequence_str: str, prefix: str):
        params_copy = params.copy()
        params_copy['sequence_content'] = sequence_str
        params_str = json.dumps(params_copy, sort_keys=True)
        hash_object = hashlib.sha256(params_str.encode())
        current_hash = hash_object.hexdigest()
        path = f"gs://{project}/{prefix}/{current_hash}"

        # Check if blob exists
        out_bucket = storage_client.bucket(project)
        # Create bucket if it doesn't exist
        if not out_bucket.exists():
            try:
                out_bucket = storage_client.create_bucket(
                    project,
                    location="us-central1"
                )
                print(f"Bucket {project} created")
            except Exception as e:
                print(f"Error creating bucket: {str(e)}")
        
        # The path is a directory; actual file might be `features.pkl` or MSA files.
        # We'll just check if there's any blob starting with this prefix.
        # For a minimal existence check, let's just see if the prefix directory has any blob.
        # If directory is empty, we consider that it doesn't exist.
        prefix_path = f"{prefix}/{current_hash}"
        blobs = list(out_bucket.list_blobs(prefix=prefix_path))
        exists = len(blobs) > 0
        
        return path, exists
    
    # Create paths for individual chains using chain_id_map, trying without skip_msa first
    chain_paths = {}
    for chain_id, fasta_chain in chain_id_map.items():
        path_no_skip, exists_no_skip = compute_hash_and_check(base_params_no_skip, fasta_chain.sequence, "chain_msas")
        if exists_no_skip:
            # Use no skip_msa path
            chain_paths[chain_id] = path_no_skip
        else:
            # Include skip_msa and recompute
            base_params_with_skip = base_params_no_skip.copy()
            base_params_with_skip['skip_msa'] = skip_msa
            path_with_skip, _ = compute_hash_and_check(base_params_with_skip, fasta_chain.sequence, "chain_msas")
            chain_paths[chain_id] = path_with_skip

    # Do the same for the full protein
    path_no_skip, exists_no_skip = compute_hash_and_check(base_params_no_skip, sequence_content, "full_protein_msas")
    if exists_no_skip:
        full_protein_path = path_no_skip
    else:
        base_params_with_skip = base_params_no_skip.copy()
        base_params_with_skip['skip_msa'] = skip_msa
        path_with_skip, _ = compute_hash_and_check(base_params_with_skip, sequence_content, "full_protein_msas")
        full_protein_path = path_with_skip

    result_paths = {
        'full_protein': full_protein_path,
        'chains': chain_paths
    }

    print("Result paths:", result_paths)
    
    return json.dumps(result_paths, sort_keys=True)