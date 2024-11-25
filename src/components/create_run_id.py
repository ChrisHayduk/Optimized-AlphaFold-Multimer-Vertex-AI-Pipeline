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
    
    # Create a dictionary of all parameters that affect the run
    run_params = {
        'sequence_content': sequence_content,
        'use_small_bfd': use_small_bfd,
        'max_template_date': max_template_date,
        'uniref_max_hits': uniref_max_hits,
        'mgnify_max_hits': mgnify_max_hits,
        'uniprot_max_hits': uniprot_max_hits
    }
    
    # Convert to sorted JSON string to ensure consistent ordering
    params_str = json.dumps(run_params, sort_keys=True)
    
    # Create hash of the combined string
    hash_object = hashlib.sha256(params_str.encode())
    full_hash = hash_object.hexdigest()
    
    unique_run_id = full_hash
    output_bucket_name = project
    gcs_path = f"gs://{output_bucket_name}/msa/{unique_run_id}"
    
    # Create output bucket if it doesn't exist
    output_bucket = storage_client.bucket(output_bucket_name)
    
    if not output_bucket.exists():
        try:
            output_bucket = storage_client.create_bucket(
                output_bucket_name,
                location="us-central1"  # You might want to make this configurable
            )
            print(f"Bucket {output_bucket_name} created")
        except Exception as e:
            print(f"Error creating bucket: {str(e)}")
            # Continue even if bucket creation fails - it might exist but not be accessible
            # to check via exists() due to permissions
            pass
    
    return gcs_path