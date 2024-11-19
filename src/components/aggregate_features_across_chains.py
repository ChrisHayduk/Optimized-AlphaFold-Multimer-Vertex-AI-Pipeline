from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Input, Output
import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def aggregate_features_across_chains(
    per_chain_features_dir: str,  # GCS path
    sequences: Input[Artifact],
    is_homomer_or_monomer: str,
    output_features_path: str,    # GCS path
    features: Output[Artifact],
):
    """Aggregates features across chains for multimer prediction."""
    import pickle
    import tempfile
    from google.cloud import storage
    import logging
    from alphafold.data import feature_processing, pipeline_multimer
    
    storage_client = storage.Client()
    
    # Load all chain features from GCS
    all_chain_features = {}
    chain_info = sequences.metadata['chain_info']
    
    # Parse source bucket info
    source_bucket_name = per_chain_features_dir.replace('gs://', '').split('/')[0]
    source_prefix = '/'.join(per_chain_features_dir.replace('gs://', '').split('/')[1:])
    source_bucket = storage_client.bucket(source_bucket_name)
    
    for chain_data in chain_info:
        chain_id = chain_data['chain_id']
        chain_blob_path = f"{source_prefix}/chain_{chain_id}_features.pkl"
        blob = source_bucket.blob(chain_blob_path)
        
        if not blob.exists():
            raise FileNotFoundError(f"Features file not found in GCS for chain {chain_id}: gs://{source_bucket_name}/{chain_blob_path}")
        
        # Download and process features
        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_filename(temp_file.name)
            with open(temp_file.name, 'rb') as f:
                chain_features = pickle.load(f)

                # Convert monomer features to multimer format
                chain_features = pipeline_multimer.convert_monomer_features(
                    monomer_features=chain_features,
                    chain_id=chain_id
                )
                all_chain_features[chain_id] = chain_features
    
    # Add assembly features and merge
    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    
    if is_homomer_or_monomer == 'true' and len(all_chain_features) == 1:
        # For monomers, just use the single chain features
        chain_id = next(iter(all_chain_features))
        np_example = all_chain_features[chain_id]
    else:
        # For multimers, pair and merge the features
        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features)
    
    # Save merged features to GCS
    dest_bucket_name = output_features_path.replace('gs://', '').split('/')[0]
    dest_blob_path = '/'.join(output_features_path.replace('gs://', '').split('/')[1:])
    dest_bucket = storage_client.bucket(dest_bucket_name)
    dest_blob = dest_bucket.blob(dest_blob_path)
    
    try:
        with tempfile.NamedTemporaryFile() as temp_file:
            with open(temp_file.name, 'wb') as f:
                pickle.dump(np_example, f, protocol=4)
            dest_blob.upload_from_filename(temp_file.name)
    except Exception as e:
        raise RuntimeError(f"Failed to save features to GCS at {output_features_path}: {str(e)}")
    
    features.uri = output_features_path
    features.metadata = {
        'is_homomer_or_monomer': is_homomer_or_monomer,
        'num_chains': len(all_chain_features)
    }

    # Print debug information
    logging.info(f"Successfully processed {len(all_chain_features)} chains")
    logging.info(f"Output features saved to: {output_features_path}")
    logging.info(f"Features metadata: {features.metadata}")