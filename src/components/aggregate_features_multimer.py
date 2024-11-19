from kfp.v2 import dsl
from kfp.v2.dsl import Input, Output, Artifact
import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def aggregate_features_multimer(
    sequence: Input[Artifact],
    msa1: Input[Artifact],
    msa2: Input[Artifact],
    msa3: Input[Artifact],
    template_features: Input[Artifact],
    features: Output[Artifact],
    chain_id: str,
    per_chain_features_dir: str,
    is_homomer: str,
    maxseq: int,
    n_cpu: int = 8
):
    """Conditionally aggregates MSAs and template features based on homomer status."""
    import logging
    import time
    import os
    import pickle
    import tempfile
    from google.cloud import storage
    from alphafold.data import parsers, pipeline
    from alphafold.data import msa_pairing
    from alphafold_utils import aggregate
    from alphafold_utils import run_jackhmmer

    logging.info(f'Starting conditional feature aggregation for chain {chain_id} (is_homomer: {is_homomer})')
    t0 = time.time()
    
    # Regular MSA processing
    msa_paths = []
    msa_paths.append((msa1.path, msa1.metadata['data_format']))
    msa_paths.append((msa2.path, msa2.metadata['data_format']))
    msa_paths.append((msa3.path, msa3.metadata['data_format']))

    # Create a temporary local directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set temporary local path for features
        local_features_path = os.path.join(temp_dir, f'chain_{chain_id}_features.pkl')
        
        model_features = aggregate(
            sequence_path=sequence.path,
            msa_paths=msa_paths,
            template_features_path=template_features.path,
            output_features_path=local_features_path
        )

        # Run UniProt search and process results for heteromers
        if is_homomer.lower() == 'false':
            try:
                logging.info('Running UniProt search for heteromer')
                # Run jackhmmer search
                msa, msa_format = run_jackhmmer(
                    input_path=sequence.path,
                    database_path='uniprot',
                    msa_path=msa.path,
                    n_cpu=n_cpu,
                    maxseq=maxseq
                )

                all_seq_features = pipeline.make_msa_features([msa])
                valid_feats = msa_pairing.MSA_FEATURES + ('msa_species_identifiers',)
                all_seq_msa_features = {
                    f'{k}_all_seq': v for k, v in all_seq_features.items()
                        if k in valid_feats
                }
                model_features.update(all_seq_msa_features)
                    
            except Exception as e:
                logging.warning(f"Failed to process uniprot MSA for chain {chain_id}: {str(e)}")
        
        # Save features locally first
        with open(local_features_path, 'wb') as f:
            pickle.dump(model_features, f, protocol=4)
        
        # Upload to GCS
        storage_client = storage.Client()
        
        # Parse bucket name and blob path from per_chain_features_dir
        bucket_name = per_chain_features_dir.replace('gs://', '').split('/')[0]
        base_path = '/'.join(per_chain_features_dir.replace('gs://', '').split('/')[1:])
        blob_path = f"{base_path}/chain_{chain_id}_features.pkl"
        
        # Get or create bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
            logging.info(f"Using existing bucket: {bucket_name}")
        except Exception:
            try:
                bucket = storage_client.create_bucket(
                    bucket_name,
                    location="us-central1"  # Specify your desired location
                )
                logging.info(f"Created new bucket: {bucket_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to create or access bucket '{bucket_name}'. Error: {str(e)}")
        
        gcs_path = f"gs://{bucket_name}/{blob_path}"
        # Create blob and upload with error handling
        try:
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_features_path)
        except Exception as e:
            raise RuntimeError(f"Failed to upload features to GCS path '{gcs_path}'. Error: {str(e)}")
        
        # Set the features artifact
        features.uri = gcs_path
        features.metadata['category'] = 'features'
        features.metadata['data_format'] = 'pkl'
        features.metadata['chain_id'] = chain_id
        features.metadata['final_dedup_msa_size'] = int(
            model_features['num_alignments'][0]
        )
        features.metadata['total_num_templates'] = int(
            model_features['template_domain_names'].shape[0]
        )

    t1 = time.time()
    logging.info(f'Feature aggregation completed for chain {chain_id}. Elapsed time: {t1-t0}')