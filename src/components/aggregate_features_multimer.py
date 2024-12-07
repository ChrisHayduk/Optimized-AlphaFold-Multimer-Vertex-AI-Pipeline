from kfp.v2 import dsl
from kfp.v2.dsl import Input, Output, Artifact
import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def aggregate_features_multimer(
    sequence: Input[Artifact],
    ref_databases: Input[Artifact],
    msa1: Input[Artifact],
    msa2: Input[Artifact],
    msa3: Input[Artifact],
    uniprot_msa: Output[Artifact],
    template_features: Input[Artifact],
    features: Output[Artifact],
    chain_id: str,
    per_chain_features_dir: str,
    is_homomer: str,
    maxseq: int,
    skip_msa: str = 'false',
    n_cpu: int = 8,
):
    """Conditionally aggregates MSAs and template features based on homomer status."""
    import logging
    import time
    import os
    import pickle
    import tempfile
    import json
    import numpy as np
    from google.cloud import storage
    from alphafold.data import parsers, pipeline
    from alphafold.data import msa_pairing
    from alphafold_utils import aggregate
    from alphafold_utils import run_jackhmmer

    from alphafold.data.pipeline import make_sequence_features, make_msa_features
    from alphafold.data.parsers import Msa
    from alphafold.common import residue_constants

    def is_artifact_empty(artifact: Artifact) -> bool:
        # Check if artifact.path exists and is non-empty
        if not artifact or not artifact.path:
            return True
        if not os.path.exists(artifact.path):
            return True
        return os.path.getsize(artifact.path) == 0

    logging.info(f'Starting conditional feature aggregation for chain {chain_id} (is_homomer: {is_homomer})')
    t0 = time.time()

    # Load the query sequence
    with open(sequence.path) as f:
        seq_str = f.read().strip()
    seqs, descs = pipeline.parsers.parse_fasta(seq_str)
    if len(seqs) != 1:
        raise ValueError("Expected exactly one sequence.")
    query_sequence_str = seqs[0]
    query_description = descs[0]

    if len(query_sequence_str) == 0:
        raise ValueError("Query sequence is empty, which is invalid. Please provide a non-empty sequence.")

    # Regular MSA processing
    msa_paths = []
    # Only add MSAs if skip_msa is false and they exist
    if skip_msa == 'false':
        if msa1 and not is_artifact_empty(msa1):
            msa_paths.append((msa1.path, msa1.metadata['data_format']))
        if msa2 and not is_artifact_empty(msa2):
            msa_paths.append((msa2.path, msa2.metadata['data_format']))
        if msa3 and not is_artifact_empty(msa3):
            msa_paths.append((msa3.path, msa3.metadata['data_format']))

    # Create a temporary local directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set temporary local path for features
        local_features_path = os.path.join(temp_dir, f'chain_{chain_id}_features.pkl')

        # If skip_msa is true or no MSAs found, create a minimal MSA:
        # A minimal MSA consists of just the single query sequence
        if skip_msa == 'true' or not msa_paths:
            # Create sequence features
            num_res = len(query_sequence_str)

            # 1. Create sequence features:
            sequence_features = pipeline.make_sequence_features(
                sequence=query_sequence_str,
                description=query_description,
                num_res=num_res
            )

            # 2. Create a minimal MSA with just the query:
            # An MSA object requires sequences, descriptions, and deletion_matrix.
            # For a single-sequence MSA, the deletion matrix can be zeros.
            msa = parsers.Msa(
                sequences=[query_sequence_str],
                descriptions=[query_description],
                deletion_matrix=[ [0]*num_res ]
            )

            # 3. Make MSA features from this single-sequence MSA:
            msa_features = pipeline.make_msa_features([msa])

            # Create default/empty template features:
            template_features_dict = {
                'template_aatype': np.zeros(
                    (1, num_res, len(residue_constants.restypes_with_x_and_gap)), dtype=np.float32),
                'template_all_atom_masks': np.zeros(
                    (1, num_res, residue_constants.atom_type_num), dtype=np.float32),
                'template_all_atom_positions': np.zeros(
                    (1, num_res, residue_constants.atom_type_num, 3), dtype=np.float32),
                'template_domain_names': np.array([''.encode()], dtype=object),
                'template_sequence': np.array([''.encode()], dtype=object),
                'template_sum_probs': np.array([0], dtype=np.float32)
            }

            model_features = {**sequence_features, **msa_features, **template_features_dict}

            # Even for is_homomer == 'true', having a stable num_alignments is critical.
            # If is_homomer == 'false', add the *_all_seq features.
            if is_homomer.lower() == 'false':
                # Encode the query sequence into integer IDs:
                query_encoded = np.array(
                    [[residue_constants.HHBLITS_AA_TO_ID[res] for res in query_sequence_str]],
                    dtype=np.int32
                )

                # Create minimal _all_seq features:
                all_seq_features = {
                    'msa_all_seq': query_encoded,                        # shape (1, num_res)
                    'msa_mask_all_seq': np.ones_like(query_encoded, np.float32), # shape (1, num_res)
                    'deletion_matrix_all_seq': np.zeros_like(query_encoded, np.float32), # (1, num_res)
                    # For consistency, you can also add deletion_matrix_int_all_seq if needed:
                    'deletion_matrix_int_all_seq': np.zeros_like(query_encoded, np.int32),

                    'num_alignments_all_seq': np.array([1], dtype=np.int32),
                    'msa_species_identifiers_all_seq': np.array([b''], dtype=object)
                }

                # Now incorporate these into your chain_features if is_homomer_or_monomer is False and no UniProt MSA was found:
                model_features.update(all_seq_features)
        
        else:
            # If we found MSAs, aggregate them normally
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
                    mount_path = ref_databases.uri
                    database_path = os.path.join(mount_path, ref_databases.metadata['uniprot'])
                    msa, msa_format = run_jackhmmer(
                        input_path=sequence.path,
                        database_path=database_path,
                        msa_path=uniprot_msa.path,
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
        
        # Parse the features path from the per_chain_features_dir (which is the JSON from create_run_id)
        paths_info = json.loads(per_chain_features_dir)
        chain_path = paths_info['chains'][chain_id]

        # Parse bucket name and blob path from features path
        bucket_name = chain_path.replace('gs://', '').split('/')[0]
        blob_path = '/'.join(chain_path.replace('gs://', '').split('/')[1:]) + '/features.pkl'
        
        # Upload to GCS
        storage_client = storage.Client()
        
        # Get or create bucket
        try:
            bucket = storage_client.get_bucket(bucket_name)
            logging.info(f"Using existing bucket: {bucket_name}")
        except Exception:
            try:
                bucket = storage_client.create_bucket(
                    bucket_name,
                    location="us-central1"
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
