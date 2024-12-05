from typing import NamedTuple

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Output

import config as config


@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def configure_run_multimer(
    sequence_path: str,
    model_preset: str,
    sequence: Output[Artifact],
    random_seed: int = None,
    num_multimer_predictions_per_model: int = 5,
    model_names: list = None,
) -> NamedTuple(
    'ConfigureRunOutputs',
    [
        ('sequence_path', str),
        ('model_runners', list),
        ('run_multimer_system', bool),
        ('num_ensemble', int),
        ('is_homomer_or_monomer', str),
        ('chain_info_list', list),
    ]
):
    """Configures a pipeline run."""

    import os
    import random
    import sys
    from collections import namedtuple
    from alphafold.data import parsers
    from alphafold.model import config as model_config
    from alphafold.data import pipeline_multimer
    from google.cloud import storage

    # Determine if we are running the multimer system
    run_multimer_system = 'multimer' == model_preset
    num_ensemble = 8 if model_preset == 'monomer_casp14' else 1
    num_predictions_per_model = (
        num_multimer_predictions_per_model if model_preset == 'multimer' else 1
    )

    # Parse the GCS path
    if sequence_path.startswith('gs://'):
        bucket_name = sequence_path.split('/')[2]
        source_blob_path = '/'.join(sequence_path.split('/')[3:])
    else:
        raise ValueError(f"Expected gs:// path, got {sequence_path}")

    # Download the input sequence file from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_path)

    sequence.uri = f'{sequence.uri}.fasta'
    with open(sequence.path, 'wb') as f:
        client.download_blob_to_file(sequence_path, f)

    # Read and parse the input FASTA file
    with open(sequence.path) as f:
        sequence_str = f.read()
    seqs, seq_descs = parsers.parse_fasta(sequence_str)

    if len(seqs) != 1 and model_preset != 'multimer':
        raise ValueError(
            f'More than one sequence found in {sequence_path}.',
            'Unsupported for monomer predictions.'
        )

    # Create a mapping from chain IDs to sequences and descriptions
    chain_id_map = pipeline_multimer._make_chain_id_map(
        sequences=seqs,
        descriptions=seq_descs
    )

    # Get the GCS directory path for chain files
    gcs_dir = os.path.dirname(sequence_path)
    sequence_basename = os.path.splitext(os.path.basename(sequence_path))[0]
    chain_info_list = []
    
    # Create and upload chain files
    for chain_id, fasta_chain in chain_id_map.items():
        # Create local chain file
        chain_fasta_str = f'>{chain_id}\n{fasta_chain.sequence}\n'
        local_chain_path = f'/tmp/chain_{chain_id}.fasta'
        with open(local_chain_path, 'w') as f:
            f.write(chain_fasta_str)
        
        # Upload to GCS with sequence name in path
        gcs_chain_path = f"{gcs_dir}/{sequence_basename}_chain_{chain_id}.fasta"
        blob = bucket.blob(gcs_chain_path.split('gs://' + bucket_name + '/')[1])
        blob.upload_from_filename(local_chain_path)
        
        chain_info_list.append({
            'chain_id': chain_id,
            'sequence_path': gcs_chain_path,  # Use GCS path
            'description': fasta_chain.description
        })

    # Determine if the multimer is a homomer or monomer
    is_homomer_or_monomer = 'true' if len(set(seqs)) == 1 else 'false'
    print(f"Debug - Number of unique sequences: {len(set(seqs))}")
    print(f"Debug - is_homomer_or_monomer: {is_homomer_or_monomer}")
    # Configure model runners
    if model_names is not None:
        models = model_names
    else:
        models = model_config.MODEL_PRESETS[model_preset]

    if random_seed is None:
        # Explicitly cast to int and ensure it's not too large
        max_seed = sys.maxsize // (len(models) * num_multimer_predictions_per_model)
        random_seed = int(random.randrange(max_seed))
    else:
        # Ensure provided random_seed is an integer
        random_seed = int(random_seed)

    model_runners = []
    for model_name in models:
        for i in range(num_predictions_per_model):
            current_seed = int(random_seed + i)  # Explicitly cast each seed to int
            model_runners.append({
                'prediction_index': int(i),  # Also cast index to int for consistency
                'model_name': model_name,
                'random_seed': current_seed
            })

    # Set metadata for the sequence artifact
    sequence.metadata['category'] = 'sequence'
    sequence.metadata['description'] = seq_descs
    sequence.metadata['num_residues'] = [len(seq) for seq in seqs]
    sequence.metadata['chain_info'] = chain_info_list

    output = namedtuple(
        'ConfigureRunOutputs',
        ['sequence_path', 'model_runners', 'run_multimer_system', 'num_ensemble', 'is_homomer_or_monomer', 'chain_info_list']
    )

    print(f"Chains output: {chain_info_list}")

    return output(sequence.path, model_runners, run_multimer_system, num_ensemble, is_homomer_or_monomer, chain_info_list)
