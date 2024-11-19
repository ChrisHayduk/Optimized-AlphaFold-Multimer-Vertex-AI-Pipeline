"""A component that handles both small and large BFD searches."""

from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Output

import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE
)
def bfd_search(
    sequence: Input[Artifact],
    ref_databases: Input[Artifact],
    use_small_bfd: str,
    msa: Output[Artifact],
    n_cpu: int = 8,
    maxseq: int = 10000,
):
    """Runs either jackhmmer (small BFD) or hhblits (large BFD) search."""

    import logging
    import os
    import time

    from alphafold_utils import run_jackhmmer, run_hhblits

    logging.info(f'Starting BFD search with use_small_bfd={use_small_bfd}')
    t0 = time.time()

    # Ensure sequence file exists
    if not os.path.exists(sequence.path):
        raise FileNotFoundError(f"Sequence file not found at {sequence.path}")

    mount_path = ref_databases.uri

    if use_small_bfd == 'true':
        # Small BFD search using jackhmmer
        database_path = os.path.join(mount_path, ref_databases.metadata['small_bfd'])
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Small BFD database not found at {database_path}")

        parsed_msa, msa_format = run_jackhmmer(
            input_path=sequence.path,
            database_path=database_path,
            msa_path=msa.path,
            n_cpu=n_cpu,
            maxseq=maxseq
        )
        tool_name = 'jackhmmer'
        databases = ['small_bfd']

    else:
        # Large BFD search using hhblits
        database_paths = []
        for db_name in ['bfd', 'uniref30']:
            db_path = os.path.join(mount_path, ref_databases.metadata[db_name])
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database {db_name} not found at {db_path}")
            database_paths.append(db_path)

        parsed_msa, msa_format = run_hhblits(
            input_path=sequence.path,
            database_paths=database_paths,
            msa_path=msa.path,
            n_cpu=n_cpu,
            maxseq=maxseq
        )
        tool_name = 'hhblits'
        databases = ['bfd', 'uniref30']

    # Set metadata
    msa.metadata['category'] = 'msa'
    msa.metadata['num_sequences'] = len(parsed_msa)
    msa.metadata['data_format'] = msa_format
    msa.metadata['databases'] = databases
    msa.metadata['tool'] = tool_name

    t1 = time.time()
    logging.info(f'BFD search completed using {tool_name}. Elapsed time: {t1-t0}')