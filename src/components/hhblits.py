# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A component encapsulating AlphaFold hhblits tool."""

from typing import List
from kfp.v2 import dsl
from kfp.v2.dsl import Artifact
from kfp.v2.dsl import Input
from kfp.v2.dsl import Output

import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE
)
def hhblits(
    sequence: Input[Artifact],
    ref_databases: Input[Artifact],
    databases: List[str],
    msa: Output[Artifact],
    n_cpu: int = 12,
    maxseq: int = 1_000_000,
):
  """Configures and runs hhblits."""

  import logging
  import os
  import time
  import json

  from alphafold_utils import run_hhblits

  logging.info(f'Starting hhblits search on {databases}')
  t0 = time.time()

  # Debug sequence artifact
  logging.info("=== Sequence Artifact Debug Info ===")
  logging.info(f"Type: {type(sequence)}")
  logging.info(f"Dir contents: {dir(sequence)}")
  logging.info(f"URI: {sequence.uri}")
  logging.info(f"Path: {sequence.path}")
  logging.info(f"Name: {sequence.name if hasattr(sequence, 'name') else 'No name'}")
  logging.info(f"Metadata: {json.dumps(sequence.metadata, indent=2)}")

  # Verify sequence file exists
  if not os.path.exists(sequence.path):
      raise FileNotFoundError(f"Sequence file not found at {sequence.path}")
        
  # Get database paths
  database_paths = []
  for db_name in databases:
    db_path = os.path.join(ref_databases.uri, ref_databases.metadata[db_name])
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_name} not found at {db_path}")
    database_paths.append(db_path)
    
  # Log the configuration
  logging.info(f"Input sequence path: {sequence.path}")
  logging.info(f"Database paths: {database_paths}")

  parsed_msa, msa_format = run_hhblits(
      input_path=sequence.path,
      database_paths=database_paths,
      msa_path=msa.path,
      n_cpu=n_cpu,
      maxseq=maxseq
  )

  msa.metadata['category'] = 'msa'
  msa.metadata['num_sequences'] = len(parsed_msa)
  msa.metadata['data_format'] = msa_format
  msa.metadata['databases'] = databases
  msa.metadata['tool'] = 'hhblits'

  t1 = time.time()
  logging.info(f'Hhblits search completed. Elapsed time: {t1-t0}')
