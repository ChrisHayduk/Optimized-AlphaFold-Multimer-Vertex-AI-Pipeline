from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, Output, component, Input

import config as config

@dsl.component(
    base_image=config.ALPHAFOLD_COMPONENTS_IMAGE,
    packages_to_install=['google-cloud-storage']
)
def download_sequence(
    sequence: Input[Artifact],
    downloaded_sequence: Output[Artifact]
):
    """Downloads sequence from GCS and sets up local path."""
    import logging
    from google.cloud import storage
    
    logging.info(f"Downloading sequence from {sequence.uri}")
    
    client = storage.Client()
    with open(downloaded_sequence.path, 'wb') as f:
        client.download_blob_to_file(sequence.uri, f)
    
    # Copy metadata
    downloaded_sequence.metadata.update(sequence.metadata)
    
    logging.info(f"Downloaded sequence to {downloaded_sequence.path}")