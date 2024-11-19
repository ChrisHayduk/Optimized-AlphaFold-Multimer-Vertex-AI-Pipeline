# src/components/extract_chain_info.py
from kfp.v2 import dsl
from kfp.v2.dsl import Input, Artifact
import json

@dsl.component
def extract_chain_info(chains_artifact: Input[Artifact]) -> list:
    return chains_artifact.metadata['chain_info']