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
"""Multimer-optimized Alphafold Inference Pipeline."""
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp.v2 import dsl
from kfp.v2.dsl import Artifact, importer

import config as config
from components.aggregate_features_multimer import aggregate_features_multimer
from components.aggregate_features_across_chains import aggregate_features_across_chains as AggregateFeaturesAcrossChainsOp
from components.configure_run_multimer import configure_run_multimer as ConfigureRunOp
from components import hhblits
from components.hmmsearch import hmmsearch
from components import jackhmmer
from components import predict as PredictOp
from components import relax as RelaxOp
from components.download_sequence import download_sequence as DownloadSequenceOp
from components.filter_chains import filter_chains as FilterChainsOp
from components.bfd_search import bfd_search
from components.create_run_id import create_run_id as CreateRunIdOp
import os

JackhmmerOp = create_custom_training_job_from_component(
    jackhmmer,
    display_name='Jackhmmer',
    machine_type=config.JACKHMMER_MACHINE_TYPE,
    nfs_mounts=[dict(
        server=config.NFS_SERVER,
        path=config.NFS_PATH,
        mountPoint=config.NFS_MOUNT_POINT)],
    network=config.NETWORK
)

AggregateOp = create_custom_training_job_from_component(
    aggregate_features_multimer,
    display_name='AggregateOp',
    machine_type=config.JACKHMMER_MACHINE_TYPE,
    nfs_mounts=[dict(
        server=config.NFS_SERVER,
        path=config.NFS_PATH,
        mountPoint=config.NFS_MOUNT_POINT)],
    network=config.NETWORK
)

BFDSearchOp = create_custom_training_job_from_component(
    bfd_search,
    display_name='BFDSearch',
    machine_type=config.HHBLITS_MACHINE_TYPE,
    nfs_mounts=[dict(
        server=config.NFS_SERVER,
        path=config.NFS_PATH,
        mountPoint=config.NFS_MOUNT_POINT,
        mountOptions=['rw,nfsvers=3,exec'] 
    )],
    network=config.NETWORK
)

HHblitsOp = create_custom_training_job_from_component(
    hhblits,
    display_name='HHblits',
    machine_type=config.HHBLITS_MACHINE_TYPE,
    nfs_mounts=[dict(
        server=config.NFS_SERVER,
        path=config.NFS_PATH,
        mountPoint=config.NFS_MOUNT_POINT,
        mountOptions=['rw,nfsvers=3,exec'] 
    )],
    network=config.NETWORK
)

HHsearchOp = create_custom_training_job_from_component(
    hmmsearch,
    display_name='HmmSearch',
    machine_type=config.HHSEARCH_MACHINE_TYPE,
    nfs_mounts=[dict(
        server=config.NFS_SERVER,
        path=config.NFS_PATH,
        mountPoint=config.NFS_MOUNT_POINT,
        mountOptions=['rw,nfsvers=3,exec']
    )],
    network=config.NETWORK
)

JobPredictOp = create_custom_training_job_from_component(
    PredictOp,
    display_name = 'Predict',
    machine_type = 'g2-standard-12' if 'PREDICT_MACHINE_TYPE' not in os.environ else os.environ['PREDICT_MACHINE_TYPE'],
    accelerator_type = 'NVIDIA_L4' if 'PREDICT_ACCELERATOR_TYPE' not in os.environ else os.environ['PREDICT_ACCELERATOR_TYPE'],
    accelerator_count = '1' if 'PREDICT_ACCELERATOR_COUNT' not in os.environ else os.environ['PREDICT_ACCELERATOR_COUNT']
)

JobRelaxOp = create_custom_training_job_from_component(
    RelaxOp,
    display_name = 'Relax',
    machine_type = 'g2-standard-12' if 'RELAX_MACHINE_TYPE' not in os.environ else os.environ['RELAX_MACHINE_TYPE'],
    accelerator_type = 'NVIDIA_L4' if 'RELAX_ACCELERATOR_TYPE' not in os.environ else os.environ['RELAX_ACCELERATOR_TYPE'],
    accelerator_count = '1' if 'RELAX_ACCELERATOR_COUNT' not in os.environ else os.environ['RELAX_ACCELERATOR_COUNT']
)

@dsl.pipeline(
    name='alphafold-multimer-optimized',
    description='AlphaFold multimer inference using parallized MSA search.'
)
def alphafold_multimer_pipeline(
    sequence_path: str,
    project: str,
    region: str,
    max_template_date: str,
    model_preset: str = 'multimer',
    uniref_max_hits: int = config.UNIREF_MAX_HITS,
    mgnify_max_hits: int = config.MGNIFY_MAX_HITS,
    uniprot_max_hits: int = config.UNIPROT_MAX_HITS,
    is_run_relax: str = 'relax',
    num_multimer_predictions_per_model: int = 5,
    use_small_bfd: str = 'true',
):
    """Multimer-optimized Alphafold Inference Pipeline."""

    from kfp.v2 import dsl
    from kfp.v2.dsl import Artifact, importer

    # Configure the run
    run_config = ConfigureRunOp(
        sequence_path=sequence_path,
        model_preset='multimer',
        num_multimer_predictions_per_model=num_multimer_predictions_per_model
    ).set_display_name('Configure Multimer Pipeline Run')

    model_parameters = dsl.importer(
        artifact_uri=config.MODEL_PARAMS_GCS_LOCATION,
        artifact_class=dsl.Artifact,
        reimport=True
    ).set_display_name('Model parameters')

    reference_databases = dsl.importer(
        artifact_uri=config.NFS_MOUNT_POINT,
        artifact_class=dsl.Dataset,
        reimport=False,
        metadata={
            'uniref90': config.UNIREF90_PATH,
            'mgnify': config.MGNIFY_PATH,
            'bfd': config.BFD_PATH,
            'small_bfd': config.SMALL_BFD_PATH,
            'uniref30': config.UNIREF30_PATH,
            'pdb70': config.PDB70_PATH,
            'pdb_mmcif': config.PDB_MMCIF_PATH,
            'pdb_obsolete': config.PDB_OBSOLETE_PATH,
            'pdb_seqres': config.PDB_SEQRES_PATH,
            'uniprot': config.UNIPROT_PATH,
        }
    ).set_display_name('Reference databases')

    # Create the features directory first without sequence dependency
    per_chain_features_dir = CreateRunIdOp(
        sequence_path=sequence_path,
        use_small_bfd=use_small_bfd,
        max_template_date=max_template_date,
        uniref_max_hits=uniref_max_hits,
        mgnify_max_hits=mgnify_max_hits,
        uniprot_max_hits=uniprot_max_hits,
        project=project
    ).set_display_name('Create unique run ID')

    chain_feature_ops = []

    # Create a component to filter chains based on precomputed features
    filter_chains = FilterChainsOp(
        chain_info_list=run_config.outputs['chain_info_list'],
        per_chain_features_dir=per_chain_features_dir.output,
        project=project
    ).set_display_name('Filter chains')

    # Process chains without precomputed features
    with dsl.ParallelFor(
        filter_chains.outputs['chains_to_process'],
        parallelism=config.PARALLELISM
    ) as chain:
        # Import the sequence artifact from GCS URI
        raw_artifact = importer(
            artifact_uri=chain.sequence_path,
            artifact_class=Artifact,
            metadata={
                'chain_id': str(chain.chain_id),
                'description': str(chain.description),
                'category': 'sequence'
            },
            reimport=True,
        ).set_display_name(f"Import sequence artifact for chain {chain.chain_id}")

        # Download sequence
        sequence_artifact = DownloadSequenceOp(raw_artifact.output)
        
        # MSA searches
        msa_searches = {}
        
        # Uniref search
        msa_searches['uniref'] = JackhmmerOp(
            project=project,
            location=region,
            database='uniref90',
            ref_databases=reference_databases.output,
            sequence=sequence_artifact.output,
            maxseq=uniref_max_hits,
        ).set_display_name('Search Uniref')

        # Mgnify search
        msa_searches['mgnify'] = JackhmmerOp(
            project=project,
            location=region,
            database='mgnify',
            ref_databases=reference_databases.output,
            sequence=sequence_artifact.output,
            maxseq=mgnify_max_hits,
        ).set_display_name('Search Mgnify')

        # BFD search (combined component)
        msa_searches['bfd'] = BFDSearchOp(
            project=project,
            location=region,
            sequence=sequence_artifact.output,
            ref_databases=reference_databases.output,
            use_small_bfd=use_small_bfd,
        ).set_display_name('Search BFD')

        # PDB search
        search_pdb = HHsearchOp(
            project=project,
            location=region,
            template_db='pdb_seqres',
            mmcif_db='pdb_mmcif',
            obsolete_db='pdb_obsolete',
            max_template_date=max_template_date,
            ref_databases=reference_databases.output,
            sequence=sequence_artifact.output,
            msa=msa_searches['uniref'].outputs['msa'],
        ).set_display_name('Search PDB')

        # Aggregate features
        aggregate_features = AggregateOp(
            project=project,
            location=region,
            sequence=sequence_artifact.output,
            msa1=msa_searches['uniref'].outputs['msa'],
            msa2=msa_searches['mgnify'].outputs['msa'],
            msa3=msa_searches['bfd'].outputs['msa'],
            template_features=search_pdb.outputs['template_features'],
            chain_id=chain.chain_id,
            per_chain_features_dir=per_chain_features_dir.output,
            is_homomer=run_config.outputs['is_homomer_or_monomer'],
            maxseq=uniprot_max_hits,
        ).after(per_chain_features_dir).after(search_pdb).set_display_name(f"Aggregate features chain {chain.chain_id}")
        
        chain_feature_ops.append(aggregate_features)

    # Aggregate features across chains
    aggregate_features_across_chains = AggregateFeaturesAcrossChainsOp(
        per_chain_features_dir=per_chain_features_dir.output,
        sequences=run_config.outputs['sequence'],
        is_homomer_or_monomer=run_config.outputs['is_homomer_or_monomer'],
        output_features_path=per_chain_features_dir.output,
    ).after(per_chain_features_dir).after(*chain_feature_ops)

    # Second ParallelFor loop for model predictions
    with dsl.ParallelFor(
        loop_args=run_config.outputs['model_runners'], 
        parallelism=config.PARALLELISM
    ) as model_runner:
        model_predict = JobPredictOp(
            project=project,
            location=region,
            model_features=aggregate_features_across_chains.outputs['features'],
            model_params=model_parameters.output,
            model_name=model_runner.model_name,
            prediction_index=model_runner.prediction_index,
            run_multimer_system=run_config.outputs['run_multimer_system'],
            num_ensemble=run_config.outputs['num_ensemble'],
            random_seed=model_runner.random_seed,
            tf_force_unified_memory=config.TF_FORCE_UNIFIED_MEMORY,
            xla_python_client_mem_fraction=config.XLA_PYTHON_CLIENT_MEM_FRACTION
        )
        model_predict.set_display_name('Predict')

        with dsl.Condition(is_run_relax == 'relax'):
            relax_protein = JobRelaxOp(
                project=project,
                location=region,
                unrelaxed_protein=model_predict.outputs['unrelaxed_protein'],
                use_gpu=True,
                tf_force_unified_memory=config.TF_FORCE_UNIFIED_MEMORY,
                xla_python_client_mem_fraction=config.XLA_PYTHON_CLIENT_MEM_FRACTION
            )
            relax_protein.set_display_name('Relax protein')