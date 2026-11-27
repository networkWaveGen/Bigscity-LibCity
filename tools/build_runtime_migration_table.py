"""Inventory every original task/model without treating import as runtime migration."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.model.runtime_registry import MODELS


VALIDATION={
    'STTN':'original_first_graph_forecast_and_gradients;stateless_graph_normalization;graph_buffer_and_input_device;multi_target_single_node_batch_one_float64;short_train;independent_evaluate;dropout_exact_resume;CPU_AMP',
    'STNN':'corrected_time_node_reference_forecast_and_gradients;bilinear_grid_reference;hard_lookup_original_12x41;all_relation_modes;trainable_input_mapping;multi_target_batch_one_float64;short_train;independent_evaluate;dropout_exact_resume;CPU_AMP',
    'TGCLSTM':'original_recurrence_prediction_objective_and_gradients;FFR_and_plain_graph;clamped_and_unclamped;geo_ID_order;missing_edge_zero;cache_invalidation;explicit_future_inputs;multi_target_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'ToGCN':'original_autoregressive_and_teacher_forecast_gradients;explicit_teacher_inputs;no_global_anomaly_state;multi_target_batch_one_float64;short_train;independent_evaluate;Python_RNG_and_dropout_exact_resume;CPU_AMP',
    'SimST':'original_forecast_and_gradients;independent_neighbor_and_calendar_reference;causal_mask_buffer;unequal_horizon;large_batch_node_semantics;multi_target_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'TESTAM':'expert_reference_forecast_gate_auxiliary_objective_and_gradients;quantile_and_error_routing;original_LR_curve;multi_target_and_observation;raw_masks;label_free_batch_one_float64;short_train;independent_evaluate;dropout_exact_resume;CPU_AMP',
    'MegaCRN':'reference_prediction_memory_auxiliary_objective_and_gradients;explicit_future_covariates;label_free_training_mode_forward;curriculum_NP_exact_resume;short_train;independent_evaluate;no_covariate_batch_one_float64;CPU_AMP',
    'ASTGCN':'reference_forward_and_gradients;all_CPT_branches;graph_buffers;unequal_filters;multiple_targets;batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP;singleton_graph',
    'MSTGCN':'reference_forward_and_gradients;all_CPT_branches;graph_buffers;unequal_filters;multiple_targets;batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP;singleton_graph',
    'ChebConv':'reference_forward_and_gradients;independent_dual_chebyshev;zero_order;train_node_scaling;missing_attributes;full_graph_short_train;independent_evaluate;exact_resume;embedding_export;float64;CPU_AMP',
    'GAT':'reference_forward_and_gradients;edge_softmax;train_node_scaling;missing_attributes;empty_graph;full_graph_short_train;independent_evaluate;exact_resume;embedding_export;float64;CPU_AMP',
    'FPMC':'factorization_scores_and_cross_entropy_gradients;personalized_user_vocabulary;variable_padding;batch_one_float64;short_train;independent_evaluate;exact_resume;full_vocabulary_ranking',
    'STMatching':'direct_algorithm;single_trajectory_reference;trajectory_state_isolation;timestamps;unmatched_tail;all_unmatched;single_point;actual_CLI;capability_guards',
    'IVMM':'direct_algorithm;single_trajectory_reference;trajectory_state_isolation;timestamps;unmatched_tail;all_unmatched;single_point;actual_CLI;capability_guards',
    'STID':'reference_forward_and_gradients;explicit_calendar;multivariate_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'STAEformer':'canonical_calendar_reference_forward_and_gradients;both_projection_modes;multivariate_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'FreTS':'reference_forward_and_gradients;multivariate_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'STNorm':'reference_forward_and_gradients;multivariate_batch_one_float64;short_train;independent_evaluate;exact_resume;singleton_norm;CPU_AMP',
    'ATDM':'reference_forward_and_gradients;multivariate_batch_one_float64;short_train;independent_evaluate;exact_resume;CPU_AMP',
    'AutoEncoder':'reference_forward_and_gradients;short_train;independent_evaluate;exact_resume',
    'AGCRN':'reference_forward_and_gradients;short_train;independent_evaluate;exact_resume',
    'TGCN':'reference_forward_and_objective_gradients;graph_buffer_dtype;short_train;independent_evaluate;exact_resume',
    'TrafficRNN':'reference_forward_and_gradients;label_free_inference;teacher_step;short_train;independent_evaluate;exact_resume',
    'TrafficLSTM':'reference_forward_and_gradients;label_free_inference;teacher_step;short_train;independent_evaluate;exact_resume',
    'Seq2Seq':'zero_start_reference_forward_and_gradients;label_free_inference;teacher_step;short_train;independent_evaluate;exact_resume',
    'CRANN':'reference_forward_and_gradients;target_time_split;raw_target_mask;batch_one;short_train;independent_evaluate;exact_resume',
    'GRU':'fixed_reference;forward;short_train;CLI_API_equality;exact_resume;real_METR_LA_prefix;CUDA_AMP_resume',
    'STGCN':'reference_forward_and_updates;buffer_dtype;short_train;independent_evaluate;shared_resume;CUDA_lifecycle',
    'DCRNN':'reference_forward_and_updates;curriculum_exact_resume;short_train;independent_evaluate;CUDA_lifecycle',
    'Trafformer':'reference_forward_and_updates;unequal_horizon;accumulation_tail;short_train;shared_resume',
    'ASTGNN':'reference_forward_and_updates;teacher_finetune_exact_resume;short_train;independent_evaluate',
    'MTGNN':'reference_updates;subgraph_exact_resume;short_train;independent_evaluate',
    'HMMM':'direct_algorithm;CLI_API_equality;timestamps;metric_subset;no_training',
    'FNN':'reference_forward;short_train;shared_resume',
    'GWNET':'corrected_reference_forward;graph_buffers;short_train;shared_resume',
    'RNN':'RNN_LSTM_GRU_short_train;variable_length;batch_one;ranking;independent_evaluate;GRU_exact_resume',
    'DeepTTE':'forward;global_local_short_train;independent_evaluate;exact_resume;duration_seconds',
    'LINE':'first_order_reference;second_order_short_train;independent_evaluate;exact_resume;embedding_export',
    'STSSL':'auxiliary_short_train;DWA_exact_resume;deterministic_inference;independent_evaluate',
}


def inventory():
    config=json.loads((ROOT/'libcity/config/task_config.json').read_text(encoding='utf-8'))
    with (ROOT/'docs/modernization/audit/model_compatibility.csv').open(encoding='utf-8-sig',newline='') as stream:
        baseline={(row['task'],row['model']):row for row in csv.DictReader(stream)}
    rows=[]
    for task,values in config.items():
        for name in values['allowed_model']:
            old=baseline[(task,name)]
            native=name
            parameters={}
            if task=='traj_loc_pred' and name in ('GRU','LSTM','RNN'):
                native='RNN'
                parameters={'rnn_type':name}
            elif task=='traffic_state_pred' and name in ('RNN','LSTM'):
                native='Traffic'+name
            spec=MODELS.get(native)
            migrated=spec is not None and spec.task==task
            extras=old['optional_extras']
            if name=='TemplateTLP' or name=='TemplateTSP':
                action='developer_template; not a trainable model'
            elif migrated:
                action='supported representative configuration; expand real-data/hardware coverage separately'
            elif extras:
                action='optional_dependencies_zh.md; replace/isolate dependency then add native data/model/step and acceptance'
            elif name.endswith('Common'):
                action='Common consolidation deferred by plan section 8; migrate only with explicit CPT/window protocol'
            else:
                action='next migration batch: extract pure module and task data; use shared loop or explicit auxiliary/custom train; verify forward/short train/resume'
            rows.append(dict(task=task,model=name,runtime_name=native if migrated else '',
                runtime_params=json.dumps(parameters),migration_status='migrated' if migrated else 'pending',
                training_entry=('direct_evaluate' if spec.kind=='direct' else spec.train or ('step_fn' if spec.step else 'train_loop')) if migrated else 'not_migrated; former='+values.get(name,{}).get('executor','legacy'),
                requires_checkpoint=spec.requires_checkpoint if migrated else 'unverified',
                supports_validation=spec.supports_validation if migrated else 'unverified',
                supports_resume=spec.supports_resume if migrated else 'unverified',dependency_extras=extras,
                actual_validation=VALIDATION[native] if migrated else 'not tested in new runtime',
                baseline_validation=old['validation'],next_action=action))
    return rows


def main():
    rows=inventory()
    path=ROOT/'docs/modernization/runtime/model-migration.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    counts={name:sum(row['migration_status']==name for row in rows) for name in ('migrated','pending')}
    print(json.dumps(dict(entries=len(rows),runtime_models=len(MODELS),**counts)))


if __name__=='__main__':
    main()
