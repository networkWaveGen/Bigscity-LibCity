"""Inventory every original task/model without treating import as runtime migration."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from libcity.model.runtime_registry import MODELS


VALIDATION={
    'DGCN':'corrected_original_default_sixty_frame_forecast_all_effective_gradients_and_BN_states;coarse_CPT_chunk_axis_reference;full_default_64_channel_network_and_dropout_training_reference;dynamic_Chebyshev_orders_and_odd_kernels;unnormalized_directed_scaled_Laplacian_and_row_cancellation_guard;explicit_CPT_family_and_oldest_first_chunk_order;arbitrary_active_chunks_horizon_features_heads_batch_one_float64;K1_omits_unused_graph_parameters;full_train_independent_evaluate_dropout_BN_Adam_exponential_RNG_exact_resume;real_default_12_step_protocol;raw_cache_train_scaling_target_isolation;single_node_tail_CPU_AMP_singleton_1d_2d_BN',
    'STPGCN':'corrected_original_registered_Gaussian_full_relation_forecast_and_all_gradients;six_layer_hop_beta_feature_combinations_and_default_64_3_8_network;exact_relation_factorization_with_bounded_memory;true_shortest_hops_separate_range_and_unreachable_bucket;all_Gaussian_graph_selection_parameters_registered;actual_UTC_weekday_time_slot_integer_history;causal_temporal_padding;label_free_batch_one_float64;target_time_split_train_only_scaling_cache_and_target_isolation;full_train_independent_evaluate_all_layer_Adam_scheduler_RNG_exact_resume;single_node_tail_CPU_AMP;spawn_worker',
    'STSGCN':'sixteen_original_individual_sharing_GLU_relu_embedding_forecast_and_all_gradient_references_with_fresh_mask_graph;default_four_block_reference;directed_owned_three_frame_graph;repeated_mask_updates_without_retain_graph;nonmutating_position_embeddings;label_free_batch_independence;multi_target_multi_horizon_batch_one_float64;raw_scale_Huber_threshold_and_mask_gradient_reference;full_train_independent_evaluate_mask_Adam_scheduler_RNG_exact_resume;single_node_empty_graph_tail_CPU_AMP',
    'ResLSTM':'corrected_original_period_channel_SAME_pool_batch_first_forward_and_all_gradients;shared_historical_external_schema;configurable_two_flow_columns;no_cross_sample_prediction_or_gradient;all_pool_dimension_parities_and_negative_input_reference;derived_node_window_channels;multi_horizon_multi_target_batch_one_float64;full_train_independent_evaluate_BN_Adam_multistep_scheduler_RNG_exact_resume;raw_cache_and_future_target_isolation;service_closures_between_complete_seasonal_windows;singleton_true_tail_BN_CPU_AMP;no_unused_residual_projection',
    'DKFN':'corrected_original_dual_recurrent_prediction_and_all_gradient_reference;per_sample_variance_fusion_with_author_coefficient_formula;positive_softplus_coefficient_and_chain_rule;no_cross_sample_gradient;static_column_degree_normalized_graph_buffers;clamped_and_plain_powers;rolling_known_future_only;scalar_constant_isolated_zero_variance_finite_gradients;batch_one_float64;full_train_independent_evaluate_RMSprop_scheduler_RNG_exact_resume;CPU_AMP',
    'CONVGCN':'original_graph_projection_and_corrected_node_channel_Conv3d_forecast_all_gradients;service_period_seasonal_window_offsets;service_closures_between_complete_windows;distinct_second_target_and_complete_final_window;immutable_self_loop_buffer;multi_output_multistep_batch_one_float64;full_train_independent_evaluate_optimizer_scheduler_RNG_exact_resume;raw_cache_reuse_and_invalidation;CPU_AMP;zero_mask_raw_scale',
    'MultiSTGCnet':'original_graph_LSTM_and_corrected_recent_daily_branch_forecast_all_gradients;explicit_finite_listed_distance_bands;maximum_equal_zero_distance_and_isolated_nodes;no_source_graph_mutation;frame_CPT_order;active_branch_only_networks;multi_output_multistep_batch_one_float64;full_train_independent_evaluate_optimizer_scheduler_RNG_exact_resume;distance_source_and_ID_guard;CPU_AMP;binary_unit_distance_and_directional_symmetry',
    'STResNet':'corrected_spatial_feature_axis_original_forward_and_all_gradient_reference;all_CPT_branches;explicit_grid_ID_order;known_future_external_only;multi_horizon_multi_output_batch_one_float64;full_train_independent_evaluate_BatchNorm_optimizer_scheduler_RNG_exact_resume;grid_and_point_atomic_inputs;target_time_split_train_only_scaling;grid_cache_invalidation;grid_source_resume_guard;singleton_BatchNorm_tail_CPU_AMP;double_precision_raw_masked_loss',
    'ACFM':'corrected_spatial_feature_axis_original_forward_and_all_gradient_reference;five_valid_residual_split_modes_with_explicit_dimension_guards;shared_temporal_weights;general_attention_hidden_width;local_global_and_single_branch_recurrence;explicit_history_and_known_future_external;multi_horizon_multi_output_batch_one_float64;full_train_independent_evaluate_dropout_BatchNorm_optimizer_scheduler_RNG_exact_resume;grid_and_point_atomic_inputs;grid_cache_invalidation;singleton_BatchNorm_tail_CPU_AMP',
    'GeomGCN':'corrected_original_edge_partition_dense_forecast_embedding_and_all_gradient_reference;all_sixteen_partition_head_merge_settings;full_in_degree_normalization;no_mutable_DGL_graph_or_forward_filesystem;full_graph_short_train;independent_evaluate;dropout_optimizer_scheduler_RNG_exact_resume;real_ID_embedding_export;label_free_batch_one_float64;empty_singleton_graph_and_deduplicated_self_loops;two_thousand_node_sparse_graph;CPU_AMP;double_precision_masked_reconstruction_loss',
    'Node2Vec':'original_weighted_directed_and_undirected_alias_walk_reference;isolated_node_coverage;bounded_second_order_transition_cache;pure_local_RNG;actual_Gensim_Word2Vec_weight_reference;streamed_corpus;real_ID_safe_tensor_weights;train_and_independent_evaluate;cross_process_determinism;evaluation_without_Gensim;no_Torch_optimizer_or_writer;honest_no_validation_resume_tuning;graph_ID_configuration_checkpoint_guard;two_thousand_node_sparse_graph',
    'DeepWalk':'corrected_zero_start_original_walk_reference;restart_and_isolated_nodes;unshared_default_RNG;actual_Gensim_Word2Vec_HS_and_negative_sampling_weight_reference;streamed_corpus;real_ID_safe_tensor_weights;train_and_independent_evaluate;cross_process_determinism;evaluation_without_Gensim;no_Torch_optimizer_or_writer;honest_no_validation_resume_tuning;graph_ID_configuration_checkpoint_guard;two_thousand_node_sparse_graph',
    'TTPNet':'original_short_long_speed_road_biLSTM_cumulative_forecast_and_all_gradients_with_double_raw_normalization;explicit_departure_known_snapshot_guard;train_only_normalization;real_road_ID_pretrained_embedding_alignment;canonical_complete_trip_splits;kmh_mps_and_km_m_equivalence;vectorized_branch_and_cumulative_pooling;masked_route_relative_RMSE_objective_and_gradients;exact_fit_finite_gradient;short_train;independent_evaluate_seconds_export;dropout_exact_resume;batch_padding_label_isolation;batch_one_float64;two_point_routes_variable_embedding_width;multiday_seconds_correct_weekend;CPU_AMP_zero_speeds_constant_features',
    'GeoSAN':'frozen_TorchText_PAD_frequency_lexical_vocabulary_contract_without_legacy_dependencies;correct_lon_lat_Mercator_quadkeys;four_original_position_decoder_score_and_all_gradient_references;weighted_binary_importance_loss_and_gradient_reference;bounded_haversine_radian_KNN_uniform_sampling_and_exhaustion;canonical_target_time_samples;static_POI_vocabulary_and_training_only_visited_statistics;causal_padding_batch_chunk_independence;selected_candidate_full_gradient_reference;corrected_fusion_user_time_and_odd_width;short_train;independent_evaluate;sampled_dropout_RNG_exact_resume;batch_one_float64;single_location_positive_only;CPU_AMP;blocked_legacy_import_actual_train',
    'LSTPM':'original_two_level_session_attention_dilated_recurrence_and_all_gradients;carried_history_state_with_explicit_boundaries;training_only_atomic_time_similarity;causal_nearest_earlier_spatial_parents_with_latest_ties;completed_session_centroid_distances;cold_start_preserves_benchmark_samples;short_train;independent_evaluate;exact_resume;batch_padding_independence;batch_one_float64;CPU_AMP',
    'CARA':'author_Keras_gate_activation_corrected_recurrence_candidate_score_and_all_gradients;per_sample_BPR;causal_query_clock_and_real_time_gaps;correct_spherical_fractional_distances;bounded_negative_sampling_with_exhaustion;prefix_reuse_selected_full_gradient_reference;128_step_stability;short_train;independent_evaluate;sampled_RNG_exact_resume;batch_padding_chunk_independence;batch_one_float64;single_location_zero_updates;CPU_AMP',
    'STAN':'original_two_level_interpolation_attention_scores_and_all_gradients_with_explicit_key_mask_fix;static_spherical_diameter_reference_without_dense_POI_matrix;causal_query_time_and_train_bounds;unique_shared_positive_sampled_softmax;all_class_and_duplicate_batches;selected_vs_full_gradient_reference;short_train;independent_evaluate;sampled_RNG_exact_resume;batch_padding_chunk_independence;batch_one_float64;single_location_zero_bounds;CPU_AMP',
    'SERM':'intact_trainable_pretrained_vectors;original_onehot_mean_classifier_and_all_gradients;static_real_POI_ID_alignment;streamed_selected_vocabulary;missing_and_all_OOV_semantics;external_vector_source_resume_guard;batch_padding_label_isolation;short_train;independent_evaluate;RMSprop_exact_resume;batch_one_float64;CPU_AMP',
    'ATSTLSTM':'original_valid_candidate_scores_normalized_BPR_and_all_gradients;masked_attention;target_free_full_vocabulary_geometry;WGS84_vectorized_Karney_reference_and_antipodal_fallback;bounded_owned_candidate_cache;train_only_popularity_without_positive_negatives;selected_training_matches_full_gradients;short_train;independent_evaluate;sampled_RNG_exact_resume;batch_one_float64;CPU_AMP;spawn_worker',
    'HSTLSTM':'corrected_memory_cell_and_gradient_reference;original_classifier_with_explicit_memory_fix;fractional_vectorized_slot_interpolation;WGS84_observed_prefix_intervals;train_only_time_and_distance_bounds;batch_padding_label_isolation;short_train;independent_evaluate;exact_resume;batch_one_float64;degenerate_intervals;CPU_AMP',
    'STRNN':'original_transition_matrix_per_candidate_score_and_gradient_reference;label_free_asof_time_and_geometry;training_prefix_time_bounds;batch_and_chunk_independence;full_vocabulary_NLL;short_train;independent_evaluate;exact_resume;batch_one_float64;degenerate_bounds;CPU_AMP',
    'DeepMove':'four_original_nonpadded_recurrent_attention_and_gradient_references;masked_history_and_batch_independence;cold_start_preserves_benchmark_samples;causal_completed_session_banks;concat_attention;RNN_GRU_LSTM_short_train;independent_evaluate;dropout_exact_resume;batch_one_float64;CPU_AMP',
    'STG2Seq':'independent_graph_time_convolution_and_attention_axis_reference_gradients;explicit_teacher_and_future_features;full_depth_train;independent_evaluate;batchnorm_exact_resume;multi_target_single_node_batch_one_float64;CPU_AMP',
    'TimeMixer':'nine_original_branch_forecast_and_gradient_references;avg_max_conv;both_channel_modes;local_normalization_state;time_axis_FFT_and_top_k;odd_scales_even_kernel;multi_target_batch_one_float64;short_train;independent_evaluate;dropout_exact_resume;CPU_AMP_FFT;zero_scale_layers',
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
            if migrated and name in ('GeoSAN','GeomGCN'):
                extras=''  # Native vocabulary/message passing no longer need legacy backends.
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
