use super::*;
use crate::search::{persist_saved_game_tree_playback, SavedGameTreePlayback};

const CANDIDATE_EVAL_MCTS_SEED: u64 = 0;
const GAME_COMMIT_BATCH_SIZE: usize = 4;

impl GameGenerator {
    pub(super) fn load_atomic_f32(value: &AtomicU32) -> f32 {
        f32::from_bits(value.load(Ordering::SeqCst))
    }

    pub(super) fn store_atomic_f32(target: &AtomicU32, value: f32) {
        target.store(value.to_bits(), Ordering::SeqCst);
    }

    pub(super) fn needs_incumbent_eval_rebaseline(
        incumbent_uses_network: bool,
        current_penalties: (f32, f32),
        candidate_penalties: (f32, f32),
    ) -> bool {
        incumbent_uses_network && current_penalties != candidate_penalties
    }

    fn saved_eval_tree_output_dir(training_data_path: &Path) -> PathBuf {
        training_data_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("analysis")
            .join("eval_trees")
    }

    fn saved_eval_tree_path(
        training_data_path: &Path,
        candidate_step: u64,
        seed: u64,
        total_attack: u32,
    ) -> PathBuf {
        Self::saved_eval_tree_output_dir(training_data_path).join(format!(
            "candidate_step_{candidate_step:09}_worst_seed_{seed}_attack_{total_attack}.json"
        ))
    }

    fn latest_saved_eval_tree_path(training_data_path: &Path) -> PathBuf {
        Self::saved_eval_tree_output_dir(training_data_path)
            .join("latest_worst_candidate_eval_tree.json")
    }

    fn persist_worst_candidate_eval_tree(
        candidate_agent: &MCTSAgent,
        rollout_config: &MCTSConfig,
        training_data_path: &Path,
        candidate: &CandidateModelRequest,
        worst_seed: u64,
        expected_replay_moves: &[ReplayMove],
        expected_total_attack: u32,
        max_placements: u32,
        promoted: bool,
        candidate_avg_attack: f32,
        evaluation_seconds: f32,
    ) -> Option<String> {
        let env = TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, worst_seed);
        let playback = match candidate_agent.play_game_with_trees(&env, max_placements, false) {
            Some(playback) => playback,
            None => {
                eprintln!(
                    "[GameGenerator] Failed to export worst candidate tree for step {} seed {}",
                    candidate.model_step, worst_seed
                );
                return None;
            }
        };

        if playback.total_attack != expected_total_attack
            || playback.replay_moves != expected_replay_moves
        {
            eprintln!(
                "[GameGenerator] Skipping worst candidate tree save for step {} seed {}: replay mismatch (expected attack {}, got {})",
                candidate.model_step,
                worst_seed,
                expected_total_attack,
                playback.total_attack
            );
            return None;
        }

        let saved = SavedGameTreePlayback::from_playback(
            &playback,
            worst_seed,
            rollout_config,
            false,
            &candidate.model_path,
            candidate.model_step,
            promoted,
            candidate_avg_attack,
            evaluation_seconds,
        );
        let exact_path = Self::saved_eval_tree_path(
            training_data_path,
            candidate.model_step,
            worst_seed,
            expected_total_attack,
        );
        if let Err(error) = persist_saved_game_tree_playback(&saved, &exact_path) {
            eprintln!(
                "[GameGenerator] Failed to persist worst candidate tree {}: {}",
                exact_path.display(),
                error
            );
            return None;
        }

        let latest_path = Self::latest_saved_eval_tree_path(training_data_path);
        if let Err(error) = persist_saved_game_tree_playback(&saved, &latest_path) {
            eprintln!(
                "[GameGenerator] Failed to refresh latest worst candidate tree {}: {}",
                latest_path.display(),
                error
            );
        }

        Some(exact_path.display().to_string())
    }
    /// Replay examples as the compact numpy batch the trainer keeps on device:
    /// `(board_bits (N, 25) u8, aux (N, AUX_FEATURES) f32, policy_legal (N, 64) u16,
    /// values (N,) f32, overhang_fields (N,) f32, mask_bits (N, 84) u8)`.
    /// See `crate::replay::compact` for the bit and policy layout.
    pub(super) fn examples_to_numpy<'py>(
        py: Python<'py>,
        examples: &[CompactExample],
    ) -> PyResult<CompactNumpyBatch<'py>> {
        let batch_size = examples.len();

        let mut board_bits = Vec::with_capacity(batch_size * BOARD_BYTES);
        let mut aux = vec![0.0f32; batch_size * AUX_FEATURES];
        let mut policy_legal = Vec::with_capacity(batch_size * POLICY_SLOTS);
        let mut values = Vec::with_capacity(batch_size);
        let mut overhangs = Vec::with_capacity(batch_size);
        let mut mask_bits = Vec::with_capacity(batch_size * MASK_BYTES);

        for (i, ex) in examples.iter().enumerate() {
            board_bits.extend_from_slice(&ex.board_bits);
            let aux_slice = &mut aux[i * AUX_FEATURES..(i + 1) * AUX_FEATURES];
            crate::inference::encode_aux_features(
                aux_slice,
                usize::from(ex.current_piece),
                ex.hold_piece_type(),
                ex.hold_available,
                &ex.next_queue_pieces(),
                ex.placement_count,
                ex.combo,
                ex.back_to_back,
                &ex.next_hidden_piece_probs,
                &ex.column_heights,
                ex.max_column_height,
                &ex.row_fill_counts,
                ex.total_blocks,
                ex.bumpiness,
                ex.holes,
                ex.overhang_fields,
            )
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to encode aux features for replay example {}: {}",
                    i, error
                ))
            })?;
            policy_legal.extend_from_slice(&ex.policy_legal);
            values.push(ex.value);
            overhangs.push(ex.overhang_fields);
            mask_bits.extend_from_slice(&ex.mask_bits);
        }

        Ok((
            PyArray1::from_vec(py, board_bits)
                .reshape([batch_size, BOARD_BYTES])
                .unwrap(),
            PyArray1::from_vec(py, aux)
                .reshape([batch_size, AUX_FEATURES])
                .unwrap(),
            PyArray1::from_vec(py, policy_legal)
                .reshape([batch_size, POLICY_SLOTS])
                .unwrap(),
            PyArray1::from_vec(py, values),
            PyArray1::from_vec(py, overhangs),
            PyArray1::from_vec(py, mask_bits)
                .reshape([batch_size, MASK_BYTES])
                .unwrap(),
        ))
    }

    pub(super) fn persist_snapshot_if_due(
        settings: &WorkerSettings,
        buffer: &SharedBuffer,
        next_snapshot_deadline: &mut Option<Instant>,
    ) {
        if settings.save_interval_seconds <= 0.0 {
            return;
        }
        let interval = Duration::from_secs_f64(settings.save_interval_seconds);
        let now = Instant::now();
        let deadline = *next_snapshot_deadline.get_or_insert_with(|| now + interval);
        if now < deadline {
            return;
        }

        Self::persist_buffer_snapshot(settings, buffer);

        let mut next_deadline = deadline + interval;
        while next_deadline <= now {
            next_deadline += interval;
        }
        *next_snapshot_deadline = Some(next_deadline);
    }

    /// Worker thread main loop.
    pub(super) fn worker_loop(worker_context: WorkerContext) {
        let WorkerContext {
            worker_id,
            is_evaluator_worker,
            settings,
            shared,
        } = worker_context;
        let mut agent = MCTSAgent::new(settings.config.clone());
        let mut loaded_model_version = u64::MAX;
        let mut loaded_model_step = 0u64;
        let mut loaded_with_network = !shared.incumbent.uses_network.load(Ordering::SeqCst);
        let mut pending_results: Vec<CompletedGameResult> =
            Vec::with_capacity(GAME_COMMIT_BATCH_SIZE);

        while shared.running.load(Ordering::SeqCst)
            && !Self::sync_incumbent_agent_if_needed(
                &settings,
                &shared.incumbent,
                &mut agent,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
            )
        {
            thread::sleep(Duration::from_millis(500));
        }

        // Only worker 0 submits periodic replay snapshots to the background writer.
        let is_save_worker = worker_id == 0;
        let mut next_snapshot_deadline = if settings.save_interval_seconds > 0.0 {
            Some(Instant::now() + Duration::from_secs_f64(settings.save_interval_seconds))
        } else {
            None
        };

        // Main generation loop
        while shared.running.load(Ordering::SeqCst) {
            if !Self::sync_incumbent_agent_if_needed(
                &settings,
                &shared.incumbent,
                &mut agent,
                &mut loaded_model_version,
                &mut loaded_model_step,
                &mut loaded_with_network,
                worker_id,
            ) {
                thread::sleep(Duration::from_millis(200));
                continue;
            }

            if is_evaluator_worker {
                let maybe_candidate = {
                    let mut pending = shared.pending_candidate.write().unwrap();
                    pending.take()
                };
                if let Some(candidate) = maybe_candidate {
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(to_commit, &shared);
                    }

                    let current_incumbent_step = shared.incumbent.model_step.load(Ordering::SeqCst);
                    if candidate.model_step <= current_incumbent_step {
                        let incumbent_path = shared.incumbent.model_path.read().unwrap().clone();
                        Self::remove_model_artifacts_if_safe(
                            &candidate.model_path,
                            &settings.bootstrap_model_path,
                            &incumbent_path,
                            None,
                        );
                        continue;
                    }

                    {
                        let mut evaluating = shared.evaluating_candidate.write().unwrap();
                        *evaluating = Some(candidate.clone());
                    }

                    Self::run_candidate_evaluation(worker_id, candidate, &settings, &shared);

                    {
                        let mut evaluating = shared.evaluating_candidate.write().unwrap();
                        *evaluating = None;
                    }
                    loaded_model_version = u64::MAX;
                    loaded_with_network = !shared.incumbent.uses_network.load(Ordering::SeqCst);

                    if is_save_worker {
                        Self::persist_snapshot_if_due(
                            &settings,
                            shared.buffer.as_ref(),
                            &mut next_snapshot_deadline,
                        );
                    }
                    continue;
                }
            }

            // Play one game. Both `add_noise` and `visit_sampling_epsilon`
            // are read live so trainer/operator overrides take effect on
            // the next game without rebuilding the agent.
            let live_add_noise = shared.live_overrides.add_noise.load(Ordering::SeqCst);
            let live_visit_sampling_epsilon =
                Self::load_atomic_f32(&shared.live_overrides.visit_sampling_epsilon);
            agent.set_visit_sampling_epsilon(live_visit_sampling_epsilon);
            let env = TetrisEnv::new(BOARD_WIDTH, BOARD_HEIGHT);
            let replay_seed = env.seed;
            if let Some((result, replay_moves)) =
                agent.play_game_on_env(env, settings.max_placements, live_add_noise)
            {
                let total_attack = result.total_attack;
                let num_moves = result.num_moves;
                pending_results.push(CompletedGameResult {
                    result,
                    completed_time_s: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("system time should be after Unix epoch")
                        .as_secs_f64(),
                    replay: Some(GameReplay {
                        seed: replay_seed,
                        moves: replay_moves,
                        total_attack,
                        num_moves,
                    }),
                });
                if pending_results.len() >= GAME_COMMIT_BATCH_SIZE {
                    let to_commit = std::mem::take(&mut pending_results);
                    Self::commit_game_results_batch(to_commit, &shared);
                }

                // Periodically save to disk for resume capability based on
                // wall-clock time.
                if is_save_worker {
                    if !pending_results.is_empty() {
                        let to_commit = std::mem::take(&mut pending_results);
                        Self::commit_game_results_batch(to_commit, &shared);
                    }
                    Self::persist_snapshot_if_due(
                        &settings,
                        shared.buffer.as_ref(),
                        &mut next_snapshot_deadline,
                    );
                }
            }
        }

        if !pending_results.is_empty() {
            let to_commit = std::mem::take(&mut pending_results);
            Self::commit_game_results_batch(to_commit, &shared);
        }

        eprintln!("[GameGenerator] Worker {} exiting", worker_id);
    }

    pub(super) fn sync_incumbent_agent_if_needed(
        settings: &WorkerSettings,
        incumbent: &IncumbentState,
        agent: &mut MCTSAgent,
        loaded_model_version: &mut u64,
        loaded_model_step: &mut u64,
        loaded_with_network: &mut bool,
        worker_id: usize,
    ) -> bool {
        let target_version = incumbent.model_version.load(Ordering::SeqCst);
        let target_uses_network = incumbent.uses_network.load(Ordering::SeqCst);
        if *loaded_model_version == target_version && *loaded_with_network == target_uses_network {
            return true;
        }

        let model_path = if target_uses_network {
            Some(incumbent.model_path.read().unwrap().clone())
        } else {
            None
        };
        let target_nn_value_weight = Self::load_atomic_f32(&incumbent.nn_value_weight);
        let target_death_penalty = Self::load_atomic_f32(&incumbent.death_penalty);
        let target_overhang_penalty_weight =
            Self::load_atomic_f32(&incumbent.overhang_penalty_weight);
        let Some(new_agent) = Self::create_rollout_agent(
            &settings.config,
            target_uses_network,
            settings.non_network_num_simulations,
            target_nn_value_weight,
            target_death_penalty,
            target_overhang_penalty_weight,
            model_path.as_deref(),
            worker_id,
            "incumbent",
        ) else {
            return false;
        };

        *loaded_model_step = incumbent.model_step.load(Ordering::SeqCst);
        if worker_id == 0 {
            if target_uses_network {
                eprintln!(
                    "[GameGenerator] Loaded incumbent NN model step {} ({} workers, sims={}, nn_value_weight={:.6}, death_penalty={:.3}, overhang_penalty_weight={:.3})",
                    *loaded_model_step,
                    settings.num_workers,
                    settings.config.num_simulations,
                    target_nn_value_weight,
                    target_death_penalty,
                    target_overhang_penalty_weight
                );
            } else {
                eprintln!(
                    "[GameGenerator] Using no-network incumbent at step {} ({} workers, sims={})",
                    *loaded_model_step, settings.num_workers, settings.non_network_num_simulations
                );
            }
        }

        *agent = new_agent;
        *loaded_model_version = target_version;
        *loaded_with_network = target_uses_network;
        true
    }

    pub(super) fn run_candidate_evaluation(
        worker_id: usize,
        candidate: CandidateModelRequest,
        settings: &WorkerSettings,
        shared: &WorkerSharedState,
    ) {
        let eval_config = Self::build_candidate_eval_config(&settings.config);
        let incumbent_uses_network_before = shared.incumbent.uses_network.load(Ordering::SeqCst);
        let incumbent_step_before = shared.incumbent.model_step.load(Ordering::SeqCst);
        let incumbent_nn_value_weight_before =
            Self::load_atomic_f32(&shared.incumbent.nn_value_weight);
        let previous_incumbent_avg_attack =
            Self::load_atomic_f32(&shared.incumbent.eval_avg_attack);
        let incumbent_path_before = shared.incumbent.model_path.read().unwrap().clone();

        let current_incumbent_death_penalty =
            Self::load_atomic_f32(&shared.incumbent.death_penalty);
        let current_incumbent_overhang_penalty_weight =
            Self::load_atomic_f32(&shared.incumbent.overhang_penalty_weight);
        let candidate_death_penalty = candidate.death_penalty;
        let candidate_overhang_penalty_weight = candidate.overhang_penalty_weight;
        let candidate_rollout_config = Self::build_rollout_config(
            &eval_config,
            true,
            settings.non_network_num_simulations,
            candidate.nn_value_weight,
            candidate_death_penalty,
            candidate_overhang_penalty_weight,
        );
        let Some(candidate_agent) = Self::create_rollout_agent(
            &eval_config,
            true,
            settings.non_network_num_simulations,
            candidate.nn_value_weight,
            candidate_death_penalty,
            candidate_overhang_penalty_weight,
            Some(&candidate.model_path),
            worker_id,
            "candidate",
        ) else {
            eprintln!(
                "[GameGenerator] Worker {} failed to load candidate step {} from {}",
                worker_id,
                candidate.model_step,
                candidate.model_path.display()
            );
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &settings.bootstrap_model_path,
                &incumbent_path_before,
                None,
            );
            return;
        };

        let eval_start = Instant::now();

        // Play candidate games on fixed seeds for consistent benchmarking.
        struct CandidateGameResult {
            seed: u64,
            game_result: GameResult,
            replay_moves: Vec<ReplayMove>,
        }

        let mut candidate_results: Vec<CandidateGameResult> =
            Vec::with_capacity(settings.candidate_eval_seeds.len());
        for &seed in settings.candidate_eval_seeds.iter() {
            if !shared.running.load(Ordering::SeqCst) {
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            }

            // Deterministic fixed-seed evaluation:
            // same seed, no root noise, deterministic MCTS seed in eval_config.
            let candidate_outcome = candidate_agent.play_game_on_env(
                TetrisEnv::with_seed(BOARD_WIDTH, BOARD_HEIGHT, seed),
                settings.max_placements,
                false,
            );
            if let Some((candidate_result, replay)) = candidate_outcome {
                candidate_results.push(CandidateGameResult {
                    seed,
                    game_result: candidate_result,
                    replay_moves: replay,
                });
            } else {
                // Keep incumbent baseline and candidate sample size comparable:
                // a failed rollout invalidates this gate decision.
                eprintln!(
                    "[GameGenerator] Candidate eval failed on seed {}; rejecting candidate step {}",
                    seed, candidate.model_step
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            }
        }

        if candidate_results.is_empty() {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &settings.bootstrap_model_path,
                &incumbent_path_before,
                None,
            );
            return;
        }

        // Build per-game results and find best/worst for replay serialization
        let per_game_results: Vec<(u64, u32, u32, u32)> = candidate_results
            .iter()
            .map(|r| {
                (
                    r.seed,
                    r.game_result.total_attack,
                    r.game_result.stats.total_lines,
                    r.game_result.num_moves,
                )
            })
            .collect();
        let per_game_prediction_metrics: Vec<(u64, f32, f32, f32, u32)> = candidate_results
            .iter()
            .map(|r| {
                (
                    r.seed,
                    r.game_result.trajectory_predicted_total_attack_variance,
                    r.game_result.trajectory_predicted_total_attack_std,
                    r.game_result.trajectory_predicted_total_attack_rmse,
                    r.game_result.trajectory_predicted_total_attack_count,
                )
            })
            .collect();

        let best_idx = candidate_results
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.game_result.total_attack)
            .map(|(i, _)| i);
        let worst_idx = candidate_results
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| r.game_result.total_attack)
            .map(|(i, _)| i);

        let build_replay = |idx: usize| -> GameReplay {
            let r = &candidate_results[idx];
            GameReplay {
                seed: r.seed,
                moves: r.replay_moves.clone(),
                total_attack: r.game_result.total_attack,
                num_moves: r.game_result.num_moves,
            }
        };

        let best_game_replay = best_idx.map(build_replay);
        let worst_game_replay = worst_idx.and_then(|i| {
            // Avoid duplicating if best and worst are the same game
            if Some(i) == best_idx {
                None
            } else {
                Some(build_replay(i))
            }
        });

        let candidate_games = candidate_results.len() as u64;
        let candidate_total_attack: u64 = candidate_results
            .iter()
            .map(|r| r.game_result.total_attack as u64)
            .sum();
        let candidate_avg_attack = candidate_total_attack as f32 / candidate_games as f32;
        let candidate_attack_variance = candidate_results
            .iter()
            .map(|r| {
                let diff = r.game_result.total_attack as f32 - candidate_avg_attack;
                diff * diff
            })
            .sum::<f32>()
            / candidate_games as f32;
        let mut incumbent_avg_attack = previous_incumbent_avg_attack;
        if Self::needs_incumbent_eval_rebaseline(
            incumbent_uses_network_before,
            (
                current_incumbent_death_penalty,
                current_incumbent_overhang_penalty_weight,
            ),
            (candidate_death_penalty, candidate_overhang_penalty_weight),
        ) {
            let Some(incumbent_agent) = Self::create_rollout_agent(
                &eval_config,
                true,
                settings.non_network_num_simulations,
                incumbent_nn_value_weight_before,
                candidate_death_penalty,
                candidate_overhang_penalty_weight,
                Some(&incumbent_path_before),
                worker_id,
                "incumbent-baseline",
            ) else {
                eprintln!(
                    "[GameGenerator] Worker {} failed to load incumbent baseline from {} for adjusted evaluation",
                    worker_id,
                    incumbent_path_before.display()
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            };
            let Some(recomputed_incumbent_avg_attack) =
                crate::runtime::evaluation::evaluate_avg_attack_on_fixed_seeds(
                    &incumbent_agent,
                    &settings.candidate_eval_seeds,
                    settings.max_placements,
                    false,
                    || shared.running.load(Ordering::SeqCst),
                )
            else {
                eprintln!(
                    "[GameGenerator] Failed to recompute incumbent baseline for adjusted penalties; rejecting candidate step {}",
                    candidate.model_step
                );
                Self::remove_model_artifacts_if_safe(
                    &candidate.model_path,
                    &settings.bootstrap_model_path,
                    &incumbent_path_before,
                    None,
                );
                return;
            };
            incumbent_avg_attack = recomputed_incumbent_avg_attack;
        }

        let auto_promoted = previous_incumbent_avg_attack == 0.0 && !incumbent_uses_network_before;
        let force_promoted = candidate.force_promote;
        let promoted =
            auto_promoted || force_promoted || candidate_avg_attack >= incumbent_avg_attack;
        let promoted_nn_value_weight = if promoted {
            candidate.nn_value_weight
        } else {
            incumbent_nn_value_weight_before
        };
        let evaluation_seconds = eval_start.elapsed().as_secs_f32();
        let worst_game_tree_path = if settings.save_eval_trees {
            worst_idx.and_then(|idx| {
                let worst = &candidate_results[idx];
                Self::persist_worst_candidate_eval_tree(
                    &candidate_agent,
                    &candidate_rollout_config,
                    &settings.training_data_path,
                    &candidate,
                    worst.seed,
                    &worst.replay_moves,
                    worst.game_result.total_attack,
                    settings.max_placements,
                    promoted,
                    candidate_avg_attack,
                    evaluation_seconds,
                )
            })
        } else {
            None
        };

        if promoted {
            let previous_incumbent_path = shared.incumbent.model_path.read().unwrap().clone();
            {
                let mut incumbent_path = shared.incumbent.model_path.write().unwrap();
                *incumbent_path = candidate.model_path.clone();
            }
            shared.incumbent.uses_network.store(true, Ordering::SeqCst);
            shared
                .incumbent
                .model_step
                .store(candidate.model_step, Ordering::SeqCst);
            Self::store_atomic_f32(&shared.incumbent.nn_value_weight, candidate.nn_value_weight);
            Self::store_atomic_f32(&shared.incumbent.death_penalty, candidate_death_penalty);
            Self::store_atomic_f32(
                &shared.incumbent.overhang_penalty_weight,
                candidate_overhang_penalty_weight,
            );
            shared
                .incumbent
                .model_version
                .fetch_add(1, Ordering::SeqCst);
            Self::store_atomic_f32(&shared.incumbent.eval_avg_attack, candidate_avg_attack);

            Self::remove_model_artifacts_if_safe(
                &previous_incumbent_path,
                &settings.bootstrap_model_path,
                &candidate.model_path,
                None,
            );

            let promotion_reason = if force_promoted {
                " (force_promote runtime override)"
            } else if auto_promoted {
                " (auto-promoted: no prior network incumbent)"
            } else {
                ""
            };
            eprintln!(
                "[GameGenerator] Promoted candidate step {} (avg_attack {:.3} >= incumbent {:.3}, games={}, nn_value_weight {:.6} -> {:.6}, eval trajectories discarded from replay){}",
                candidate.model_step,
                candidate_avg_attack,
                incumbent_avg_attack,
                candidate_games,
                incumbent_nn_value_weight_before,
                candidate.nn_value_weight,
                promotion_reason
            );
        } else {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &settings.bootstrap_model_path,
                &incumbent_path_before,
                None,
            );

            eprintln!(
                "[GameGenerator] Rejected candidate step {} (avg_attack {:.3} < incumbent {:.3}, games={}, candidate_nn_value_weight {:.6}, incumbent_nn_value_weight {:.6})",
                candidate.model_step,
                candidate_avg_attack,
                incumbent_avg_attack,
                candidate_games,
                candidate.nn_value_weight,
                incumbent_nn_value_weight_before
            );
        }

        let promoted_death_penalty = Self::load_atomic_f32(&shared.incumbent.death_penalty);
        let promoted_overhang_penalty_weight =
            Self::load_atomic_f32(&shared.incumbent.overhang_penalty_weight);

        shared
            .model_eval_events
            .write()
            .unwrap()
            .push_back(ModelEvalEvent {
                incumbent_step: incumbent_step_before,
                incumbent_uses_network: incumbent_uses_network_before,
                incumbent_avg_attack,
                incumbent_nn_value_weight: incumbent_nn_value_weight_before,
                candidate_step: candidate.model_step,
                candidate_games,
                candidate_avg_attack,
                candidate_attack_variance,
                candidate_nn_value_weight: candidate.nn_value_weight,
                promoted_nn_value_weight,
                promoted_death_penalty,
                promoted_overhang_penalty_weight,
                promoted,
                auto_promoted,
                force_promoted,
                evaluation_seconds,
                best_game_replay,
                worst_game_replay,
                worst_game_tree_path,
                per_game_results,
                per_game_prediction_metrics,
            });
    }

    pub(super) fn build_candidate_eval_config(base_config: &MCTSConfig) -> MCTSConfig {
        let mut eval_config = base_config.clone();
        eval_config.seed = Some(CANDIDATE_EVAL_MCTS_SEED);
        eval_config.visit_sampling_epsilon = 0.0;
        eval_config
    }

    pub(super) fn build_rollout_config(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
    ) -> MCTSConfig {
        let mut rollout_config = base_config.clone();
        rollout_config.num_simulations = if uses_network {
            base_config.num_simulations
        } else {
            non_network_num_simulations
        };
        rollout_config.nn_value_weight = nn_value_weight;
        rollout_config.death_penalty = death_penalty;
        rollout_config.overhang_penalty_weight = overhang_penalty_weight;
        rollout_config
    }

    pub(super) fn create_rollout_agent(
        base_config: &MCTSConfig,
        uses_network: bool,
        non_network_num_simulations: u32,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
        model_path: Option<&Path>,
        worker_id: usize,
        role: &str,
    ) -> Option<MCTSAgent> {
        // Keep rollout behavior consistent across training and evaluator code paths,
        // including visit_sampling_epsilon and all other shared MCTS settings.
        let rollout_config = Self::build_rollout_config(
            base_config,
            uses_network,
            non_network_num_simulations,
            nn_value_weight,
            death_penalty,
            overhang_penalty_weight,
        );
        let mut agent = MCTSAgent::new(rollout_config);
        if uses_network {
            let model_path = model_path.expect("network rollout requires model path");
            let path_str = model_path
                .to_str()
                .expect("Model path contains invalid UTF-8");
            if !agent.load_model(path_str) {
                eprintln!(
                    "[GameGenerator] Worker {} failed to load {} model {}",
                    worker_id,
                    role,
                    model_path.display()
                );
                return None;
            }
        }
        Some(agent)
    }

    pub(super) fn commit_game_results_batch(
        results: Vec<CompletedGameResult>,
        shared: &WorkerSharedState,
    ) -> usize {
        if results.is_empty() {
            return 0;
        }

        let mut all_examples: Vec<TrainingExample> = Vec::new();
        let mut total_examples = 0u64;
        let mut completed_infos: Vec<LastGameInfo> = Vec::with_capacity(results.len());

        for completed in results {
            let CompletedGameResult {
                result,
                completed_time_s,
                replay,
            } = completed;
            let GameResult {
                mut examples,
                total_attack,
                num_moves,
                avg_valid_actions,
                max_valid_actions,
                stats,
                tree_stats,
                avg_overhang_fields,
                cache_hits,
                cache_misses,
                cache_size,
                tree_reuse_hits,
                tree_reuse_misses,
                tree_reuse_carry_fraction,
                traversal_total,
                traversal_expansions,
                traversal_terminal_ends,
                traversal_horizon_ends,
                traversal_expansion_fraction,
                traversal_terminal_fraction,
                traversal_horizon_fraction,
                trajectory_predicted_total_attack_count,
                trajectory_predicted_total_attack_variance,
                trajectory_predicted_total_attack_std,
                trajectory_predicted_total_attack_rmse,
                ..
            } = result;

            let game_number = shared.games_generated.fetch_add(1, Ordering::SeqCst) + 1;
            for example in &mut examples {
                example.game_number = game_number;
                example.game_total_attack = total_attack;
            }

            total_examples += examples.len() as u64;
            all_examples.extend(examples);

            completed_infos.push(LastGameInfo {
                game_number,
                completed_time_s,
                stats,
                total_attack,
                avg_overhang_fields,
                num_moves,
                avg_valid_actions,
                max_valid_actions,
                tree_stats,
                cache_hits,
                cache_misses,
                cache_size,
                tree_reuse_hits,
                tree_reuse_misses,
                tree_reuse_carry_fraction,
                traversal_total,
                traversal_expansions,
                traversal_terminal_ends,
                traversal_horizon_ends,
                traversal_expansion_fraction,
                traversal_terminal_fraction,
                traversal_horizon_fraction,
                trajectory_predicted_total_attack_count,
                trajectory_predicted_total_attack_variance,
                trajectory_predicted_total_attack_std,
                trajectory_predicted_total_attack_rmse,
                replay,
            });
        }

        if !all_examples.is_empty() {
            shared.buffer.add_examples(all_examples);
        }
        if total_examples > 0 {
            shared
                .examples_generated
                .fetch_add(total_examples, Ordering::SeqCst);
        }
        let committed_games = completed_infos.len();
        if committed_games > 0 {
            shared
                .completed_games
                .write()
                .unwrap()
                .extend(completed_infos);
        }

        committed_games
    }

    pub(super) fn persist_buffer_snapshot(settings: &WorkerSettings, buffer: &SharedBuffer) {
        settings
            .snapshot_persister
            .submit_buffer_snapshot(buffer, &settings.training_data_path);
    }

    pub(super) fn model_artifact_paths(model_path: &Path) -> [PathBuf; 7] {
        let base_path = model_path.with_extension("");
        [
            model_path.to_path_buf(),
            model_path.with_extension("onnx.data"),
            base_path.with_extension("conv.onnx"),
            base_path.with_extension("conv.onnx.data"),
            base_path.with_extension("heads.onnx"),
            base_path.with_extension("heads.onnx.data"),
            base_path.with_extension("fc.bin"),
        ]
    }

    pub(super) fn remove_model_artifacts(model_path: &Path) {
        for artifact_path in Self::model_artifact_paths(model_path) {
            if let Err(error) = fs::remove_file(&artifact_path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    eprintln!(
                        "[GameGenerator] Failed to remove model artifact {}: {}",
                        artifact_path.display(),
                        error
                    );
                }
            }
        }
    }

    pub(super) fn remove_model_artifacts_if_safe(
        model_path: &Path,
        bootstrap_model_path: &Path,
        incumbent_model_path: &Path,
        evaluating_model_path: Option<&Path>,
    ) {
        if model_path == bootstrap_model_path {
            return;
        }
        if model_path == incumbent_model_path {
            return;
        }
        if let Some(path) = evaluating_model_path {
            if model_path == path {
                return;
            }
        }
        Self::remove_model_artifacts(model_path);
    }

    pub(super) fn cleanup_queued_candidate_artifacts(&self) {
        let incumbent_path = self.incumbent_model_path.read().unwrap().clone();

        let pending = self.pending_candidate.write().unwrap().take();
        if let Some(candidate) = pending {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
        }

        let evaluating = self.evaluating_candidate.write().unwrap().take();
        if let Some(candidate) = evaluating {
            Self::remove_model_artifacts_if_safe(
                &candidate.model_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
        }
    }
}

impl Drop for GameGenerator {
    fn drop(&mut self) {
        // Ensure all threads are stopped when generator is dropped
        self.running.store(false, Ordering::SeqCst);
        for handle in self.thread_handles.drain(..) {
            let _ = handle.join();
        }
        self.cleanup_queued_candidate_artifacts();
    }
}
