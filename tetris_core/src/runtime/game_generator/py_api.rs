use super::*;

impl GameGenerator {
    fn invalid_generator_arg(message: impl Into<String>) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(message.into())
    }

    fn validate_new_args(
        max_placements: u32,
        max_examples: usize,
        save_interval_seconds: f64,
        num_workers: usize,
        candidate_gating_enabled: bool,
        candidate_eval_seeds: &[u64],
        non_network_num_simulations: u32,
        nn_value_weight: f32,
    ) -> PyResult<()> {
        if max_placements == 0 {
            return Err(Self::invalid_generator_arg("max_placements must be > 0"));
        }
        if max_examples == 0 {
            return Err(Self::invalid_generator_arg("max_examples must be > 0"));
        }
        if !save_interval_seconds.is_finite() || save_interval_seconds < 0.0 {
            return Err(Self::invalid_generator_arg(
                "save_interval_seconds must be finite and >= 0",
            ));
        }
        if num_workers == 0 {
            return Err(Self::invalid_generator_arg("num_workers must be > 0"));
        }
        if candidate_gating_enabled && candidate_eval_seeds.is_empty() {
            return Err(Self::invalid_generator_arg(
                "candidate_eval_seeds must not be empty",
            ));
        }
        if non_network_num_simulations == 0 {
            return Err(Self::invalid_generator_arg(
                "non_network_num_simulations must be > 0",
            ));
        }
        if !nn_value_weight.is_finite() || nn_value_weight < 0.0 {
            return Err(Self::invalid_generator_arg(
                "config.nn_value_weight must be finite and >= 0",
            ));
        }
        Ok(())
    }

    fn worker_settings(&self, snapshot_persister: Arc<SnapshotPersister>) -> WorkerSettings {
        WorkerSettings {
            bootstrap_model_path: self.bootstrap_model_path.clone(),
            training_data_path: self.training_data_path.clone(),
            snapshot_persister,
            config: self.config.clone(),
            max_placements: self.max_placements,
            save_interval_seconds: self.save_interval_seconds,
            num_workers: self.num_workers,
            candidate_eval_seeds: Arc::clone(&self.candidate_eval_seeds),
            non_network_num_simulations: self.non_network_num_simulations,
            save_eval_trees: self.save_eval_trees,
        }
    }

    fn worker_shared_state(&self) -> WorkerSharedState {
        WorkerSharedState {
            buffer: Arc::clone(&self.buffer),
            running: Arc::clone(&self.running),
            games_generated: Arc::clone(&self.games_generated),
            examples_generated: Arc::clone(&self.examples_generated),
            completed_games: Arc::clone(&self.completed_games),
            pending_candidate: Arc::clone(&self.pending_candidate),
            evaluating_candidate: Arc::clone(&self.evaluating_candidate),
            model_eval_events: Arc::clone(&self.model_eval_events),
            incumbent: IncumbentState {
                model_path: Arc::clone(&self.incumbent_model_path),
                uses_network: Arc::clone(&self.incumbent_uses_network),
                model_step: Arc::clone(&self.incumbent_model_step),
                model_version: Arc::clone(&self.incumbent_model_version),
                nn_value_weight: Arc::clone(&self.incumbent_nn_value_weight),
                death_penalty: Arc::clone(&self.incumbent_death_penalty),
                overhang_penalty_weight: Arc::clone(&self.incumbent_overhang_penalty_weight),
                eval_avg_attack: Arc::clone(&self.incumbent_eval_avg_attack),
            },
            live_overrides: LiveSearchOverrides {
                add_noise: Arc::clone(&self.live_add_noise),
                visit_sampling_epsilon: Arc::clone(&self.live_visit_sampling_epsilon),
            },
        }
    }
}

#[pymethods]
impl GameGenerator {
    #[new]
    #[pyo3(signature = (model_path, training_data_path, config=None, max_placements=100, add_noise=true, max_examples=100_000, save_interval_seconds=0.0, num_workers=3, initial_model_step=0, candidate_eval_seeds=None, start_with_network=true, non_network_num_simulations=4000, initial_incumbent_eval_avg_attack=0.0, candidate_gating_enabled=true, save_eval_trees=true))]
    pub fn new(
        model_path: String,
        training_data_path: String,
        config: Option<MCTSConfig>,
        max_placements: u32,
        add_noise: bool,
        max_examples: usize,
        save_interval_seconds: f64,
        num_workers: usize,
        initial_model_step: u64,
        candidate_eval_seeds: Option<Vec<u64>>,
        start_with_network: bool,
        non_network_num_simulations: u32,
        initial_incumbent_eval_avg_attack: f32,
        candidate_gating_enabled: bool,
        save_eval_trees: bool,
    ) -> PyResult<Self> {
        let candidate_eval_seeds = candidate_eval_seeds.ok_or_else(|| {
            Self::invalid_generator_arg("candidate_eval_seeds must be provided explicitly")
        })?;

        let mut resolved_config = config.unwrap_or_default();
        resolved_config.max_placements = max_placements;
        Self::validate_new_args(
            max_placements,
            max_examples,
            save_interval_seconds,
            num_workers,
            candidate_gating_enabled,
            &candidate_eval_seeds,
            non_network_num_simulations,
            resolved_config.nn_value_weight,
        )?;
        let initial_nn_value_weight_bits = resolved_config.nn_value_weight.to_bits();
        let initial_death_penalty_bits = resolved_config.death_penalty.to_bits();
        let initial_overhang_penalty_weight_bits =
            resolved_config.overhang_penalty_weight.to_bits();
        let initial_visit_sampling_epsilon_bits = resolved_config.visit_sampling_epsilon.to_bits();
        let bootstrap_model_path = PathBuf::from(model_path);

        Ok(GameGenerator {
            bootstrap_model_path: bootstrap_model_path.clone(),
            training_data_path: PathBuf::from(training_data_path),
            config: resolved_config,
            max_placements,
            live_add_noise: Arc::new(AtomicBool::new(add_noise)),
            live_visit_sampling_epsilon: Arc::new(AtomicU32::new(
                initial_visit_sampling_epsilon_bits,
            )),
            save_interval_seconds,
            num_workers,
            candidate_gating_enabled,
            snapshot_persister: None,
            candidate_eval_seeds: Arc::from(candidate_eval_seeds),
            non_network_num_simulations,
            buffer: Arc::new(SharedBuffer::new(max_examples)),
            running: Arc::new(AtomicBool::new(false)),
            games_generated: Arc::new(AtomicU64::new(0)),
            examples_generated: Arc::new(AtomicU64::new(0)),
            completed_games: Arc::new(RwLock::new(VecDeque::new())),
            pending_candidate: Arc::new(RwLock::new(None)),
            evaluating_candidate: Arc::new(RwLock::new(None)),
            model_eval_events: Arc::new(RwLock::new(VecDeque::new())),
            incumbent_model_path: Arc::new(RwLock::new(bootstrap_model_path)),
            incumbent_uses_network: Arc::new(AtomicBool::new(start_with_network)),
            incumbent_model_step: Arc::new(AtomicU64::new(initial_model_step)),
            incumbent_model_version: Arc::new(AtomicU64::new(0)),
            incumbent_nn_value_weight: Arc::new(AtomicU32::new(initial_nn_value_weight_bits)),
            incumbent_death_penalty: Arc::new(AtomicU32::new(initial_death_penalty_bits)),
            incumbent_overhang_penalty_weight: Arc::new(AtomicU32::new(
                initial_overhang_penalty_weight_bits,
            )),
            incumbent_eval_avg_attack: Arc::new(AtomicU32::new(
                initial_incumbent_eval_avg_attack.to_bits(),
            )),
            save_eval_trees,
            thread_handles: Vec::new(),
        })
    }

    /// Start background game generation.
    ///
    /// Spawns worker threads that continuously generate games and write
    /// them to training_data_path. When candidate gating is enabled, one
    /// worker is dedicated to candidate evaluation and promotion decisions.
    pub fn start(&mut self) -> PyResult<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Generator is already running",
            ));
        }

        // Create training data parent directory if needed.
        if let Some(parent_dir) = self.training_data_path.parent() {
            if !parent_dir.as_os_str().is_empty() {
                fs::create_dir_all(parent_dir).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to create training data directory {}: {}",
                        parent_dir.display(),
                        e
                    ))
                })?;
            }
        }

        // Load existing replay buffer snapshot if present.
        if self.training_data_path.exists() {
            let loaded_examples =
                read_examples_from_npz(&self.training_data_path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to load replay data from {}: {}",
                        self.training_data_path.display(),
                        e
                    ))
                })?;

            if !loaded_examples.is_empty() {
                let loaded_examples_count = loaded_examples.len();
                let loaded_max_game_number = loaded_examples
                    .iter()
                    .map(|example| example.game_number)
                    .max()
                    .expect("loaded_examples was checked as non-empty");
                self.buffer.add_compact_examples(loaded_examples);
                let retained_examples_count = self.buffer.len();
                let replay_end_index = self.buffer.window_end_index();
                self.games_generated
                    .store(loaded_max_game_number, Ordering::SeqCst);
                self.examples_generated
                    .store(replay_end_index, Ordering::SeqCst);
                eprintln!(
                    "[GameGenerator] Loaded {} replay examples from {} (max_game_number={})",
                    retained_examples_count,
                    self.training_data_path.display(),
                    loaded_max_game_number
                );
                if retained_examples_count < loaded_examples_count {
                    eprintln!(
                        "[GameGenerator] Dropped {} replay examples due to max_examples cap",
                        loaded_examples_count - retained_examples_count
                    );
                }
            }
        }

        // Set running flag
        self.running.store(true, Ordering::SeqCst);
        let snapshot_persister = SnapshotPersister::new();
        self.snapshot_persister = Some(Arc::clone(&snapshot_persister));
        let evaluator_worker_id = self
            .candidate_gating_enabled
            .then_some(self.num_workers - 1);
        let worker_settings = self.worker_settings(snapshot_persister);
        let worker_shared_state = self.worker_shared_state();

        // Spawn worker threads
        for worker_id in 0..self.num_workers {
            let worker_context = WorkerContext {
                worker_id,
                is_evaluator_worker: evaluator_worker_id == Some(worker_id),
                settings: worker_settings.clone(),
                shared: worker_shared_state.clone(),
            };

            let handle = thread::spawn(move || Self::worker_loop(worker_context));

            self.thread_handles.push(handle);
        }

        Ok(())
    }

    /// Stop background game generation.
    ///
    /// Signals the worker threads to stop and waits for them to finish.
    ///
    /// The wait loop checks for pending Python signals so repeated Ctrl+C can
    /// abort shutdown promptly instead of blocking on a long join.
    pub fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        if !self.running.load(Ordering::SeqCst) {
            if let Some(snapshot_persister) = self.snapshot_persister.take() {
                snapshot_persister.shutdown();
            }
            return Ok(());
        }

        // Signal stop
        self.running.store(false, Ordering::SeqCst);

        let mut handles = std::mem::take(&mut self.thread_handles);

        // Join finished workers, but keep checking Python signals while waiting.
        while !handles.is_empty() {
            if let Some(finished_index) = handles.iter().position(|handle| handle.is_finished()) {
                let handle = handles.swap_remove(finished_index);
                handle.join().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Worker thread panicked")
                })?;
                continue;
            }
            py.check_signals()?;
            thread::sleep(Duration::from_millis(10));
        }

        if let Some(snapshot_persister) = self.snapshot_persister.take() {
            let saved_examples = self.buffer.len();
            if saved_examples > 0
                && snapshot_persister
                    .submit_buffer_snapshot(self.buffer.as_ref(), &self.training_data_path)
            {
                eprintln!(
                    "[GameGenerator] Writing final replay snapshot ({} examples)...",
                    saved_examples
                );
                let start = Instant::now();
                snapshot_persister.shutdown();
                let elapsed = start.elapsed();
                eprintln!(
                    "[GameGenerator] Saved {} examples to disk in {:.1}s",
                    saved_examples,
                    elapsed.as_secs_f64()
                );
            } else {
                snapshot_persister.shutdown();
            }
        }

        self.cleanup_queued_candidate_artifacts();

        Ok(())
    }

    /// Check if the generator is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the number of games generated since start.
    pub fn games_generated(&self) -> u64 {
        self.games_generated.load(Ordering::SeqCst)
    }

    /// Get the number of training examples generated since start.
    pub fn examples_generated(&self) -> u64 {
        self.examples_generated.load(Ordering::SeqCst)
    }

    pub fn incumbent_model_step(&self) -> u64 {
        self.incumbent_model_step.load(Ordering::SeqCst)
    }

    pub fn incumbent_model_path(&self) -> String {
        self.incumbent_model_path
            .read()
            .unwrap()
            .to_string_lossy()
            .into_owned()
    }

    pub fn incumbent_uses_network(&self) -> bool {
        self.incumbent_uses_network.load(Ordering::SeqCst)
    }

    pub fn incumbent_eval_avg_attack(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_eval_avg_attack)
    }

    pub fn incumbent_nn_value_weight(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_nn_value_weight)
    }

    pub fn incumbent_death_penalty(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_death_penalty)
    }

    pub fn incumbent_overhang_penalty_weight(&self) -> f32 {
        Self::load_atomic_f32(&self.incumbent_overhang_penalty_weight)
    }

    /// Current live `add_noise` setting applied to self-play games.
    pub fn live_add_noise(&self) -> bool {
        self.live_add_noise.load(Ordering::SeqCst)
    }

    /// Current live `visit_sampling_epsilon` applied to self-play games.
    pub fn live_visit_sampling_epsilon(&self) -> f32 {
        Self::load_atomic_f32(&self.live_visit_sampling_epsilon)
    }

    /// Live-update self-play search overrides without rebuilding workers.
    ///
    /// `None` arguments leave the corresponding setting unchanged. Workers
    /// pick up new values at the start of the next game. Candidate-eval
    /// games still force `add_noise=false` and `visit_sampling_epsilon=0`
    /// inside the worker loop, so this only affects training self-play.
    #[pyo3(signature = (add_noise=None, visit_sampling_epsilon=None))]
    pub fn update_search_overrides(
        &self,
        add_noise: Option<bool>,
        visit_sampling_epsilon: Option<f32>,
    ) -> PyResult<()> {
        if let Some(epsilon) = visit_sampling_epsilon {
            if !epsilon.is_finite() || !(0.0..=1.0).contains(&epsilon) {
                return Err(Self::invalid_generator_arg(format!(
                    "visit_sampling_epsilon must be in [0, 1] (got {epsilon})"
                )));
            }
        }
        if let Some(value) = add_noise {
            self.live_add_noise.store(value, Ordering::SeqCst);
        }
        if let Some(epsilon) = visit_sampling_epsilon {
            Self::store_atomic_f32(&self.live_visit_sampling_epsilon, epsilon);
        }
        Ok(())
    }

    /// Return whether a candidate is pending or currently being evaluated.
    pub fn candidate_gate_busy(&self) -> bool {
        if !self.candidate_gating_enabled {
            return false;
        }
        self.pending_candidate.read().unwrap().is_some()
            || self.evaluating_candidate.read().unwrap().is_some()
    }

    /// Queue a candidate model for evaluator-worker gating.
    ///
    /// If another candidate is already pending, it is dropped in favor of this one.
    /// Returns True when the candidate is queued, False when ignored as stale.
    /// `force_promote=True` skips the avg-attack gate for this candidate.
    /// `death_penalty` and `overhang_penalty_weight` are taken verbatim from
    /// the trainer-computed schedule and applied to candidate eval and (on
    /// promotion) the incumbent search state.
    #[pyo3(signature = (model_path, model_step, nn_value_weight, death_penalty, overhang_penalty_weight, force_promote=false))]
    pub fn queue_candidate_model(
        &self,
        model_path: String,
        model_step: u64,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
        force_promote: bool,
    ) -> PyResult<bool> {
        if !self.candidate_gating_enabled {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Candidate gating is disabled; use sync_model_directly instead",
            ));
        }
        let candidate_path = PathBuf::from(model_path);
        if !nn_value_weight.is_finite() || nn_value_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "nn_value_weight must be finite and >= 0",
            ));
        }
        if !death_penalty.is_finite() || death_penalty < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "death_penalty must be finite and >= 0",
            ));
        }
        if !overhang_penalty_weight.is_finite() || overhang_penalty_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "overhang_penalty_weight must be finite and >= 0",
            ));
        }
        if !candidate_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Candidate model does not exist: {}",
                candidate_path.display()
            )));
        }

        let incumbent_step = self.incumbent_model_step.load(Ordering::SeqCst);
        let incumbent_path = self.incumbent_model_path.read().unwrap().clone();
        if model_step <= incumbent_step || candidate_path == incumbent_path {
            Self::remove_model_artifacts_if_safe(
                &candidate_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
            return Ok(false);
        }

        let request = CandidateModelRequest {
            model_path: candidate_path.clone(),
            model_step,
            nn_value_weight,
            death_penalty,
            overhang_penalty_weight,
            force_promote,
        };
        let replaced = {
            let mut pending = self.pending_candidate.write().unwrap();
            pending.replace(request)
        };
        if let Some(old_request) = replaced {
            if old_request.model_path != candidate_path {
                let evaluating_path = self
                    .evaluating_candidate
                    .read()
                    .unwrap()
                    .as_ref()
                    .map(|r| r.model_path.clone());
                Self::remove_model_artifacts_if_safe(
                    &old_request.model_path,
                    &self.bootstrap_model_path,
                    &incumbent_path,
                    evaluating_path.as_deref(),
                );
            }
        }
        Ok(true)
    }

    /// Immediately switch all workers to a freshly exported model artifact.
    ///
    /// Only available when candidate gating is disabled.
    /// Returns True when the sync updates the incumbent, False when ignored as stale.
    pub fn sync_model_directly(
        &self,
        model_path: String,
        model_step: u64,
        nn_value_weight: f32,
        death_penalty: f32,
        overhang_penalty_weight: f32,
    ) -> PyResult<bool> {
        if self.candidate_gating_enabled {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Direct model sync is unavailable while candidate gating is enabled",
            ));
        }
        let synced_model_path = PathBuf::from(model_path);
        if !nn_value_weight.is_finite() || nn_value_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "nn_value_weight must be finite and >= 0",
            ));
        }
        if !death_penalty.is_finite() || death_penalty < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "death_penalty must be finite and >= 0",
            ));
        }
        if !overhang_penalty_weight.is_finite() || overhang_penalty_weight < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "overhang_penalty_weight must be finite and >= 0",
            ));
        }
        if !synced_model_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Synced model does not exist: {}",
                synced_model_path.display()
            )));
        }

        let incumbent_step = self.incumbent_model_step.load(Ordering::SeqCst);
        let incumbent_path = self.incumbent_model_path.read().unwrap().clone();
        if model_step <= incumbent_step {
            Self::remove_model_artifacts_if_safe(
                &synced_model_path,
                &self.bootstrap_model_path,
                &incumbent_path,
                None,
            );
            return Ok(false);
        }

        {
            let mut incumbent_model_path = self.incumbent_model_path.write().unwrap();
            *incumbent_model_path = synced_model_path.clone();
        }
        self.incumbent_uses_network.store(true, Ordering::SeqCst);
        self.incumbent_model_step
            .store(model_step, Ordering::SeqCst);
        self.incumbent_model_version.fetch_add(1, Ordering::SeqCst);
        Self::store_atomic_f32(&self.incumbent_nn_value_weight, nn_value_weight);
        Self::store_atomic_f32(&self.incumbent_death_penalty, death_penalty);
        Self::store_atomic_f32(
            &self.incumbent_overhang_penalty_weight,
            overhang_penalty_weight,
        );
        Self::store_atomic_f32(&self.incumbent_eval_avg_attack, 0.0);

        Self::remove_model_artifacts_if_safe(
            &incumbent_path,
            &self.bootstrap_model_path,
            &synced_model_path,
            None,
        );
        Ok(true)
    }

    /// Drain all completed game stats in generation order.
    pub fn drain_completed_game_stats(&self) -> Vec<(u64, HashMap<String, f32>)> {
        let mut queue = self.completed_games.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(info) = queue.pop_front() {
            drained.push((info.game_number, info.to_dict()));
        }
        drained
    }

    /// Drain completed games with optional replay payloads in generation order.
    pub fn drain_completed_games(&self, py: Python<'_>) -> Vec<HashMap<String, PyObject>> {
        let mut queue = self.completed_games.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(info) = queue.pop_front() {
            let mut d: HashMap<String, PyObject> = HashMap::new();
            d.insert("game_number".into(), info.game_number.into_py(py));
            d.insert("stats".into(), info.to_dict().into_py(py));
            d.insert("completed_time_s".into(), info.completed_time_s.into_py(py));
            d.insert("replay".into(), info.replay.into_py(py));
            drained.push(d);
        }
        drained
    }

    /// Drain evaluator decision events in generation order.
    pub fn drain_model_eval_events(&self, py: Python<'_>) -> Vec<HashMap<String, PyObject>> {
        let mut queue = self.model_eval_events.write().unwrap();
        let mut drained = Vec::with_capacity(queue.len());
        while let Some(event) = queue.pop_front() {
            let mut d: HashMap<String, PyObject> = HashMap::new();
            d.insert(
                "incumbent_step".into(),
                (event.incumbent_step as f64).into_py(py),
            );
            d.insert(
                "incumbent_uses_network".into(),
                (if event.incumbent_uses_network {
                    1.0
                } else {
                    0.0_f64
                })
                .into_py(py),
            );
            d.insert(
                "incumbent_avg_attack".into(),
                (event.incumbent_avg_attack as f64).into_py(py),
            );
            d.insert(
                "incumbent_nn_value_weight".into(),
                (event.incumbent_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "candidate_step".into(),
                (event.candidate_step as f64).into_py(py),
            );
            d.insert(
                "candidate_games".into(),
                (event.candidate_games as f64).into_py(py),
            );
            d.insert(
                "candidate_avg_attack".into(),
                (event.candidate_avg_attack as f64).into_py(py),
            );
            d.insert(
                "candidate_attack_variance".into(),
                (event.candidate_attack_variance as f64).into_py(py),
            );
            d.insert(
                "candidate_nn_value_weight".into(),
                (event.candidate_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "promoted_nn_value_weight".into(),
                (event.promoted_nn_value_weight as f64).into_py(py),
            );
            d.insert(
                "promoted_death_penalty".into(),
                (event.promoted_death_penalty as f64).into_py(py),
            );
            d.insert(
                "promoted_overhang_penalty_weight".into(),
                (event.promoted_overhang_penalty_weight as f64).into_py(py),
            );
            d.insert(
                "promoted".into(),
                (if event.promoted { 1.0 } else { 0.0_f64 }).into_py(py),
            );
            d.insert(
                "auto_promoted".into(),
                (if event.auto_promoted { 1.0 } else { 0.0_f64 }).into_py(py),
            );
            d.insert(
                "force_promoted".into(),
                (if event.force_promoted { 1.0 } else { 0.0_f64 }).into_py(py),
            );
            d.insert(
                "evaluation_seconds".into(),
                (event.evaluation_seconds as f64).into_py(py),
            );
            d.insert(
                "best_game_replay".into(),
                event.best_game_replay.into_py(py),
            );
            d.insert(
                "worst_game_replay".into(),
                event.worst_game_replay.into_py(py),
            );
            d.insert(
                "worst_game_tree_path".into(),
                event.worst_game_tree_path.into_py(py),
            );
            d.insert(
                "per_game_results".into(),
                event.per_game_results.into_py(py),
            );
            d.insert(
                "per_game_prediction_metrics".into(),
                event.per_game_prediction_metrics.into_py(py),
            );
            drained.push(d);
        }
        drained
    }

    /// Get the current number of examples in the replay buffer.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }

    /// Fetch a delta slice from the replay buffer's logical index space.
    ///
    /// Returns:
    /// - `window_start_index`: global index of current FIFO window start
    /// - `window_end_index`: global index one-past current FIFO window end
    /// - `slice_start_index`: global index of first returned example
    /// - compact arrays (see `examples_to_numpy`) for up to `max_examples` examples in `[slice_start_index, window_end_index)`
    ///
    /// Returns None if the buffer is empty.
    #[pyo3(signature = (from_index, max_examples))]
    pub fn replay_buffer_delta<'py>(
        &self,
        py: Python<'py>,
        from_index: u64,
        max_examples: usize,
    ) -> PyResult<Option<(u64, u64, u64, CompactNumpyBatch<'py>)>> {
        if max_examples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_examples must be > 0",
            ));
        }

        let Some((window_start, window_end, slice_start, slice_examples)) =
            self.buffer.logical_delta_slice(from_index, max_examples)
        else {
            return Ok(None);
        };

        let arrays = Self::examples_to_numpy(py, &slice_examples)?;
        Ok(Some((window_start, window_end, slice_start, arrays)))
    }

    /// Write a delta slice from the replay buffer directly to an NPZ file.
    ///
    /// Used by remote generator workers to upload incremental replay chunks
    /// without first marshaling Vec<f32> tensors into Python.
    ///
    /// Returns None when the buffer is empty. When the requested slice is
    /// non-empty, writes an NPZ file in the same format as the periodic
    /// snapshot and returns `(window_start, window_end, slice_start, count)`.
    /// When `from_index >= window_end` the slice is empty and no file is
    /// written; the tuple is returned with `count = 0` so the caller can
    /// advance its cursor.
    #[pyo3(signature = (filepath, from_index, max_examples))]
    pub fn dump_replay_delta_to_npz(
        &self,
        filepath: String,
        from_index: u64,
        max_examples: usize,
    ) -> PyResult<Option<(u64, u64, u64, usize)>> {
        if max_examples == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_examples must be > 0",
            ));
        }
        let Some((window_start, window_end, slice_start, slice_examples)) =
            self.buffer.logical_delta_slice(from_index, max_examples)
        else {
            return Ok(None);
        };
        let count = slice_examples.len();
        if count == 0 {
            return Ok(Some((window_start, window_end, slice_start, 0)));
        }
        write_examples_to_npz(Path::new(&filepath), &slice_examples).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to write replay chunk to {}: {}",
                filepath, e
            ))
        })?;
        Ok(Some((window_start, window_end, slice_start, count)))
    }

    /// Read training examples from an NPZ file and append them to the replay buffer.
    ///
    /// Used by trainer-side ingestion of remote replay chunks fetched from R2.
    /// `game_number_offset` is added to each example's `game_number` before
    /// insertion so cross-machine W&B per-game logging stays unique. Local
    /// (non-remote) ingest paths pass 0 (the default).
    /// Returns the number of examples ingested.
    #[pyo3(signature = (filepath, game_number_offset = 0))]
    pub fn ingest_examples_from_npz(
        &self,
        filepath: String,
        game_number_offset: u64,
    ) -> PyResult<usize> {
        let mut examples = read_examples_from_npz(Path::new(&filepath)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to read replay chunk from {}: {}",
                filepath, e
            ))
        })?;
        let count = examples.len();
        if count == 0 {
            return Ok(0);
        }
        if game_number_offset != 0 {
            for example in examples.iter_mut() {
                example.game_number = example.game_number.saturating_add(game_number_offset);
            }
        }
        self.buffer.add_compact_examples(examples);
        self.examples_generated
            .fetch_add(count as u64, Ordering::SeqCst);
        Ok(count)
    }

    /// Sample a uniform batch from the replay buffer as compact arrays (see
    /// `examples_to_numpy`). Returns None if the buffer is empty.
    #[pyo3(signature = (batch_size))]
    pub fn sample_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<Option<CompactNumpyBatch<'py>>> {
        // Snapshot sampled examples under lock, then release lock before heavy work.
        let sampled_examples: Vec<CompactExample> = {
            let state = self.buffer.state.read().unwrap();
            let n = state.examples.len();
            if n == 0 {
                return Ok(None);
            }

            let actual_batch = batch_size.min(n);
            let mut rng = thread_rng();
            (0..actual_batch)
                .map(|_| {
                    let idx = rng.gen_range(0..n);
                    state.examples[idx].clone()
                })
                .collect()
        };
        Self::examples_to_numpy(py, &sampled_examples).map(Some)
    }
}
