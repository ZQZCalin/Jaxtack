# Jaxtack System Design

## 1) Vision

Jaxtack is a modular JAX training framework with:

- strong defaults so users can run quickly via Hydra overrides;
- typed extensibility so users can add custom components with a paired Pydantic config and implementation;
- clear separation between pure training math and runtime side effects.

Primary goals:

- easy-to-use default training pipeline;
- low-friction custom component authoring;
- predictable runtime behavior;
- minimal coupling across components.

---

## 2) Design Principles

- **Config-first composition**: all runs are composed through Hydra config groups.
- **Typed component boundaries**: each pluggable family has a `Base...Config` contract.
- **Registry-driven extensibility**: components are selected by `name` and built through registries.
- **Compute-runtime split**:
  - compute path: model, loss, optimizer, train step;
  - runtime path: logging, metric aggregation, checkpointing, eval, and stop policy.
- **Strict-by-default validation** with opt-out relaxed mode.
- **No observability state in optimizer/train pytrees** unless explicitly required.

---

## 3) High-Level Architecture

```text
Hydra Config
   -> initialize()
      -> build model
      -> build dataset/dataloader
      -> build schedule
      -> build optimizer
      -> build metric_logger
      -> build metric aggregator
      -> build early stopper
      -> build trainer loop
      -> assemble Runtime container
   -> trainer_loop(runtime, train_state)
```

Runtime components handle side effects. Device-side train state remains lean and JAX-friendly.

---

## 4) Component Families

Each pluggable component family follows the same structure:

- `config.py`: base and concrete Pydantic configs
- `registry.py`: name -> config type + builder mapping
- `builder.py`: Hydra `DictConfig` -> typed config -> built instance
- `base/`: built-in implementations
- `conf/<family>/.../*.yaml`: runnable examples

Families:

- model
- dataset / dataloader
- optimizer
- schedule
- trainer
- metric_logger
- early stopper
- metric aggregator

---

## 5) Public API Shape

### 5.1 User-facing run config

The main run config composes:

- `model`
- `dataset`
- `optimizer` (with nested `schedule`)
- `trainer`
- `metric_logger`
- `early_stopper`
- `metric_aggregator`
- `runtime` (seed, precision, checkpoint, wandb, strictness, etc.)

Typical usage:

```bash
python main.py model=... dataset=... optimizer=... trainer=...
```

### 5.2 Plugin author contract

To add a custom component:

1. define config class inheriting family base config and include `name`;
2. implement builder/function/class with required signature;
3. register with family registry;
4. provide example yaml under `conf/...`.

---

## 6) Trainer API Decision

Jaxtack v1 uses a function-based trainer API.

- `build_trainer(config) -> train_loop_fn`
- `train_loop_fn(runtime, train_state) -> train_state`

Why:

- simpler than class hierarchy for current scope;
- still fully customizable as a component;
- easy to transition to class wrappers later if needed.

---

## 7) Runtime Container

A single `Runtime` object is passed into the trainer loop. It contains:

- built compute components (model, optimizer, dataloader, schedule, loss);
- runtime services (metric_logger, metric aggregator, early stopper, checkpoint manager, optional eval manager);
- global run context (config, seeds, precision, device info).

This keeps trainer signatures stable and avoids argument explosion.

---

## 8) Metrics, Logging, Early Stopping

### 8.1 Separation decision

Keep **metric aggregator**, **metric_logger**, and **early stopper** as separate components.

Rationale:

- clean separation of concerns;
- easier unit testing and replacement;
- easier user customization;
- avoids coupling stop decisions to logging backends.

### 8.2 Data flow

1. train step produces raw metrics (`loss`, `accuracy`, throughput, etc.);
2. internal components may emit additional metrics to a host sink;
3. metric aggregator updates reduced/rolling metrics;
4. metric_logger emits selected metrics;
5. early stopper queries aggregated metrics and returns stop decision.

---

## 9) Internal Metrics Without State Pollution

Requirement: optimizers/models/schedules can emit internal metrics without storing log objects in optimizer state.

Chosen pattern:

- host-side `MetricSink` for per-step metric emission;
- `MetricAggregator` consumes sink events;
- no logging buffers in `opt_state` or model params.

For jitted internals, host callbacks are allowed behind explicit config flags to avoid unnecessary performance cost.

Future multi-GPU compatibility:

- the side channel is a global runtime service that receives events from all parallel workers;
- each event includes source metadata (for example process id, device id, replica id, and step);
- aggregator reduction policy is explicit (`local_only`, `cross_replica`, `cross_process`) so behavior is predictable as distributed training is added.

---

## 10) DataLoader Contract

- trainer assumes dataloader is an **iterable**;
- random shuffling/sampling is handled during dataloader initialization;
- random-access slicing is not a framework-level requirement.

---

## 11) Config Strictness

Default: strict mode enabled.

Strict mode behavior:

- reject unknown config fields;
- reject missing required fields;
- reject unknown registry names.

Relaxed mode is available via config flag for experimentation.

---

## 12) Checkpoint Policy

Backward compatibility with previous framework checkpoints is not required.

Checkpoint v1 should include:

- model state;
- optimizer state;
- training counters and random keys.

Runtime service state (metric_logger/aggregator internals) remains host-side and is excluded by default, with optional small-state serialization later if needed.

---

## 13) Optional Callback Layer (v1.1)

Callbacks are not required for v1, but planned as a non-breaking extension.

Potential hook points:

- `on_train_start`
- `on_step_start`
- `on_step_end`
- `on_eval_end`
- `on_train_end`
- `on_exception`

Why callbacks help:

- checkpoint/eval/profiling logic becomes plug-and-play;
- trainer loop remains focused on training computation;
- feature growth does not force trainer monolith growth.

---

## 14) Target Project Layout

```text
Jaxtack/
  main.py
  src/
    initialize.py
    train.py
    runtime.py
  common/
    types.py
    registry_utils.py
  _model/
    config.py
    registry.py
    builder.py
    base/
  _dataset/
    config.py
    registry.py
    builder.py
    base/
  _optimizer/
    config.py
    registry.py
    builder.py
    base/
    schedule/
      config.py
      registry.py
      builder.py
      base/
  _trainer/
    config.py
    registry.py
    builder.py
    base/
  _metric_logger/
    config.py
    registry.py
    builder.py
    base/
  _metric/
    config.py
    registry.py
    builder.py
    base/
  _early_stopper/
    config.py
    registry.py
    builder.py
    base/
  conf/
    config.yaml
    model/
    dataset/
    optimizer/
    schedule/
    trainer/
    metric_logger/
    metric_aggregator/
    early_stopper/
```

---

## 15) Non-Goals for v1

- old checkpoint compatibility layer;
- advanced external plugin auto-discovery;
- full callback/event bus from day one;
- equal-depth support for all model frameworks on first release.

---

## 16) Acceptance Criteria

The design is considered realized when:

- one full vertical slice trains end-to-end through registries;
- adding a new optimizer does not require trainer/runtime core edits;
- early stopper can query aggregated metrics without touching device train state;
- strict mode catches config issues before training starts.

