from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import numpy as np
import torch

from scripts.baselines import BASELINE_RUNNERS
from scripts.checkpoints import TTA_PREREQ_BASELINES, load_model, resolve_or_build_prereq_checkpoint
from scripts.cli import apply_cli_overrides, build_parser
from scripts.config import ExperimentConfig, load_config
from scripts.data.dataset import ClientData
from scripts.data.loader import load_csv_as_clients, load_parquet_as_clients
from scripts.tta.adapter import AffineAdapter
from scripts.tta.loop import WINDOW_TRACE_FIELDS, run_local_tta_loop
from scripts.utils.efficiency import RunEfficiency, count_parameters
from scripts.utils.run_logging import tee_run_output
from scripts.utils.tools import make_run_dir, save_results, seed_everything


def _load_clients(config: ExperimentConfig) -> list[ClientData]:
    if config.data_format == "csv":
        return load_csv_as_clients(
            csv_path=config.data_path,
            timestamp_col=config.timestamp_col,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    if config.data_format == "parquet":
        clients_dir = config.parquet_clients_dir or str(Path(config.data_path).parent / "clients")
        return load_parquet_as_clients(
            manifest_path=config.data_path,
            clients_dir=clients_dir,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            max_clients=config.max_clients,
        )
    raise ValueError(f"Unknown data_format: {config.data_format!r}")


def _aggregate(per_client: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = ("mse", "mae", "smape")
    return {
        key: float(np.mean([m[key] for m in per_client.values() if not np.isnan(m[key])]))
        for key in keys
    }


def _build_random_gate_masks(
    reference_run: Path,
    clients: list[ClientData],
    n_windows: int,
    seed: int,
) -> list[np.ndarray]:
    """Rate-matched random gate (KCI Phase 2).

    For each client, read how many windows DELTA's drift gate let through in the
    reference run (`total_steps - drift_skips` from per_client_diagnostics) and
    select exactly that many windows uniformly at random, without replacement,
    from the same eligible window set. Every other component is untouched, so
    window selection is the only difference from DELTA.
    """
    import json

    metrics_path = reference_run / "metrics.json"
    if not metrics_path.exists():
        candidates = sorted(p for p in reference_run.rglob("metrics.json"))
        if not candidates:
            raise FileNotFoundError(f"No metrics.json under {reference_run}")
        metrics_path = candidates[-1]
    diagnostics = json.loads(metrics_path.read_text())["per_client_diagnostics"]

    rng = np.random.default_rng(seed)
    masks: list[np.ndarray] = []
    for client in clients:
        diag = diagnostics[client.client_id]
        total = int(round(diag["total_steps"]))
        if total != n_windows:
            raise ValueError(
                f"client {client.client_id}: reference run has {total} windows, "
                f"this run has {n_windows} — window sets must match"
            )
        n_pass = total - int(round(diag["drift_skips"]))
        mask = np.zeros(n_windows, dtype=bool)
        mask[rng.choice(n_windows, size=n_pass, replace=False)] = True
        masks.append(mask)

    passed = sum(int(m.sum()) for m in masks)
    print(
        f"[Random gate] reference={metrics_path.parent}  seed={seed}  "
        f"gate-pass {passed:,} / {n_windows * len(clients):,} windows "
        f"({passed / (n_windows * len(clients)):.6f})"
    )
    return masks


def _write_window_trace(path: Path, dataset: str, records: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset",) + WINDOW_TRACE_FIELDS)
        for record in records:
            writer.writerow((dataset,) + record)
    print(f"[Affine Local TTA] Window trace: {len(records):,} rows -> {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Affine-Adapter local TTA runner"
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--dump-windows",
        type=Path,
        default=None,
        help=(
            "Write per-window drift-gate diagnostics to this CSV. Opt-in: when the "
            "flag is absent the TTA loop runs exactly as before."
        ),
    )
    parser.add_argument(
        "--gate-mode",
        choices=("drift", "random"),
        default="drift",
        help="'random' replaces the drift gate with a rate-matched random selection.",
    )
    parser.add_argument(
        "--gate-reference",
        type=Path,
        default=None,
        help="Run dir of the DELTA run whose per-client gate-pass counts are matched.",
    )
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Phase 6 overhead measurement: synchronize CUDA around every window so "
            "per-window latency is real, and emit median/p95/reserved-VRAM/peak-RSS. "
            "Forces --max-workers 1 (overlapping client threads make per-window "
            "timings meaningless) and should be run with the machine otherwise idle."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args, parser)
    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(config.output_dir, config.dataset, f"{config.baseline}_affine")

    with tee_run_output(run_dir) as log_path:
        print(f"=== Affine Local TTA | baseline={config.baseline} | dataset={config.dataset} ===")
        print(f"  device={device}  seed={config.seed}  max_workers={args.max_workers}")
        print(f"  run_dir={run_dir}")
        print(f"  run_log={log_path}")

        clients = _load_clients(config)
        print(f"  Loaded {len(clients)} clients.")

        if config.baseline not in TTA_PREREQ_BASELINES:
            raise ValueError(
                f"Affine local TTA expects one of {sorted(TTA_PREREQ_BASELINES)}; "
                f"got {config.baseline!r}."
            )
        if config.checkpoint_path:
            print(f"  Using explicit checkpoint_path={config.checkpoint_path}")
        else:
            config = resolve_or_build_prereq_checkpoint(
                config=config,
                config_path=args.config,
                clients=clients,
                device=device,
                auto_prereq=args.auto_prereq,
                baseline_runners=BASELINE_RUNNERS,
            )

        global_model = load_model(config, device)
        adapter = AffineAdapter(
            global_model.revin.num_features,
            pred_len=config.model.pred_len,
            mode=config.tta.adapter_mode,
        )
        efficiency = RunEfficiency(device, sync_timing=args.timing)
        efficiency.start()
        window_trace: list | None = [] if args.dump_windows else None
        gate_masks = None
        if args.gate_mode == "random":
            if args.gate_reference is None:
                parser.error("--gate-mode random requires --gate-reference")
            test_s, test_e = clients[0].split_indices["test"]
            n_windows = test_e - (test_s + config.model.seq_len + config.k() - 1)
            gate_masks = _build_random_gate_masks(
                args.gate_reference, clients, n_windows, args.gate_seed
            )
        per_client, per_client_diagnostics, diagnostic_summary = run_local_tta_loop(
            global_model=global_model,
            clients=clients,
            seq_len=config.model.seq_len,
            pred_len=config.model.pred_len,
            k=config.k(),
            tta_config=config.tta,
            device=device,
            max_workers=1 if args.timing else max(1, args.max_workers),
            efficiency=efficiency,
            window_trace=window_trace,
            gate_masks=gate_masks,
            sync_timing=args.timing,
        )
        if window_trace is not None:
            _write_window_trace(args.dump_windows, config.dataset, window_trace)
        avg = _aggregate(per_client)
        print(
            f"\n[Affine Local TTA] Avg: MSE={avg['mse']:.4f}  "
            f"MAE={avg['mae']:.4f}  sMAPE={avg['smape']:.2f}%"
        )
        print(
            "[Affine Local TTA] Diagnostics: "
            f"adapt_rate={diagnostic_summary['adapt_rate']:.3f}  "
            f"accept_gate={diagnostic_summary['accept_gate_skip_rate']:.3f}  "
            f"hard_gate={diagnostic_summary['hard_gate_skip_rate']:.3f}  "
            f"drift_skip={diagnostic_summary['drift_skip_rate']:.3f}  "
            f"rollback_skip={diagnostic_summary['rollback_skip_rate']:.3f}  "
            f"reset/adapt={diagnostic_summary['reset_rate_given_adapt']:.3f}  "
            f"mean|gamma-1|={diagnostic_summary['mean_gamma_l1']:.6f}  "
            f"mean|delta|={diagnostic_summary['mean_delta_l1']:.6f}"
        )
        payload = {
            "engine": "affine_local_tta",
            "per_client": per_client,
            "per_client_diagnostics": per_client_diagnostics,
            "avg": avg,
            "diagnostic_summary": diagnostic_summary,
            "max_workers": 1 if args.timing else max(1, args.max_workers),
            "timing_mode": bool(args.timing),
            "gate_mode": args.gate_mode,
            "gate_seed": args.gate_seed if args.gate_mode == "random" else None,
            "gate_reference": str(args.gate_reference) if args.gate_reference else None,
            **efficiency.payload(
                seed=config.seed,
                trainable_params=count_parameters(adapter),
                total_backbone_params=count_parameters(global_model),
            ),
        }
        save_results(run_dir, payload, dataclasses.asdict(config))


if __name__ == "__main__":
    main()
