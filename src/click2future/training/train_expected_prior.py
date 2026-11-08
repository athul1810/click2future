"""D019: paired direct refinements with original or expected loss, on TC2."""

import argparse
import copy
import json
import math
import os
import signal
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from click2future.models.object_motion import MOTION_CONTRACT
from click2future.models.prior_motion import PRIOR_CONTRACT, PriorTrajectoryModel
from click2future.models.rest_anchored_prior import REST_ANCHOR_CONTRACT, RestAnchoredPriorModel
from click2future.models.world_motion import WORLD_CONTRACT, BoundWorld

from .direct_objective import DIRECT_OBJECTIVE, expected_coefficients, expected_direct_objective
from .train_cvae import require_tc2, sha256
from .train_prior import PriorConfig, PriorTrainer, prior_config, prior_sources
from .world_data import load_cache


@dataclass(frozen=True)
class ExpectedPriorConfig(PriorConfig):
    expected_objective: bool = True
    rest_anchored: bool = False

    def validate(self):
        super().validate()
        if not isinstance(self.expected_objective, bool) or not self.world_prior_only:
            raise ValueError("The expected-loss comparison requires two direct-decoder arms")
        if self.risk_penetration_weight != 0:
            raise ValueError("Nonlinear tail risk is not part of the expected-loss comparison")
        if not isinstance(self.rest_anchored, bool):
            raise ValueError("rest_anchored must be boolean")


def sources():
    return {
        **prior_sources(),
        "models/rest_anchored_prior.py": sha256(
            Path(__file__).parents[1] / "models/rest_anchored_prior.py"
        ),
        **{
            "training/" + name: sha256(Path(__file__).with_name(name))
            for name in ("direct_objective.py", "train_expected_prior.py")
        },
    }


class ExpectedPriorTrainer(PriorTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.rest_anchored:
            with torch.random.fork_rng(devices=[]):
                anchored = RestAnchoredPriorModel(**prior_config(self.config)).to(self.device)
            anchored.load_state_dict(self.model.state_dict(), strict=True)
            self.model = anchored
            self.ema = copy.deepcopy(anchored).eval().requires_grad_(False)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01
            )

    def train_step(self):
        if not self.config.expected_objective:
            return super().train_step()
        c, row = self.config, self.data["train"]
        indices = torch.multinomial(
            self.weights, c.batch_size, replacement=True, generator=self.generator
        )
        clean, features, force = (row[key][indices] for key in ("motion", "features", "force"))
        # Consume precisely the original RNG draws so the paired arms retain
        # identical future batches. These nuisance variables do not enter loss.
        torch.randn(c.batch_size, device=self.device, generator=self.generator)
        torch.randn(clean.shape, device=self.device, generator=self.generator)
        loss, parts = expected_direct_objective(
            BoundWorld(self.model, self.worlds["train"][indices]),
            clean,
            features,
            force,
            {key: value[indices] for key, value in self.geometry["train"].items()},
            c,
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite expected direct loss")
        rate = c.learning_rate * (0.03 + 0.97 * (1 + math.cos(math.pi * self.step / c.steps)) / 2)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(
                self.ema.parameters(), self.model.parameters(), strict=True
            ):
                averaged.lerp_(current, 1 - c.ema_decay)
        self.step += 1
        return {
            "step": self.step,
            "loss": float(loss.detach()),
            "lr": rate,
            "gradient_norm": float(norm),
            **{key: float(value) for key, value in parts.items()},
        }

    def save(self, path, provenance):
        state = {
            "trajectory_format_version": 8,
            "training_variant": "expected_direct_objective",
            "motion_contract": MOTION_CONTRACT,
            "world_contract": WORLD_CONTRACT,
            "prior_contract": PRIOR_CONTRACT,
            "objective_contract": DIRECT_OBJECTIVE,
            "expected_coefficients": expected_coefficients(self.config),
            "rest_anchor_contract": REST_ANCHOR_CONTRACT if self.config.rest_anchored else None,
            "config": asdict(self.config),
            "model_config": prior_config(self.config),
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "provenance": provenance,
        }
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def resume(self, path, provenance):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if (
            state["trajectory_format_version"] != 8
            or state["training_variant"] != "expected_direct_objective"
            or state["objective_contract"] != DIRECT_OBJECTIVE
            or state["expected_coefficients"] != expected_coefficients(self.config)
            or (
                self.config.rest_anchored
                and state.get("rest_anchor_contract") != REST_ANCHOR_CONTRACT
            )
            or state["motion_contract"] != MOTION_CONTRACT
            or state["world_contract"] != WORLD_CONTRACT
            or state["prior_contract"] != PRIOR_CONTRACT
            or state["model_config"] != prior_config(self.config)
            or state["config"] != asdict(self.config)
            or state["provenance"] != provenance
            or isinstance(state["step"], bool)
            or not isinstance(state["step"], int)
            or not 0 <= state["step"] <= self.config.steps
        ):
            raise ValueError("Resume requires identical objective, data, config and source")
        for name in ("model", "ema", "optimizer"):
            getattr(self, name).load_state_dict(state[name])
        self.generator.set_state(state["generator"].cpu())
        self.step = state["step"]
        return state


def load_expected_prior(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ExpectedPriorConfig(**state["config"])
    config.validate()
    if (
        state["trajectory_format_version"] != 8
        or state["training_variant"] != "expected_direct_objective"
        or state["objective_contract"] != DIRECT_OBJECTIVE
        or state["expected_coefficients"] != expected_coefficients(config)
        or (config.rest_anchored and state.get("rest_anchor_contract") != REST_ANCHOR_CONTRACT)
        or state["motion_contract"] != MOTION_CONTRACT
        or state["world_contract"] != WORLD_CONTRACT
        or state["prior_contract"] != PRIOR_CONTRACT
        or state["model_config"] != prior_config(config)
        or isinstance(state["step"], bool)
        or not isinstance(state["step"], int)
        or state["step"] < 1
    ):
        raise ValueError("Unsupported or untrained expected-objective checkpoint")
    current = sources()
    for name in ("models/object_motion.py", "models/world_motion.py", "models/prior_motion.py"):
        if state["provenance"]["source_sha256"].get(name) != current[name]:
            raise ValueError("Expected-objective model source differs: " + name)
    if (
        config.rest_anchored
        and state["provenance"]["source_sha256"].get("models/rest_anchored_prior.py")
        != current["models/rest_anchored_prior.py"]
    ):
        raise ValueError("Rest-anchored model source differs")
    with torch.random.fork_rng(devices=[]):
        factory = RestAnchoredPriorModel if config.rest_anchored else PriorTrajectoryModel
        model = factory(**state["model_config"])
    model.load_state_dict(state["ema"], strict=True)
    model.to(device).eval()
    model.force_limit, model.sigma_min, model.sigma_max = (
        config.force_limit,
        config.sigma_min,
        config.sigma_max,
    )
    if sha256(checkpoint) != digest:
        raise ValueError("Checkpoint changed while loading")
    return model, state


def resume_probe(data, geometry, worlds, config, path, *, device):
    """Check the actual device before relying on interrupted-run continuation."""
    tiny = replace(config, steps=4, width=16, layers=1, batch_size=2)
    probe = ExpectedPriorTrainer(data, geometry, worlds, tiny, device=device)
    identity = {"scope": "two-update device fixture; not model-quality evidence"}
    probe.train_step()
    probe.save(path, identity)
    expected = probe.train_step()
    restored = ExpectedPriorTrainer(data, geometry, worlds, tiny, device=device)
    restored.resume(path, identity)
    actual = restored.train_step()
    differences = {
        kind: max(
            float((value - getattr(restored, kind).state_dict()[name]).abs().max())
            for name, value in getattr(probe, kind).state_dict().items()
        )
        for kind in ("model", "ema")
    }
    if expected != actual or any(differences.values()):
        raise ValueError("Expected-prior device checkpoint continuation differs")
    return {"device": device, "losses_equal": True, "max_abs_difference": differences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "warm-start", "config", "run-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    require_tc2()
    revision = os.environ["C2F_SOURCE_REVISION"]
    if not revision or revision == "unrecorded":
        parser.error("Require immutable committed source")
    config = ExpectedPriorConfig(**json.loads(args.config.read_text()))
    config.validate()
    if args.run_dir.exists() and not args.resume:
        parser.error("Preserve earlier attempts; use --resume for this exact run")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    torch.use_deterministic_algorithms(True)
    data, geometry, worlds, provenance = load_cache(args.cache)
    provenance = {**provenance, "source_sha256": sources(), "source_revision": revision}
    check = resume_probe(
        data, geometry, worlds, config, args.run_dir / "resume-fixture.pt", device="cuda"
    )
    (args.run_dir / "resume-check.json").write_text(json.dumps(check, indent=2) + "\n")
    trainer = ExpectedPriorTrainer(data, geometry, worlds, config, device="cuda")
    checkpoint = args.run_dir / "latest.pt"
    if args.resume:
        old = torch.load(checkpoint, map_location="cpu", weights_only=True)
        provenance["initialization"] = old["provenance"]["initialization"]
        if provenance["initialization"]["checkpoint_sha256"] != sha256(args.warm_start):
            raise ValueError("Resume teacher differs")
        trainer.resume(checkpoint, provenance)
    else:
        provenance["initialization"] = trainer.initialize(args.warm_start, provenance)
        # A durable initial checkpoint also makes a very early interruption resumable.
        trainer.save(checkpoint, provenance)
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, stop)
    started, previous_step = time.perf_counter(), trainer.step
    record = {
        "config": asdict(config),
        "provenance": provenance,
        "expected_coefficients": expected_coefficients(config),
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "resumed_from_step": previous_step,
        "selection": "fixed final update; validation is diagnostic only",
        "test_splits_evaluated": [],
    }
    torch.cuda.reset_peak_memory_stats()
    try:
        with (args.run_dir / "training.jsonl").open("a", buffering=1) as log:
            while trainer.step < config.steps and not stopping:
                row = trainer.train_step()
                if trainer.step % 50 == 0:
                    log.write(json.dumps(row) + "\n")
                if trainer.step % config.checkpoint_every == 0:
                    trainer.save(checkpoint, provenance)
                if time.perf_counter() - started >= config.max_seconds:
                    stopping = True
        trainer.save(checkpoint, provenance)
        complete = trainer.step == config.steps
        if complete:
            trainer.save(args.run_dir / "final.pt", provenance)
        record["status"] = "completed" if complete else "checkpointed_interruption"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        # Preserve the last atomically written checkpoint; never replace it
        # with state from a possibly partially applied failed optimizer update.
        raise
    finally:
        record.update(
            steps_completed=trainer.step,
            elapsed_seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        temporary = args.run_dir / "summary.tmp"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.run_dir / "summary.json")
    print(json.dumps({"status": record["status"], "steps": trainer.step}), flush=True)
    return 0 if complete else 3


if __name__ == "__main__":
    raise SystemExit(main())
