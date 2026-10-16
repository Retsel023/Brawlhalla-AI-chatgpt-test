"""Keep duel and team PPO progress separate while transferring learned weights.

Model files use Python serialization: load only checkpoints from trusted sources.
Metadata binds each checkpoint to the observation/action contract and its digest.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from controls import ACTIONS
from game_state import FEATURE_NAMES as LEGACY_FEATURE_NAMES
from team_state import FEATURE_NAMES as DEFAULT_FEATURE_NAMES


OBSERVATION_SCHEMA = "team-v2"
MATCH_FORMATS = ("duel", "teams")


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


ACTION_NAMES_SHA256 = hashlib.sha256(_json_bytes(ACTIONS)).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_size(env):
    from gymnasium import spaces
    space = env.observation_space
    action = env.action_space
    if (not isinstance(space, spaces.Box) or len(space.shape) != 1
            or space.dtype.kind != "f"):
        raise ValueError("Session policies require flat floating-point team-state observations, not CNN pixels.")
    size = space.shape[0]
    if size < len(DEFAULT_FEATURE_NAMES) or size % len(DEFAULT_FEATURE_NAMES):
        raise ValueError("Session observations must contain 26 features per history frame.")
    if (not isinstance(action, spaces.Discrete) or action.n != len(ACTIONS)
            or action.start != 0):
        raise ValueError("Checkpoint action space must match the existing Brawlhalla action ordering.")
    return size


def _contract(contract, frames):
    result = {
        "observation_schema": OBSERVATION_SCHEMA,
        "feature_names": list(DEFAULT_FEATURE_NAMES),
        "history_frames": frames,
        "history_order": "oldest_to_newest",
        "position_scale": 1000.0,
        "damage_scale": 100.0,
        "damage_reward_scale": 0.01,
        "slot_semantics": ["self", "primary_opponent", "teammate", "secondary_opponent"],
        "action_names": [list(action) for action in ACTIONS],
    }
    if contract is not None:
        if not isinstance(contract, dict):
            raise ValueError("Observation contract must be a JSON object.")
        result.update(deepcopy(contract))
    if result["observation_schema"] != OBSERVATION_SCHEMA:
        raise ValueError("Unsupported observation schema; expected team-v2.")
    if (isinstance(result["history_frames"], bool)
            or not isinstance(result["history_frames"], int)
            or result["history_frames"] != frames):
        raise ValueError("Contract history_frames does not match the environment observation shape.")
    names = result["feature_names"]
    if (not isinstance(names, (list, tuple)) or len(names) != len(DEFAULT_FEATURE_NAMES)
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Contract must define 26 unique feature names in order.")
    if result["action_names"] != [list(action) for action in ACTIONS]:
        raise ValueError("Contract action ordering differs from the supported action mapping.")
    for field in ("position_scale", "damage_scale", "damage_reward_scale"):
        value = result[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"Contract {field} must be a positive finite number.")
    try:
        return json.loads(_json_bytes(result))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Observation contract must contain JSON-safe finite values: {error}") from error


def _check_policy(model, *, allowed_sizes):
    from gymnasium import spaces
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    if (type(model.policy) is not ActorCriticPolicy
            or type(model.policy.features_extractor) is not FlattenExtractor):
        raise ValueError("Only structured-state PPO MlpPolicy checkpoints can transfer; CNN pixels are incompatible.")
    if (not isinstance(model.observation_space, spaces.Box)
            or model.observation_space.shape not in {(size,) for size in allowed_sizes}):
        raise ValueError("Checkpoint has an incompatible observation shape.")
    action = model.action_space
    if (not isinstance(action, spaces.Discrete) or action.n != len(ACTIONS)
            or action.start != 0):
        raise ValueError("Checkpoint action space does not match the Brawlhalla action mapping.")
    weights = model.policy.state_dict()
    first_layers = ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight")
    if any(name not in weights or weights[name].ndim != 2 for name in first_layers):
        raise ValueError("Unsupported policy architecture: actor and value networks need an initial linear layer.")


class ModelStore:
    def __init__(self, root="models/sessions", model_kwargs=None):
        self.root = Path(root).expanduser()
        self.model_kwargs = deepcopy(model_kwargs or {})

    def path_for(self, match_format):
        if match_format not in MATCH_FORMATS:
            raise ValueError("match_format must be duel or teams.")
        return self.root / match_format / "latest.zip"

    def _new(self, env, device, *, policy_kwargs=None):
        from stable_baselines3 import PPO
        options = {"n_steps": 256, "batch_size": 64, **deepcopy(self.model_kwargs)}
        if policy_kwargs is not None:
            options["policy_kwargs"] = deepcopy(policy_kwargs)
        return PPO("MlpPolicy", env, device=device, **options)

    def _metadata(self, path, *, required=True):
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            if required:
                raise ValueError(f"Checkpoint metadata is missing: {sidecar}. Use an explicit new-branch initialization for legacy files.")
            return None
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read checkpoint metadata {sidecar}: {error}") from error
        if not isinstance(data, dict):
            raise ValueError(f"Checkpoint metadata must be a JSON object: {sidecar}")
        if data.get("checkpoint_sha256") != _sha256(path):
            raise ValueError("Checkpoint hash does not match metadata; the saved pair may be incomplete or changed.")
        if (data.get("observation_schema") != OBSERVATION_SCHEMA
                or data.get("action_names_sha256") != ACTION_NAMES_SHA256):
            raise ValueError("Checkpoint observation schema or action ordering is incompatible.")
        if not isinstance(data.get("contract"), dict):
            raise ValueError("Checkpoint metadata has no observation contract.")
        return data

    def _lineage(self, strategy, match_format, contract, source=None, source_hash=None):
        return {"strategy": strategy, "match_format": match_format,
                "observation_schema": OBSERVATION_SCHEMA,
                "action_names_sha256": ACTION_NAMES_SHA256,
                "contract": contract,
                "source_path": str(source.resolve()) if source is not None else None,
                "source_sha256": source_hash}

    def load_or_create(self, env, match_format, device="cpu", initialize_from=None,
                       evaluation=False, contract=None):
        """Resume this branch, or initialize a fresh optimizer from trusted weights.

        A new teams branch automatically inherits a compatible duel branch.
        Explicit legacy 10/26-feature initialization supports history expansion;
        old weights read the newest frame and ignore earlier history initially.
        """
        from stable_baselines3 import PPO
        size = _observation_size(env)
        normalized = _contract(contract, size // len(DEFAULT_FEATURE_NAMES))
        destination = self.path_for(match_format)
        if destination.exists():
            if initialize_from is not None:
                raise ValueError("initialize_from is allowed only for a new branch; existing branch progress must resume.")
            metadata = self._metadata(destination)
            if metadata.get("match_format") != match_format or metadata["contract"] != normalized:
                raise ValueError("Saved checkpoint contract differs from this session; preserve the branch and use a separate model store.")
            model = PPO.load(str(destination), env=env, device=device)
            _check_policy(model, allowed_sizes=(size,))
            lineage = self._lineage("resumed", match_format, normalized,
                                    destination, metadata["checkpoint_sha256"])
            for key in ("initialization_strategy", "initialization_source_path", "initialization_source_sha256"):
                lineage[key] = metadata.get(key)
            return model, lineage
        if evaluation:
            raise FileNotFoundError(f"No saved {match_format} policy to evaluate: {destination}")
        source = None
        explicit = initialize_from is not None
        if explicit:
            source = Path(initialize_from).expanduser()
            if not source.exists() and not str(source).endswith(".zip"):
                source = Path(str(source) + ".zip")
            if source.resolve() == destination.resolve():
                raise ValueError("Initialization source must differ from the destination branch.")
            if not source.is_file():
                raise FileNotFoundError(f"Initialization checkpoint does not exist: {source}")
        elif match_format == "teams" and self.path_for("duel").exists():
            source = self.path_for("duel")
        if source is None:
            return self._new(env, device), self._lineage("new", match_format, normalized)
        metadata = self._metadata(source, required=not explicit)
        old = PPO.load(str(source), device=device)
        _check_policy(old, allowed_sizes=(size, 10, 26))
        old_size = old.observation_space.shape[0]
        migrating = old_size != size
        if migrating and (not explicit or old_size not in (10, 26) or old_size > size):
            raise ValueError("Only explicit legacy 10/26-feature policies can expand into the newest history frame.")
        if metadata is not None:
            source_contract = deepcopy(metadata["contract"])
            if migrating and old_size == 26:
                source_contract["history_frames"] = normalized["history_frames"]
            if source_contract != normalized:
                raise ValueError("Initialization checkpoint contract differs from this session.")
        else:
            known = _contract(None, size // 26)
            known_fields = ("feature_names", "history_order", "position_scale", "damage_scale",
                            "damage_reward_scale", "slot_semantics", "action_names")
            if (any(normalized[name] != known[name] for name in known_fields)
                    or tuple(normalized["feature_names"][:10]) != tuple(LEGACY_FEATURE_NAMES)):
                raise ValueError("Legacy checkpoints without metadata require the defined code feature ordering and default scales.")
        model = self._new(env, device, policy_kwargs=old.policy_kwargs)
        old_weights = old.policy.state_dict()
        new_weights = model.policy.state_dict()
        first_layers = {"mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"}
        if set(old_weights) != set(new_weights):
            raise ValueError("Unsupported policy architecture: checkpoint parameter names differ.")
        for name, tensor in old_weights.items():
            if migrating and name in first_layers:
                target = new_weights[name].clone().zero_()
                if target.shape[0] != tensor.shape[0] or tensor.shape[1] != old_size:
                    raise ValueError("Unsupported initial actor/value layer architecture.")
                start = size - 26
                target[:, start:start + old_size] = tensor
                new_weights[name] = target
            elif tensor.shape != new_weights[name].shape:
                raise ValueError(f"Incompatible checkpoint architecture at {name}.")
            else:
                new_weights[name] = tensor.detach().clone()
        model.policy.load_state_dict(new_weights, strict=True)
        strategy = "migrated_legacy" if migrating else "warm_started"
        lineage = self._lineage(strategy, match_format, normalized, source, _sha256(source))
        lineage.update(initialization_strategy=strategy,
                       initialization_source_path=lineage["source_path"],
                       initialization_source_sha256=lineage["source_sha256"])
        return model, lineage

    def save(self, model, match_format, lineage, contract=None):
        """Stage complete artifacts before replacing the branch checkpoint.

        A process failure between the two replacements is detected by the
        checkpoint digest on the next load; a mismatched pair is never resumed.
        """
        destination = self.path_for(match_format)
        size = _observation_size(model)
        _check_policy(model, allowed_sizes=(size,))
        if not isinstance(lineage, dict) or lineage.get("match_format") != match_format:
            raise ValueError("Lineage must belong to the destination match format.")
        normalized = _contract(contract if contract is not None else lineage.get("contract"), size // 26)
        if lineage.get("contract") != normalized:
            raise ValueError("Saving cannot change the checkpoint observation contract.")
        source = lineage.get("source_path")
        if (source is not None and lineage.get("strategy") != "resumed"
                and Path(source).resolve() == destination.resolve()):
            raise ValueError("Saving a warm start must never overwrite its source checkpoint.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = []
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".zip", delete=False) as stream:
                checkpoint_temp = Path(stream.name)
            temporary.append(checkpoint_temp)
            model.save(str(checkpoint_temp))
            metadata = {**deepcopy(lineage), "lineage": deepcopy(lineage), "metadata_version": 1,
                        "checkpoint_sha256": _sha256(checkpoint_temp)}
            payload = _json_bytes(metadata) + b"\n"
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".json", delete=False) as stream:
                metadata_temp = Path(stream.name)
                temporary.append(metadata_temp)
                stream.write(payload)
            os.replace(checkpoint_temp, destination)
            os.replace(metadata_temp, destination.with_suffix(".json"))
            return destination
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
