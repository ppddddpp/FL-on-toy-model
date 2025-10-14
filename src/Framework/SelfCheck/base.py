from .SelfCheck import SelfCheckManager
import numpy as np

# =============================================================================
# [Scenario 1] Mild vs Extreme Anomaly Detection Test
# =============================================================================
def run_selfcheck_scenario_1():
    print("\n[Scenario 1] Mild vs Extreme Anomaly Detection Test")

    fake_updates = {
        "client_normal": np.random.randn(10) * 1.0,
        "client_mild": np.random.randn(10) * 5.0,   # moderate abnormal
        "client_extreme": np.random.randn(10) * 50.0  # strong abnormal
    }

    print("\n[Debug] Stats:")
    for cid, vec in fake_updates.items():
        print(f"  {cid:12s} | mean={np.mean(vec):8.3f} | std={np.std(vec):8.3f} | norm={np.linalg.norm(vec):8.3f}")

    manager = SelfCheckManager()
    for r in range(1, 3):
        print(f"\n=== Round {r} ===")
        result = manager.run_round(fake_updates, round_id=r)
        print(result)

# =============================================================================
# [Scenario 2] Gradual Drift + Sudden Poison Attack
# =============================================================================
def run_selfcheck_scenario_2():
    print("\n[Scenario 2] Gradual Drift + Sudden Poison Attack")

    rng = np.random.default_rng(123)
    base = rng.normal(0, 1, 10)
    manager = SelfCheckManager()

    for r in range(1, 8):
        if r < 5:
            drift_factor = 1.0 + 0.2 * r
        else:
            drift_factor = 10.0  # sudden poisoning

        fake_updates = {
            "client_A": base * 1.0,
            "client_B": base * drift_factor,  # drifting
            "client_C": base * 50.0 if r == 6 else base * 1.0  # one-time strong outlier
        }

        print(f"\n=== Round {r} | drift_factor={drift_factor:.2f} ===")
        result = manager.run_round(fake_updates, round_id=r)


# =============================================================================
# [Scenario 3] Collusion Attack Simulation
# =============================================================================
def run_selfcheck_scenario_3():
    print("\n[Scenario 3] Collusion Attack Simulation")

    rng = np.random.default_rng(99)
    base = rng.normal(0, 1, 10)
    manager = SelfCheckManager()

    for r in range(1, 4):
        fake_updates = {
            f"client_{i}": base * (2.0 if i < 6 else 1.0)
            for i in range(1, 11)
        }
        print(f"\n=== Round {r} (Collusion) ===")
        result = manager.run_round(fake_updates, round_id=r)


# =============================================================================
# [Scenario 4] Noise Flooding (Entropy Stress Test)
# =============================================================================
def run_selfcheck_scenario_4():
    print("\n[Scenario 4] Noise Flooding (Entropy Stress Test)")

    def random_entropy_vector(size=10, scale=1.0):
        base = np.random.randn(size)
        noise = np.random.randn(size) * scale
        return base + noise

    fake_updates = {
        "client_clean": np.random.randn(10),
        "client_noisy": random_entropy_vector(10, 5.0),
        "client_flood": random_entropy_vector(10, 15.0),
    }

    print("\n[Debug] Norms and stats for entropy flood:")
    for cid, vec in fake_updates.items():
        print(f"  {cid:12s} | mean={np.mean(vec):8.3f} | std={np.std(vec):8.3f} | norm={np.linalg.norm(vec):8.3f}")

    manager = SelfCheckManager()
    for r in range(1, 4):
        print(f"\n=== Round {r} ===")
        results = manager.run_round(fake_updates, round_id=r)
        print(results)


# =============================================================================
# [Scenario 5] Label-Shift-Like Drift (Cosine Anomaly)
# =============================================================================
def run_selfcheck_scenario_5():
    print("\n[Scenario 5] Label-Shift-Like Drift (Cosine Feature Anomaly)")

    rng = np.random.default_rng(2025)
    base_vec = rng.normal(0, 1, 10)
    shift_dir = rng.normal(0, 1, 10)

    fake_updates = {
        "client_base": base_vec,
        "client_shift1": base_vec + 0.3 * shift_dir,
        "client_shift2": base_vec + 1.0 * shift_dir,
    }

    print("\n[Debug] Cosine-based anomalies:")
    for cid, vec in fake_updates.items():
        cos_sim = np.dot(base_vec, vec) / (np.linalg.norm(base_vec) * np.linalg.norm(vec))
        print(f"  {cid:12s} | cos_sim={cos_sim:6.3f} | norm={np.linalg.norm(vec):6.3f}")

    manager = SelfCheckManager()
    for r in range(1, 4):
        print(f"\n=== Round {r} ===")
        results = manager.run_round(fake_updates, round_id=r)
        print(results)


# =============================================================================
# [Scenario 6] Adaptive Poisoning (Persistence Test)
# =============================================================================
def run_selfcheck_scenario_6():
    print("\n[Scenario 6] Adaptive Poisoning (Persistence Test)")

    rng = np.random.default_rng(42)
    poisoned_client = rng.normal(0, 50, 10)
    base_client = rng.normal(0, 1, 10)
    manager = SelfCheckManager()

    for round_id in range(1, 9):
        poison_factor = 1.0 + 0.1 * round_id
        fake_updates = {
            "client_safe": base_client,
            "client_poison": poisoned_client * poison_factor,
        }

        print(f"\n=== Round {round_id} | poison_factor={poison_factor:.2f} ===")
        results = manager.run_round(fake_updates, round_id=round_id)
        print(results)


# =============================================================================
# [Scenario 7] Hybrid Collusion + Adaptive Poison Drift Test
# =============================================================================
def run_selfcheck_scenario_7():
    print("\n[Scenario 7] Hybrid Collusion + Adaptive Poison Drift Test")

    rng = np.random.default_rng(2025)
    manager = SelfCheckManager()

    # --- Configuration ---
    num_rounds = 10
    num_benign = 4
    num_malicious = 3

    # Base update pattern (benign behavior)
    base_update = rng.normal(0, 1, 20)

    # Initialize cumulative poison factor (starts small)
    poison_factor = 1.0
    poison_increment = 0.15  # gradual poison intensity per round

    for r in range(1, num_rounds + 1):
        poison_factor += poison_increment  # increase poison strength
        print(f"\n=== Round {r} | poison_factor={poison_factor:.2f} ===")

        fake_updates = {}

        # Benign clients (stable)
        for i in range(num_benign):
            cid = f"ben_{i+1}"
            fake_updates[cid] = base_update + rng.normal(0, 0.2, size=20)

        # Malicious clients (colluding with controlled drift)
        for j in range(num_malicious):
            cid = f"mal_{j+1}"
            drift = poison_factor * (base_update + rng.normal(0, 0.5, size=20))
            fake_updates[cid] = drift

        # Optional: Inject synchronized cosine shift after round 5
        if r > 5:
            for cid in fake_updates:
                if cid.startswith("mal_"):
                    # Push them to same directional subspace (collusion)
                    fake_updates[cid] *= np.cos(0.3 * r)

        # Run round
        result = manager.run_round(fake_updates, round_id=r)
        manager.triage.report(result)