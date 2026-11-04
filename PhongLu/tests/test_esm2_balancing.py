"""Safety checks for peptide-frequency sampling in the ESM-2 classifier."""

from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from PhongLu.Modeleval.models.peptide_balancing import make_peptide_sampler, peptide_balanced_weights


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "peptide": ["COMMON"] * 4 + ["RARE"],
        "binder": [1, 0, 1, 0, 1],
    })


def test_balance_power_has_documented_endpoints_and_gentle_midpoint():
    train = _training_rows()
    no_balance = peptide_balanced_weights(train, 0.0)
    gentle = peptide_balanced_weights(train, 0.5)
    full = peptide_balanced_weights(train, 1.0)
    assert np.allclose(no_balance, 1.0)
    assert gentle[-1] / gentle[0] == pytest.approx(2.0)
    assert full[-1] / full[0] == pytest.approx(4.0)


def test_weights_use_only_the_dataframe_passed_as_training_data():
    train = _training_rows()
    held_out = pd.DataFrame({"peptide": ["COMMON"] * 100 + ["TEST_ONLY"] * 100})
    before = peptide_balanced_weights(train, 0.5)
    _combined_but_not_passed = pd.concat([train, held_out], ignore_index=True)
    after = peptide_balanced_weights(train, 0.5)
    assert np.array_equal(before, after)
    assert "TEST_ONLY" not in set(train.peptide)


def test_sampler_is_reproducible_for_a_fixed_seed():
    train = _training_rows()
    first = list(iter(make_peptide_sampler(train, 0.5, seed=123)))
    second = list(iter(make_peptide_sampler(train, 0.5, seed=123)))
    assert first == second


def test_balance_power_rejects_values_outside_controlled_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        peptide_balanced_weights(_training_rows(), 1.1)


def test_prepared_split_files_are_not_modified_by_weight_calculation():
    paths = [
        "PhongLu/data/preprocessing/splits/matched_strict_five_fold.csv",
        "PhongLu/data/preprocessing/splits/matched_uniform_five_fold.csv",
    ]
    before = {path: sha256(open(path, "rb").read()).hexdigest() for path in paths}
    for path in paths:
        data = pd.read_csv(path)
        peptide_balanced_weights(data[data.fold.ne(0)], 0.5)
    after = {path: sha256(open(path, "rb").read()).hexdigest() for path in paths}
    assert before == after
