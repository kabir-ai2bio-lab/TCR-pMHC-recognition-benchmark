import pandas as pd
import pytest

from PhongLu.data.preprocessing.import_sources import import_iedb, import_vdjdb
from PhongLu.data.preprocessing.extract_vdjdb_v1 import build_vdjdb_v1
from PhongLu.data.preprocessing.extract_vdjdb_structural_candidates import build_structural_candidates
from PhongLu.data.preprocessing.preprocess import preprocess
from PhongLu.data.preprocessing.split import add_within_fold_negatives, assert_strict_no_leakage, make_folds


def _positive_data():
    return pd.DataFrame({
        "cdr3_beta": ["CASSLGQETQYF", "CASSIRSSYEQYF", "CASSQETQYF", "CASSLGQETQYF"],
        "peptide": ["NLVPMVATV", "GILGFVFTL", "LLFGYPVYV", "NLVPMVATV"],
        "mhc": ["HLA-A*02:01"] * 4,
        "source": ["test"] * 4,
    })


def test_preprocessing_filters_and_audits_duplicates():
    processed, audit = preprocess(_positive_data())
    assert len(processed) == 3
    assert audit[-1]["stage"] == "duplicate_cdr3_beta_peptide"
    assert processed.peptide.str.len().eq(9).all()


def test_strict_split_has_no_peptide_leakage():
    data = pd.concat([_positive_data().iloc[:3]] * 3, ignore_index=True)
    data["cdr3_beta"] = [f"CASS{index:02d}GQETQYF" for index in range(len(data))]
    folds = make_folds(data, "strict", n_folds=3, seed=7)
    assert_strict_no_leakage(folds)


def test_uniform_split_rejects_peptides_that_cannot_reach_every_fold():
    with pytest.raises(ValueError, match="at least five"):
        make_folds(_positive_data().iloc[:3], "uniform")


def test_negatives_are_fold_local_and_not_positives():
    rows = pd.DataFrame({
        "cdr3_beta": ["CASSAAAQYF", "CASSBBBQYF", "CASSCCCQYF"],
        "peptide": ["NLVPMVATV", "GILGFVFTL", "LLFGYPVYV"],
    })
    result = add_within_fold_negatives(rows, seed=1)
    positive_pairs = set(map(tuple, result.loc[result.binder.eq(1), ["cdr3_beta", "peptide"]].itertuples(index=False, name=None)))
    negative_pairs = set(map(tuple, result.loc[result.binder.eq(0), ["cdr3_beta", "peptide"]].itertuples(index=False, name=None)))
    assert positive_pairs.isdisjoint(negative_pairs)
    assert result.binder.value_counts().to_dict() == {1: 3, 0: 3}


def test_missing_required_column_fails_loudly():
    with pytest.raises(ValueError, match="cdr3_beta"):
        preprocess(pd.DataFrame({"peptide": ["NLVPMVATV"]}))


def test_iedb_adapter_prefers_calculated_beta_cdr3(tmp_path):
    path = tmp_path / "iedb.csv"
    first = ["Receptor", "Reference", "Epitope", "Assay", "Chain 1", "Chain 1", "Chain 1"]
    second = ["CEDAR Receptor ID", "CEDAR IRI", "Name", "MHC Allele Names", "Type", "CDR3 Curated", "CDR3 Calculated"]
    row = ["id1", "ref1", "NLVPMVATV", "HLA-A*02:01", "beta", "CASSCURATEDF", "CASSCALCULF"]
    path.write_text("\n".join(",".join(items) for items in (first, second, row)) + "\n")
    result = import_iedb(path)
    assert result.loc[0, "cdr3_beta"] == "CASSCALCULF"
    assert result.loc[0, "peptide"] == "NLVPMVATV"


def test_vdjdb_adapter_reads_positive_chunk(tmp_path):
    chunk = tmp_path / "example.txt"
    chunk.write_text(
        "chunk.id\tcdr3.beta\tantigen.epitope\tmhc.a\tmhc.class\tspecies\treference.id\n"
        "7\tCASSLGQETQYF\tNLVPMVATV\tHLA-A*02:01\tMHCI\tHomoSapiens\tPMID:1\n"
    )
    result = import_vdjdb(tmp_path)
    assert result.loc[0, "source"] == "vdjdb"
    assert result.loc[0, "source_record_id"] == "example.txt:7"


def test_vdjdb_v1_extraction_requires_human_class_i_and_deng_filters(tmp_path):
    chunk = tmp_path / "example.txt"
    chunk.write_text(
        "chunk.id\tcdr3.beta\tantigen.epitope\tmhc.a\tmhc.class\tspecies\treference.id\n"
        "1\tCASSLGQETQYF\tNLVPMVATV\tHLA-A*02:01\tMHCI\tHomoSapiens\tPMID:1\n"
        "2\tCASSLGQETQYF\tNLVPMVATV\tHLA-A*03:01\tMHCI\tHomoSapiens\tPMID:2\n"
        "3\tCASSLGQETQYF\tNLVPMVATV\tHLA-A*02:01\tMHCII\tHomoSapiens\tPMID:3\n"
    )
    result, audit = build_vdjdb_v1(tmp_path)
    assert len(result) == 1
    assert result.loc[0, "mhc"] == "HLA-A*02:01"
    assert audit[0]["stage"] == "not_homo_sapiens"


def test_structural_candidate_extraction_requires_paired_chains_and_vj_genes(tmp_path):
    chunk = tmp_path / "example.txt"
    chunk.write_text(
        "chunk.id\tcdr3.alpha\tv.alpha\tj.alpha\tcdr3.beta\tv.beta\td.beta\tj.beta\t"
        "antigen.epitope\tmhc.a\tmhc.b\tmhc.class\tspecies\treference.id\tmeta.structure.id\n"
        "1\tCAVRDSNYQLIW\tTRAV12-2\tTRAJ33\tCASSLGQETQYF\tTRBV7-9\tTRBD1\tTRBJ2-7\t"
        "NLVPMVATV\tHLA-A*02:01\tB2M\tMHCI\tHomoSapiens\tPMID:1\t\n"
        "2\t\t\t\tCASSLGQETQYF\tTRBV7-9\tTRBD1\tTRBJ2-7\t"
        "NLVPMVATV\tHLA-A*02:01\tB2M\tMHCI\tHomoSapiens\tPMID:2\t\n"
    )
    result, audit = build_structural_candidates(tmp_path)
    assert len(result) == 1
    assert result.loc[0, "cdr3_alpha"] == "CAVRDSNYQLIW"
    assert result.loc[0, "v_beta"] == "TRBV7-9"
    assert result.loc[0, "label"] == 1
    assert audit[-2]["stage"] == "missing_alpha_or_beta_vj_gene"
