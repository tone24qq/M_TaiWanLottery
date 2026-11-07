"""Formula validation only; no replacement manufacturing generator."""
import os
import pytest
from taiwan_lottery.ml.layout.blind.workbook import translated_formula,validate_values,inspect_workbook


def test_shared_excel_rank_formulas_preserve_absolute_range():
    assert translated_formula('RANK(C2,$C$1:$C$160)','D2','D65')=='RANK(C65,$C$1:$C$160)'
    assert translated_formula('RAND()','C2','C65')=='RAND()'
    assert translated_formula('RANK(C139,$C$1:$C$160)','D139','D160')=='RANK(C160,$C$1:$C$160)'


def test_manufacturing_values_must_be_complete_without_repair():
    with pytest.raises(ValueError):validate_values([1]*160)
    with pytest.raises(ValueError):validate_values([1,2])
    with pytest.raises(ValueError):validate_values([False]*160)
    with pytest.raises(ValueError):validate_values([1.5]*160)


@pytest.mark.skipif(not os.getenv('LAYOUT_MANUFACTURING_WORKBOOK'),reason='private source workbook not supplied')
def test_actual_supplied_workbook_formulas_and_cached_rank_consistency():
    report=inspect_workbook(os.environ['LAYOUT_MANUFACTURING_WORKBOOK'])
    assert report['formula_cells_verified']==320
    assert report['cached_permutation_verified']
    assert report['cached_rank_matches_random_values']
    assert report['logical_reshape']['rows']==10
    assert report['logical_reshape']['cols']==16
    assert len(report['print_layout_array_formulas'])==20
