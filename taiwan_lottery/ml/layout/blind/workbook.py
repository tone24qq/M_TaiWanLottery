"""Inspect the supplied manufacturing XLSX; never simulate its RAND source."""
import hashlib
from pathlib import Path,PurePosixPath
import re
import xml.etree.ElementTree as ET
import zipfile

MAIN='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
NS={'m':MAIN,'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def translated_formula(formula,source,destination):
    def location(reference):
        col,row=re.fullmatch(r'([A-Z]+)([0-9]+)',reference).groups()
        n=0
        for c in col:n=n*26+ord(c)-64
        return n,int(row)
    sc,sr=location(source);dc,dr=location(destination)
    def replace(match):
        locked_col,col,locked_row,row=match.groups()
        ci,_=location(col+'1')
        ci=ci if locked_col else ci+dc-sc
        ri=int(row) if locked_row else int(row)+dr-sr
        if ci<1 or ri<1:raise ValueError('Invalid relative formula translation')
        letters=''
        while ci:ci,rem=divmod(ci-1,26);letters=chr(65+rem)+letters
        return locked_col+letters+locked_row+str(ri)
    return re.sub(r'(\$?)([A-Z]+)(\$?)([0-9]+)',replace,formula)


def read_sheet(workbook,name):
    with zipfile.ZipFile(workbook) as archive:
        if any(p.startswith('xl/externalLinks/') or p.endswith('vbaProject.bin') for p in archive.namelist()):
            raise ValueError('External links/macros require separate review')
        root=ET.fromstring(archive.read('xl/workbook.xml'))
        sheets={s.attrib['name']:s.attrib['{'+NS['r']+'}id'] for s in root.find('m:sheets',NS)}
        rels=ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets={r.attrib['Id']:r.attrib['Target'] for r in rels}
        target=targets[sheets[name]]
        path=target.lstrip('/') if target.startswith('/') else str(PurePosixPath('xl')/target)
        sheet=ET.fromstring(archive.read(path))
        cells={c.attrib['r']:c for c in sheet.findall('.//m:sheetData/m:row/m:c',NS)}
        shared={}
        for ref,c in cells.items():
            f=c.find('m:f',NS)
            if f is not None and f.attrib.get('t')=='shared' and f.text:
                shared[f.attrib['si']]=(ref,f.text)
        result={}
        for ref,c in cells.items():
            f=c.find('m:f',NS);value=c.find('m:v',NS)
            formula=None
            if f is not None:
                formula=f.text
                if f.attrib.get('t')=='shared' and not formula:
                    source,text=shared[f.attrib['si']]
                    formula=translated_formula(text,source,ref)
            result[ref]={'formula':formula,'cached_value':value.text if value is not None else None}
        return result


def validate_values(values):
    if len(values)!=160 or any(isinstance(v,bool) or int(v)!=v for v in values):
        raise ValueError('Require 160 integer values')
    values=[int(v) for v in values]
    if sorted(values)!=list(range(1,161)):raise ValueError('Not a complete 1..160 permutation')
    return values


def inspect_workbook(path):
    path=Path(path);sheet=read_sheet(path,'隨機產生號碼')
    for i in range(1,161):
        if (sheet[f'C{i}']['formula'] or '').upper()!='RAND()':raise ValueError('Unexpected C-column manufacturing formula')
        if (sheet[f'D{i}']['formula'] or '').upper()!=f'RANK(C{i},$C$1:$C$160)':
            raise ValueError('Unexpected D-column manufacturing formula')
    printed=read_sheet(path,'160刮')
    formulas={ref:c['formula'] for ref,c in printed.items() if c['formula'] and '隨機產生號碼!D' in c['formula']}
    cached=validate_values([float(sheet[f'D{i}']['cached_value']) for i in range(1,161)])
    c=[float(sheet[f'C{i}']['cached_value']) for i in range(1,161)]
    ranks=[1+sum(other>x for other in c) for x in c]
    if ranks!=cached:raise ValueError('Cached D values disagree with descending RAND ranks')
    return {'workbook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_sheet':'隨機產生號碼','manufacturing_range':'D1:D160',
        'random_formula':'C1:C160 = RAND()',
        'rank_formula':'D_i = RANK(C_i,$C$1:$C$160), default descending order',
        'mechanism':'actual_manufacturing_RAND_RANK_permutation',
        'formula_cells_verified':320,'cached_permutation_verified':True,
        'cached_rank_matches_random_values':True,
        'print_layout_array_formulas':formulas,
        'logical_reshape':{'rows':10,'cols':16,'order':'D1..D160 row-major, explicitly requested; differs from print layout'},
        'inference_consequence':'Under independent continuous RAND draws, each hidden target location is exchangeable, conditional on the visible values. No positional method can systematically beat 1/unknown_count without extra information.',
        'limits':'Formula inspection does not certify PRNG independence or equivalence of spreadsheet engines. Finite RAND ties must be rejected, never silently repaired.'}
