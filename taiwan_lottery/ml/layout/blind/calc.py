"""Recalculate original XLSX using actual Calc and verified completion callbacks.

C API layout: LibreOffice/include/LibreOfficeKit/LibreOfficeKit.h (official).
Python never supplies RAND values, random permutations, or manufacturing seeds.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import zipfile
import xml.etree.ElementTree as ET
from .workbook import inspect_workbook,read_sheet,validate_values,MAIN
from .protocol import save_new

OFFICE=['destroy','documentLoad','getError','documentLoadWithOptions','freeError','registerCallback','getFilterTypes','setOptionalFeatures','setDocumentPassword','getVersionInfo']
DOCUMENT=['destroy','saveAs','getDocumentType','getParts','getPartPageRectangles','getPart','setPart','getPartName','setPartMode','paintTile','getTileMode','getDocumentSize','initializeForRendering','registerCallback','postKeyEvent','postMouseEvent','postUnoCommand']
class OfficeTable(C.Structure):_fields_=[('nSize',C.c_size_t)]+[(n,C.c_void_p) for n in OFFICE]
class Office(C.Structure):_fields_=[('table',C.POINTER(OfficeTable))]
class DocumentTable(C.Structure):_fields_=[('nSize',C.c_size_t)]+[(n,C.c_void_p) for n in DOCUMENT]
class Document(C.Structure):_fields_=[('table',C.POINTER(DocumentTable))]
OP=C.POINTER(Office);DP=C.POINTER(Document)


def frozen_values(path):
    sheet=read_sheet(path,'隨機產生號碼')
    values=validate_values([float(sheet[f'D{i}']['cached_value']) for i in range(1,161)])
    draws=[float(sheet[f'C{i}']['cached_value']) for i in range(1,161)]
    if not all(0<=v<1 for v in draws) or len(set(draws))!=160:raise ValueError('Invalid/tied Calc RAND values')
    if [1+sum(other>x for other in draws) for x in draws]!=values:raise ValueError('Calc D values disagree with frozen C ranks')
    return values,draws


def recalculate(source,output,count,program):
    source=Path(source).resolve();output=Path(output).resolve();program=Path(program).resolve()
    if 'unipoll' in os.environ.get('SAL_LOK_OPTIONS',''):raise ValueError('Use ordinary threaded LOK with completion callbacks')
    audit=inspect_workbook(source);output.mkdir(parents=True,exist_ok=False)
    library=C.CDLL(str(program/'libsofficeapp.so'),mode=C.RTLD_GLOBAL)
    library.libreofficekit_hook_2.argtypes=[C.c_char_p,C.c_char_p];library.libreofficekit_hook_2.restype=OP
    with tempfile.TemporaryDirectory(prefix='lottery-calc-profile-') as tmp:
        office=library.libreofficekit_hook_2(str(program).encode(),Path(tmp).as_uri().encode())
        if not office:raise RuntimeError('LibreOfficeKit initialization failed')
        ot=office.contents.table.contents
        version=json.loads(C.CFUNCTYPE(C.c_char_p,OP)(ot.getVersionInfo)(office))
        doc=C.CFUNCTYPE(DP,OP,C.c_char_p)(ot.documentLoad)(office,source.as_uri().encode())
        if not doc:raise RuntimeError('Original XLSX load failed')
        dt=doc.contents.table.contents
        if dt.nSize<C.sizeof(DocumentTable):raise RuntimeError('Unsupported LibreOfficeKit ABI')
        C.CFUNCTYPE(None,DP,C.c_char_p)(dt.initializeForRendering)(doc,b'{}')
        completion=threading.Event();completed=[]
        @C.CFUNCTYPE(None,C.c_int,C.c_char_p,C.c_void_p)
        def callback(kind,payload,data):
            if kind==16:  # Official LOK_CALLBACK_UNO_COMMAND_RESULT.
                try:
                    result=json.loads(payload)
                    if result.get('commandName')=='.uno:CalculateHard':
                        completed.append(result);completion.set()
                except Exception:pass
        C.CFUNCTYPE(None,DP,C.c_void_p,C.c_void_p)(dt.registerCallback)(doc,C.cast(callback,C.c_void_p),None)
        seen=set();draw_vectors=set();snapshots=[];settings=None
        for i in range(count):
            completion.clear()
            C.CFUNCTYPE(None,DP,C.c_char_p,C.c_char_p,C.c_bool)(dt.postUnoCommand)(doc,b'.uno:CalculateHard',b'{}',True)
            if not completion.wait(30) or len(completed)!=i+1 or completed[-1].get('success') is not True:
                raise RuntimeError('Calc hard-recalculation completion not confirmed')
            folder=output/f'board-{i:06d}';folder.mkdir();path=folder/source.name
            if not C.CFUNCTYPE(C.c_int,DP,C.c_char_p,C.c_char_p,C.c_char_p)(dt.saveAs)(doc,path.as_uri().encode(),b'xlsx',None):
                raise RuntimeError('Calc snapshot export failed')
            values,draws=frozen_values(path)
            if tuple(values) in seen or tuple(draws) in draw_vectors:raise ValueError('Repeated manufacturing permutation/draw')
            seen.add(tuple(values));draw_vectors.add(tuple(draws))
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            save_new(folder/'values.json',{'rows':10,'cols':16,'cells':values,'snapshot_sha256':sha})
            save_new(folder/'recalculation.json',completed[-1]);path.chmod(0o444)
            snapshots.append({'index':i,'snapshot_sha256':sha})
            if settings is None:
                with zipfile.ZipFile(path) as archive:
                    root=ET.fromstring(archive.read('xl/workbook.xml'));settings=root.find('{'+MAIN+'}calcPr').attrib
            if (i+1)%100==0:print('Verified Calc recalculations:',i+1,flush=True)
        C.CFUNCTYPE(None,DP)(dt.destroy)(doc);C.CFUNCTYPE(None,OP)(ot.destroy)(office)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=audit['workbook_sha256']:raise ValueError('Source XLSX modified')
    manifest={'engine':version,'count':count,'distinct_permutations':len(seen),'source_audit':audit,
        'recalculation':'.uno:CalculateHard before every export; wait for LOK_CALLBACK_UNO_COMMAND_RESULT success=true',
        'completion_callbacks_verified':len(completed),'exported_calcPr':settings,
        'automatic_calculation':'OOXML calcMode omitted (automatic); every sample additionally forced through hard recalculation',
        'frozen_values':'D1:D160 read from exported cached XML, immediately copied to immutable values.json before inference',
        'rng_state':'Never set, serialized or exposed to predictor; frozen C values remain scorer-only',
        'snapshots':snapshots,'caveat':'LibreOffice Calc recalculation of original XLSX formulas; not Microsoft Excel-native RAND sequence reproduction'}
    save_new(output/'calc_manifest.json',manifest);return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','program'):p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,required=True)
    a=p.parse_args();recalculate(a.source,a.output,a.count,a.program)
