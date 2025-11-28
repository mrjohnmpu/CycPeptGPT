"""
RDKit util functions.
"""
import rdkit.Chem as rkc
from rdkit.Chem import AllChem
from rdkit import DataStructs

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()

def to_fp_Morgan(smi):
    if smi:
        mol = rkc.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprint(mol, radius=4, useChirality=True, useCounts=True)

def tanimoto_similarity_pool(args):
    return tanimoto_similarity(*args)

def tanimoto_similarity(smi1, smi2):
    fp1, fp2 = None, None
    if smi1 and type(smi1)==str and len(smi1)>0:
        fp1 = to_fp_Morgan(smi1)
    if smi2 and type(smi2)==str and len(smi2)>0:
        fp2 = to_fp_Morgan(smi2)

    if fp1 is not None and fp2 is not None:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    else:
        return None
