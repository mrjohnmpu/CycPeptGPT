from rdkit import Chem
from pepinvent.scoring_function.scoring_components.predictive_model import PredictiveModel
from pepinvent.scoring_function.scoring_components.scoring_component_parameters import ScoringComponentParameters
import os
import pickle
from rdkit.Chem import AllChem
import numpy as np

def predict_permeability(peptide_smiles):
    """
    PepINVENT researches created a privately sourced permeability model.
    This loads the model and calls its prediction function.
    
    Args:
        peptide_smiles (str): SMILES string of the peptide
        
    Returns:
        float: Probability of the peptide being permeable (0-1)
    """
    try:
        # Load model and scalar directly with pickle
        model_path = os.path.join(os.getcwd(), "model", "pepinvent", "models", "predictive_model.pckl")
        scalar_path = os.path.join(os.getcwd(), "model", "pepinvent", "models", "feature_scalar.pckl")
        
        model = pickle.load(open(model_path, 'rb'))
        scalar = pickle.load(open(scalar_path, 'rb'))
        
        # Generate fingerprint for the molecule
        mol = Chem.MolFromSmiles(peptide_smiles)
        if mol is None:
            return None
            
        # Calculate Morgan fingerprint
        fps = AllChem.GetMorganFingerprint(mol, radius=4, useChirality=True, useCounts=True)
        
        # Convert to array format
        size = 2048  # Common fingerprint size
        arr = np.zeros((size,), np.int32)
        for idx, v in fps.GetNonzeroElements().items():
            nidx = idx % size
            arr[nidx] += int(v)
        
        # Transform using scalar
        x_test = scalar.transform([arr])
        
        # Make prediction
        predictions = model.predict_proba(x_test)
        return predictions[0, 1]  # Return probability of positive class
        
    except Exception as e:
        print(f"Error predicting permeability: {str(e)}")
        return None


if __name__ == "__main__":

    # median / low permiability: -6.2 from cycpeptmpdb
    print(predict_permeability(" 	CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)NC(=O)[C@@H](CC(C)C)NC(=O)[C@H](Cc2ccc(O)cc2)NC(=O)[C@@H]2CCCN2C(=O)[C@@H](CC(C)C)NC1=O"))

    # high -4.00
    print(predict_permeability("CC(C)C[C@H]1C(=O)N[C@@H](Cc2ccccc2)C(=O)N(C)[C@@H](C)C(=O)N[C@H](CC(C)C)C(=O)N(C)[C@H](CC(C)C)C(=O)N2CCC[C@H]2C(=O)N1C"))

    #very low -10
    print(predict_permeability("CC(C)C[C@@H]1NC(=O)CNC(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)NC(=O)CNC1=O"))