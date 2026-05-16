import os
import sys

import re
import argparse
from collections import defaultdict
import statistics

from tqdm import tqdm
import pandas as pd
from pyteomics.mass import mass

def get_argparser():
    parser = argparse.ArgumentParser(description="Making consensus library")
    parser.add_argument("--datadir",
                        type=str,
                        help="Directory contains data files",
                        required=True
                        )
    parser.add_argument("--out_path",
                        type=str,
                        help="File path of the output library",
                        required=True
                        )
    return parser

def get_fragment_seq(mod_pept_seq, ion_type, length):
    res = ''
    k = length
    if ion_type == 'b':
        i = 0
        while k > 0:
            if (mod_pept_seq[i] >= 'A') and (mod_pept_seq[i] <= 'Z'):
                k -= 1
            i += 1
        return mod_pept_seq[:i]
    elif ion_type == 'y':
        i = -1
        while k > 0:
            if (mod_pept_seq[i] >= 'A') and (mod_pept_seq[i] <= 'Z'):
                k -= 1
            i -= 1
        while mod_pept_seq[i] >= 'a' and mod_pept_seq[i] <= 'z':
            i -= 1
        return mod_pept_seq[i+1:]

def normalize_intensities(intensities):
    """
    Args:
        intensities: List of intensities.

    Returns:
        list - A list of relative intensities.
    """
    max_intensity = max(intensities)
    if max_intensity > 0:
        return [val / max_intensity for val in intensities]
    else:
        return [0.0] * len(intensities)

def process_fragment_and_intensities_strings(fragments_string, intensities_string):
    """
    Process fragment types and intensities strings.

    Args:
        fragments_string - a string contains fragment types separated by ';'.
        intensities_string - a string contains intensities separated by ';'.
    Returns:
        tuple - a list of fragment types and a list of corresponding intensities
    """
    fragment_types = fragments_string.split(";")
    intensities = intensities_string.split(";")

    cleaned_fragment_types = []
    cleaned_intensities = []

    for i, fragment_type in enumerate(fragment_types):
        # Filter out fragment types containing '-', 'a', 'c', 'x', or 'z'
        # "-" is contained in neutral loss fragment (like -H2O -NH3)
        if not any(char in fragment_type for char in ['-', 'a', 'c', 'x', 'z']):
            if fragment_type.count("+") == 0: # If don't have charge at the end of str
                cleaned_fragment_types.append(fragment_type + ("^1"))
            else:
                def repl(matchobj):
                    return "^" + matchobj.group(1)
                pattern = r"\(([0-9]+)\+\)"
                cleaned_fragment_types.append(re.sub(pattern, repl, fragment_type)) # e.g, y7(2+) -> y7^2
            # Convert intensity to numeric type as elements of [intensities] are still strings
            cleaned_intensities.append(float(intensities[i]))
    return cleaned_fragment_types, cleaned_intensities

if __name__ == "__main__":
    arg_parser = get_argparser()
    args = arg_parser.parse_args()

    data_dir = args.datadir
    lookup_tsv = pd.read_csv(os.path.join(data_dir, "lookup.tsv"), sep='\t')
    precursors_tsv = ...
    header_row = pd.DataFrame(columns=["PrecursorMz",
                                       "ProductMz", 
                                       "ProteinId",
                                       "GeneName",
                                       "Annotation", 
                                       "PeptideSequence", 
                                       "ModifiedPeptideSequence", 
                                       "PrecursorCharge", 
                                       "LibraryIntensity", 
                                       "RetentionTime",
                                       "PrecursorIonMobility",
                                       "FragmentType", 
                                       "FragmentCharge", 
                                       "FragmentSeriesNumber",
                                       "FragmentLossType"
                                       ])
    fileout_path = args.out_path
    if os.path.exists(fileout_path):
        accept_remove = input("Output file already exists. Remove it? (Y/n): ")
        accept_remove = str(accept_remove).lower()
        while accept_remove != 'y' and accept_remove != 'n':
            accept_remove = input("Output file already exists. Remove it? (Y/n): ")
            accept_remove = str(accept_remove).lower()
        if str(accept_remove).lower() == 'n':
            sys.exit(0)
        elif str(accept_remove).lower() == 'y':
            os.remove(fileout_path)

    header_row.to_csv(fileout_path, sep='\t', index=False, header=True, mode='a')

    db = mass.Unimod()
    aa_mass = dict(mass.std_aa_mass)
    aa_mass['c'] = db.by_id(4)['mono_mass']
    aa_mass['o'] = db.by_id(35)['mono_mass']

    current_ref_file = None
    for row in tqdm(lookup_tsv.itertuples(index=False), miniters=1000):
        precursor = row[0]
        ref_file = row[1]
        if ref_file != current_ref_file:
            current_ref_file = ref_file
            precursors_tsv = pd.read_csv(os.path.join(data_dir, current_ref_file), sep='\t')
            precursors_tsv = precursors_tsv[["Sequence", "Precursor", "Matches", "Intensities", "id"]]
        start_id = row[2]
        n_spec = row[3]
        end_id = start_id + n_spec
        idx = (precursors_tsv['id'] >= start_id) & (precursors_tsv['id'] < end_id)
        query = precursors_tsv.loc[idx]

        fragments_dict = defaultdict(list)
        for precur_row in query.itertuples(index=False):
            fragment_types, intensities = process_fragment_and_intensities_strings(precur_row[2], precur_row[3])
            normalized_intensities = normalize_intensities(intensities)
            for annotation, intensity in zip(fragment_types, normalized_intensities):
                fragments_dict[annotation].append(intensity)

        consensus_dict = {}
        for k,v in fragments_dict.items():
            consensus_dict[k] = statistics.median(v)

        fragment_types = consensus_dict.keys()
        consensus_intensities = normalize_intensities(list(consensus_dict.values()))

        seq = query.iloc[0]["Sequence"]
        modified_peptide_seq = precursor[:-1]
        transformed_pept_seq = modified_peptide_seq.replace("C(UniMod:4)", "cC")
        transformed_pept_seq = transformed_pept_seq.replace("M(UniMod:35)", "oM")
        precursor_mz = mass.fast_mass2(transformed_pept_seq, aa_mass=aa_mass, ion_type="M", charge=int(precursor[-1]))
        
        rows = []
        for annotation, intensity in zip(fragment_types, consensus_intensities):
                annot_split = annotation.split("^")
                product_mz = mass.fast_mass2(
                    get_fragment_seq(transformed_pept_seq, annot_split[0][0], int(annot_split[0][1:])),
                    aa_mass=aa_mass,
                    ion_type=annot_split[0][0],
                    charge=int(annot_split[1])
                    )
                rows.append([precursor_mz,
                             product_mz,
                             None, #ProteinId
                             None, #GeneName
                             annotation, 
                             seq, 
                             precursor[:-1], # remove charge
                             precursor[-1],
                             intensity,
                             0, #RT
                             0, #IM
                             annot_split[0], # e.g, y7^1 -> y7
                             annot_split[1], # 1, product charge
                             annot_split[0][1:], # 7, fragment series length
                             ""
                             ])
        pd.DataFrame(rows).to_csv(fileout_path, sep='\t', index=False, header=False, mode='a')