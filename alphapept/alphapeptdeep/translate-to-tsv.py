from peptdeep.protein.fasta import PredictSpecLibFasta
import os, psutil
import numpy as np

from peptdeep.spec_lib.translate import translate_to_tsv

alphapeptdeep_hdf = r'D:\optimising-plasma-proteomics\alphapept\alphapeptdeep\output\output.speclib.hdf'
top_k_frag = 12

frag_inten = 0.001


min_frag_mz = 200
max_frag_mz = 1800
min_frag_nAA = 0

output_diann_tsv = (
    f"{alphapeptdeep_hdf[:-len('.speclib.hdf')]}_frags={top_k_frag}.speclib.tsv"
)

print(output_diann_tsv)



fasta_lib = PredictSpecLibFasta(
    None,
    decoy=None
)

fasta_lib.load_hdf(alphapeptdeep_hdf, load_mod_seq=True)

if 'id' in fasta_lib.protein_df.columns:
    fasta_lib.protein_df.rename(columns={'id':'protein_id'}, inplace=True)

fasta_lib.append_protein_name()


if 'decoy' in fasta_lib.precursor_df.columns:
    fasta_lib._precursor_df = fasta_lib.precursor_df[fasta_lib._precursor_df.decoy == 0]

if __name__ == "__main__":
    translate_to_tsv(
        fasta_lib,
        output_diann_tsv,
        keep_k_highest_fragments=top_k_frag,
        min_frag_nAA=min_frag_nAA,
        min_frag_mz=min_frag_mz,
        min_frag_intensity=frag_inten,
        batch_size=100000,
        translate_mod_dict=None,
        multiprocessing=False,
    )