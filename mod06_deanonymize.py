import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux

def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """

    identifiers = ['age', 'zip3', 'gender']

    anon_id_col = 'anon_id'
    name_col = 'name'

    # Drop rows with missing quasi-identifiers (NaN-to-NaN matches are meaningless)
    anon = anon_df.dropna(subset=identifiers)
    aux = aux_df.dropna(subset=identifiers)

    # Keep only aux rows whose identifier combination appears exactly once
    unique_aux = aux.drop_duplicates(subset=identifiers, keep=False)

    # Select only the columns we need so overlapping columns (e.g. 'key')
    # don't get suffixed and cause KeyErrors
    matches = anon[[anon_id_col] + identifiers].merge(
        unique_aux[[name_col] + identifiers],
        on=identifiers,
        how='inner',
        validate='many_to_one'
    )

    result = matches[[anon_id_col, name_col]].copy()
    result.columns = ['anon_id', 'matched_name']

    return result.drop_duplicates().reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
     # Use whichever ID column the anonymized data actually has
    if 'anon_id' in anon_df.columns:
        denom_col = 'anon_id'
    elif 'key' in anon_df.columns:
        denom_col = 'key'
    else:
        denom_col = None

    total = anon_df[denom_col].nunique() if denom_col else len(anon_df)
    if total == 0:
        return 0.0

    return matches_df['anon_id'].nunique() / total
