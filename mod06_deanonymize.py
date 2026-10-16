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
    unique_aux = aux_df.drop_duplicates(subset=identifiers, keep=False)
    matches = anon_df[['anon_id'] + identifiers].merge(
        unique_aux[['name'] + identifiers],
        on=identifiers,
        how='inner',
        validate='many_to_one'
    )

    result = matches[['anon_id','name']].copy()
    result.columns = ['anon_id','matched_name']

    return result.drop_duplicates().reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return matches_df['anon_id'].nunique() / anon_df['anon_id'].nunique() if 'anon_id' else len(anon_df)
