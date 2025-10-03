import pandas as pd

dft_results = pd.read_csv("results/StructPropBench/dft_statistics.csv")
input_info_bandgap = pd.read_csv("src/struct_prop_bench/bandgap_nonmetal.csv")
input_info_bulkmodulus = pd.read_csv("src/struct_prop_bench/bulkmodulus_nonmetal.csv")

# delete '.cif' suffix
input_info_bandgap['input_cif_name'] = input_info_bandgap['input_cif_name'].apply(lambda x: x[:-4] if x.endswith('.cif') else x)
input_info_bulkmodulus['input_cif_name'] = input_info_bulkmodulus['input_cif_name'].apply(lambda x: x[:-4] if x.endswith('.cif') else x[:-4])


# get the 'trend' column from input_info and merge it into dft_results when input_cif_name matches structure_name in dft_results
dft_results = dft_results.merge(input_info_bandgap[['input_cif_name', 'trend']], left_on='structure_name', right_on='input_cif_name', how='left')
dft_results = dft_results.merge(input_info_bulkmodulus[['input_cif_name', 'trend']], left_on='structure_name', right_on='input_cif_name', how='left', suffixes=('_bandgap', '_bulkmodulus'))

# delete the redundant 'input_cif_name' columns
dft_results = dft_results.drop(columns=['input_cif_name_bandgap', 'input_cif_name_bulkmodulus'])

# generate a new dataframe for the results, with columns: structure_name, bandgap_[model_name1], bandgap_[model_name2], ..., bulkmodulus_[model_name1], bulkmodulus_[model_name2], ...
# where the bandgap and bulkmodulus columns are filled with right or wrong based on the trend column, the dft_results, and the bandgap_[model_namex]'s value
results = pd.DataFrame()
results['structure_name'] = dft_results['structure_name']

def evaluate_property(row, property_name, model_name):
    trend = row[f'trend_{property_name}']
    if pd.isna(trend):
        return 'N/A'
    
    orig_value = row[f'{property_name}_orig']
    fina_value = row[f'{property_name}_{model_name}']
    if pd.isna(orig_value):
        return 'N/A'
    
    # the fina_value can be float, "LLM fail", "DFT fail", "Structure same" etc.
    # keep the string values as it is
    # first try to convert fina_value to float
    try:
        fina_value = float(fina_value)
    except:
        return fina_value
    # if isinstance(fina_value, str):
    #     return fina_value
    if trend == 'increase': 
        return 'true' if fina_value > orig_value else 'false'
    elif trend == 'decrease':
        return 'true' if fina_value < orig_value else 'false'
    else:
        return 'N/A'
    
# automatically get all model names from dft_results columns
model_names = set()
for col in dft_results.columns:
    if col.startswith('bandgap_'):
        model_names.add(col[len('bandgap_'):])
    elif col.startswith('bulkmodulus_'):
        model_names.add(col[len('bulkmodulus_'):])
model_names = list(model_names)
# remove 'orig' from model_names if exists
if 'orig' in model_names:
    model_names.remove('orig')

for model_name in model_names:
    results[f'bandgap_{model_name}'] = dft_results.apply(lambda row: evaluate_property(row, 'bandgap', model_name), axis=1)
    results[f'bulkmodulus_{model_name}'] = dft_results.apply(lambda row: evaluate_property(row, 'bulkmodulus', model_name), axis=1)

# save results to csv
results.to_csv("results/StructPropBench/structprop_evaluation_results.csv", index=False)

# compute the true rate for each model and each property
summary = pd.DataFrame(columns=['model_name', 'property', 'true_count', 'total_count', 'true_rate'])
for model_name in model_names:
    for property_name in ['bandgap', 'bulkmodulus']:
        true_count = (results[f'{property_name}_{model_name}'] == 'true').sum()
        total_count = (results[f'{property_name}_{model_name}'] != 'N/A').sum()
        true_rate = true_count / total_count if total_count > 0 else 0
        summary = summary._append({
            'model_name': model_name,
            'property': property_name,
            'true_count': true_count,
            'total_count': total_count,
            'true_rate': true_rate
        }, ignore_index=True)

# save summary to csv
summary.to_csv("results/StructPropBench/structprop_evaluation_summary.csv", index=False)