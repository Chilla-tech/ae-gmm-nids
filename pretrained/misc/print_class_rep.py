# print classification report save with the model via sklearn's classification_report
def print_classification_report_from_dict(metrics):
    """
    metrics: dict in the form saved by sklearn's classification_report(output_dict=True)
    """
    import pandas as pd

    # label rows (only actual class labels, exclude aggregates)
    label_keys = [k for k, v in metrics.items()
                  if isinstance(v, dict) and k not in ('macro avg', 'weighted avg')]

    cols = ['precision', 'recall', 'f1-score', 'support']

    # build dataframe for the label rows
    rows = []
    for k in label_keys:
        row = metrics[k]
        rows.append([row.get(c, 0) for c in cols])
    df = pd.DataFrame(rows, index=label_keys, columns=cols)

    # format numeric columns
    for c in ['precision', 'recall', 'f1-score']:
        df[c] = df[c].astype(float).map(lambda x: f"{x:.2f}")
    df['support'] = df['support'].astype(float).map(lambda x: f"{int(x):d}")

    # widths for pretty printing
    name_w = max(max(len(str(i)) for i in df.index), 9) + 2
    col_w = 9

    # header
    header = f"{'':{name_w}}{'precision':>{col_w}}{'recall':>{col_w}}{'f1-score':>{col_w}}{'support':>{col_w}}"
    print(header)

    # label rows
    for idx, row in df.iterrows():
        print(f"{idx:{name_w}}{row['precision']:>{col_w}}{row['recall']:>{col_w}}{row['f1-score']:>{col_w}}{row['support']:>{col_w}}")

    # accuracy row (scalar)
    if 'accuracy' in metrics:
        acc = float(metrics['accuracy'])
        total_support = int(metrics.get('macro avg', {}).get('support', 0))
        print()
        print(f"{'accuracy':{name_w}}{'':{col_w}}{'':{col_w}}{acc:>{col_w}.2f}{total_support:>{col_w}d}")

    # macro avg and weighted avg (print once)
    for avg_key in ('macro avg', 'weighted avg'):
        if avg_key in metrics:
            a = metrics[avg_key]
            p, r, f, s = float(a.get('precision', 0)), float(a.get('recall', 0)), float(a.get('f1-score', 0)), int(a.get('support', 0))
            print(f"{avg_key:{name_w}}{p:>{col_w}.2f}{r:>{col_w}.2f}{f:>{col_w}.2f}{s:>{col_w}d}")