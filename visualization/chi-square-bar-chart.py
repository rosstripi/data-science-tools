import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rc("font", size=14)


def chi_square(data_df: pd.DataFrame, output_file=None):
    counts_df = data_df.groupby('DO')['fraudulent'].value_counts()
    print(counts_df.head())
    categories = ('DO', 'MD')
    y_pos = np.arange(len(categories))
    do_fraud_k = counts_df[1][1] / (counts_df[1][1] + counts_df[1][0]) * 1000
    md_fraud_k = counts_df[0][1] / (counts_df[0][1] + counts_df[0][0]) * 1000
    fraud_per_k = [do_fraud_k, md_fraud_k]
    
    plt.bar(y_pos, fraud_per_k, align='center', alpha=0.5)
    plt.axhline(y=data_df['fraudulent'].value_counts()[1]/data_df.shape[0] * 1000,
                color='r', linestyle='--')
    plt.xticks(y_pos, categories)
    plt.ylabel('Fraudulent doctors/1k')
    plt.title('Fraudulent doctors by License Type')

    if output_file is None:
        plt.show()
    else:
        # TODO: test that file exists
        plt.savefig(fname=output_file, dpi=600, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures