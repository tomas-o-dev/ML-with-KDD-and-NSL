# mylib

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

from mlxtend.evaluate import bias_variance_decomp
## bias_variance_decomp() requires 
##    1. numpy ndarrays
##    2. numeric targets
from sklearn.preprocessing import LabelEncoder


def show_labels_dist(xtrn,xtst,ytrn,ytst):
# shape method gives the dimensions of the dataset
    print('features_train: {} rows, {} columns'.format(xtrn.shape[0], xtrn.shape[1]))
    print('features_test:  {} rows, {} columns'.format(xtst.shape[0], xtst.shape[1]))
    print()
    print('labels_train: {} rows, 1 column'.format(ytrn.shape[0]))
    print('labels_test:  {} rows, 1 column'.format(ytst.shape[0]))
    print()

## Here's a nice report:  
# 1. series to dataframe conversion
    my_train = pd.DataFrame(ytrn)
    my_test = pd.DataFrame(ytst)
# 2. dataframe copy with [[ -- ]]
    av_train = my_train[[my_train.columns[0]]].apply(lambda x: x.value_counts())
    av_test = my_test[[my_train.columns[0]]].apply(lambda x: x.value_counts())
# 3. add a new column
    av_train['%_train'] = round((100 * av_train / av_train.sum()),2)
    av_test['%_test'] = round((100 * av_test / av_test.sum()),2)
# 4. combine the dataframes
    av_tt = pd.concat([av_train,av_test], axis=1) 
# 5. print the report
    print('Frequency and Distribution of labels')
    print(av_tt)


def show_metrics(ytst,ygx,lbls):
    tptn_df = pd.DataFrame(confusion_matrix(ytst, ygx, labels=lbls), 
                           index=['train:{:}'.format(x) for x in lbls], 
                           columns=['pred:{:}'.format(x) for x in lbls])
    print(tptn_df)    
    print("\n~~~~")
    
    TP = np.diag(tptn_df.values)
    FP = tptn_df.values.sum(axis=0) - TP
    FN = tptn_df.values.sum(axis=1) - TP
    TN = np.sum(tptn_df.values) - (FP + FN + TP)
# false positive rates
    FPR = FP/(FP+TN)
# false negative rates
    FNR = FN/(TP+FN)
# overall 
    sfpr=FP.sum()/(FP.sum()+TN.sum())
    sfnr=FN.sum()/(TP.sum()+FN.sum())
    
    if len(lbls) >2:
        for x in range(len(lbls)):
            print('{:>12} : '.format(lbls[x]),
                  'FPR = %.3f   FNR = %.3f' % (FPR[x], FNR[x]))
        print()

    print('{:>12} : '.format('macro avg'),
          'FPR = %.3f   FNR = %.3f'  % (FPR.mean(), FNR.mean()))
    print('weighted avg :  FPR = %.3f   FNR = %.3f' % (sfpr, sfnr))
 
    print("\n~~~~")
    
#    macro average: unweighted mean per label 
# weighted average: support-weighted mean per label  
    print(classification_report(ytst, ygx, digits=3, target_names=lbls))

    print("~~~~")
# Matthews correlation coefficient: 
#   correlation between prediction and ground truth
#   (+1 perfect, 0 random prediction, -1 inverse)

    mcc = matthews_corrcoef(ytst, ygx)
    print('MCC: Overall :  %.3f' % mcc)
    if len(lbls) >2:
        for tc in lbls:
            bin_mcc = matthews_corrcoef(ytst == tc, ygx == tc)
            print('{:>12} :'.format(tc),' %.3f' % bin_mcc)  

    return '~~~~'


def bias_var_metrics(xtrn,xtst,ytrn,ytst,clf,folds=200):
# slow because it does num_rounds (default=200) bootstrap cross validation

# numeric targets
    ytrain = LabelEncoder().fit_transform(ytrn)
    ytest = LabelEncoder().fit_transform(ytst)

    avg_loss, avg_bias, avg_var = bias_variance_decomp(
        clf, xtrn.values, ytrain, xtst.values, ytest, 
        loss='0-1_loss', num_rounds=folds, random_seed=44)

    print('   Average bias: %.3f' % avg_bias)
    print('   Average variance: %.3f' % avg_var)
    print('   Average expected loss: %.3f  "Goodness": %.3f' % (avg_loss, (1-avg_loss)))
    print()