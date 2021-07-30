#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# classes for data preprocessing
from AdvancedAnalytics.ReplaceImputeEncode import DT, ReplaceImputeEncode
from AdvancedAnalytics.Regression          import logreg
from AdvancedAnalytics.Forest              import forest_classifier
from AdvancedAnalytics.Tree                import tree_classifier

# classes for logistic regression and random forest
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
# classes for model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

warnings.filterwarnings('ignore', category=UserWarning)

attribute_map = {
 'TARGET_B':                    [DT.Binary,   (0, 1)],
# Control Number was originally listed as 4. not sure if this is correct 
 'CONTROL_NUMBER':              [DT.ID,       (4., 191779)],
 'SES':                         [DT.Nominal,  (1, 2, 3, 4)],
 'URBAN_CITY':                  [DT.Nominal,  ('C', 'R', 'S', 'T')],
 'IN_HOUSE':                    [DT.Binary,   (0, 1)],
 'HOME_OWNER':                  [DT.Binary,   ('H', 'U')],
 'DONOR_GENDER':                [DT.Binary,   ('F', 'M')],
 'INCOME_GROUP':                [DT.Nominal,  (1, 2, 3, 4, 5, 6, 7)],
 'PUBLISHED_PHONE':             [DT.Binary,   (0, 1)],
 'OVERLAY_SOURCE':              [DT.Nominal,  ('B', 'M', 'P')],
 'PEP_STAR':                    [DT.Binary,   (0, 1)],
 'RECENT_STAR_STATUS':          [DT.Interval, (0, 22)],
 'RECENCY_STATUS_96NK':         [DT.Nominal,  ('A', 'E', 'F', 'L', 'N', 'S')],
 'FREQUENCY_STATUS_97NK':       [DT.Nominal,  (1, 2, 3, 4)],
 'WEALTH_RATING':               [DT.Interval, ( 0,   9)],
 'MONTHS_SINCE_ORIGIN':         [DT.Interval, ( 4, 137)],
 'DONOR_AGE':                   [DT.Interval, (0.0, 87)],
 'MOR_HIT_RATE':                [DT.Interval, (0.0, 241)],
 'MEDIAN_HOME_VALUE':           [DT.Interval, (0.0, 6000)],
 'MEDIAN_HOUSEHOLD_INCOME':     [DT.Interval, (0.0, 1500)],
 'PCT_OWNER_OCCUPIED':          [DT.Interval, (0.0, 100.0)],
 'PER_CAPITA_INCOME':           [DT.Interval, (0.0, 174523)],
 'RECENT_RESPONSE_PROP':        [DT.Interval, (0.0, 1.0)],
 'RECENT_AVG_GIFT_AMT':         [DT.Interval, (0.0, 260)],
 'RECENT_CARD_RESPONSE_PROP':   [DT.Interval, (0.0, 1.0)],
 'RECENT_AVG_CARD_GIFT_AMT':    [DT.Interval, (0.0, 300)],
 'RECENT_RESPONSE_COUNT':       [DT.Interval, (0, 16)],
 'RECENT_CARD_RESPONSE_COUNT':  [DT.Interval, (0,  9)],
 'MONTHS_SINCE_LAST_PROM_RESP': [DT.Interval, (-12, 36)],
 'FILE_CARD_GIFT':              [DT.Interval, (0, 41)],
 'LIFETIME_CARD_PROM':          [DT.Interval, (1, 56)],
 'LIFETIME_PROM':               [DT.Interval, (4, 194)],
 'LIFETIME_GIFT_AMOUNT':        [DT.Interval, (14, 3775)],
 'LIFETIME_GIFT_COUNT':         [DT.Interval, (0, 95)],
 'LIFETIME_AVG_GIFT_AMT':       [DT.Interval, (1, 450)],
 'LIFETIME_GIFT_RANGE':         [DT.Interval, (0, 997)],
 'LIFETIME_MAX_GIFT_AMT':       [DT.Interval, (4, 1000)],
 'LIFETIME_MIN_GIFT_AMT':       [DT.Interval, (0, 450)],
 'LAST_GIFT_AMT':               [DT.Interval, (0, 450)],
 'CARD_PROM_12':                [DT.Interval, (0, 17)],
 'NUMBER_PROM_12':              [DT.Interval, ( 1,  64)],
 'MONTHS_SINCE_LAST_GIFT':      [DT.Interval, ( 3,  27)],
 'MONTHS_SINCE_FIRST_GIFT':     [DT.Interval, (42, 296)]
}

# Set target, 0 = did not donate and 1 = donate
target = 'TARGET_B'
# Read in data frame
df     = pd.read_excel("donor_data.xlsx")

# Used the below to verify the data map, and found that control # lowest level
# was 11.5
#rie = ReplaceImputeEncode()
#features_map = rie.draft_data_map(df)

DataExploration = True
RIE_logreg = False
Logistic = False
RIE_tree = False
Tree = False
Forest = False

if DataExploration:
    # seaborn distribution plot of SES
    hist, ax =plt.subplots()
    ax = sns.distplot(df['SES'], kde=False)
    ax.set_title('SES')
    ax.set_xlabel('SES')
    ax.set_ylabel('Frequency')
    plt.show()
    
    # seaborn distribution plot of Income Group
    hist, ax =plt.subplots()
    ax = sns.distplot(df['INCOME_GROUP'], kde=False)
    ax.set_title('INCOME_GROUP')
    ax.set_xlabel('INCOME_GROUP')
    ax.set_ylabel('Frequency')
    plt.show()
    
    # seaborn distribution plot of Wealth Rating
    hist, ax =plt.subplots()
    ax = sns.distplot(df['WEALTH_RATING'], kde=False)
    ax.set_title('WEALTH_RATING')
    ax.set_xlabel('WEALTH_RATING')
    ax.set_ylabel('Frequency')
    plt.show()
    
    # seaborn distribution plot of Donor Age
    hist, ax =plt.subplots()
    ax = sns.distplot(df['DONOR_AGE'], kde=False)
    ax.set_title('DONOR_AGE')
    ax.set_xlabel('DONOR_AGE')
    ax.set_ylabel('Frequency')
    plt.show()
    
    # seaborn distribution plot of months since last promotion
    hist, ax =plt.subplots()
    ax = sns.distplot(df['MONTHS_SINCE_LAST_PROM_RESP'], kde=False)
    ax.set_title('MONTHS_SINCE_LAST_PROM_RESP')
    ax.set_xlabel('MONTHS_SINCE_LAST_PROM_RESP')
    ax.set_ylabel('Frequency')
    plt.show()
    
    
    
if RIE_logreg:
    print("\n***************RIE for Logistic Regression*********************")
    # Drop the last columnas of the nominal and binary predictors in order to 
    # avoid collinearity within the dataset
    rie = ReplaceImputeEncode(data_map=attribute_map, 
                              nominal_encoding='one-hot', 
                              binary_encoding ='one-hot', 
                              drop=True, display=True)
    encoded_df = rie.fit_transform(df)
    # Set variables
    y_log = encoded_df[target] 
    X_log = encoded_df.drop(target,axis=1)
    
# Running the RIE_logreg function shows us there are several missing #s, we 
# don't have anything missing from the target so there is no need to set 
# no_impute = target in the RIE function

if Logistic:
    print("\n***Cross-Validation for Regularized (L2) Logistic Regression***")
    
    # Using accurracy in the score_list didn't change overall results by much
    # decided to just use precision, recall, and f1
    score_list = ['precision', 'recall',  'f1']
    C_list     = [1e-4, 1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, np.inf]
    best_f1    = 0
    for c in C_list:
        # Default is l2 and tol default is 1e-4, didn't specify as wasn't 
        # necessary. Had to increase max_iter to 10,000
        # Attempted cv of 4 and 10, not a huge difference in output so stuck
        # with 10
        lr     = LogisticRegression(C=c, solver='lbfgs', 
                                        max_iter=10000)
        scores = cross_validate(lr, X_log, y_log, scoring=score_list, cv=10,
                                    return_train_score=False, n_jobs=-1)
        print("\nLogistic Regression for C=", c)
        print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if s=='f1' and mean>best_f1:
                best_f1   = mean
                best_c    = c
    
    # Run the logistic regression with the chosen parameter and display the
    # metrics for cross validation
    lr = LogisticRegression(C=best_c, solver='lbfgs', max_iter=10000)
    lr = lr.fit(X_log,y_log)
    print("\nLogistic Regression Best C= ", best_c)
    logreg.display_metrics(lr, X_log, y_log)
        
    # Using the most effective parameter can see how model performs in a 70/30
    # split
    print("***Logistic Regression 70/30 Validation with Chosen Parameters***")
    Xt, Xv, yt, yv = \
            train_test_split(X_log, y_log, test_size = 0.3, random_state=12345)
        
    lr = LogisticRegression(C=best_c, solver='lbfgs', max_iter=10000)
    lr = lr.fit(Xt, yt)
    logreg.display_split_metrics(lr, Xt, yt, Xv, yv)    
    
if RIE_tree:
    print('\n**************RIE for Decision Tree*****************************')
    # Set drop = False because we want the decision tree to use all variables
    # Even though they may end up being collinear. DT accounts for this.
    rie = ReplaceImputeEncode(data_map=attribute_map, 
                              nominal_encoding='one-hot', 
                          binary_encoding='one-hot', drop=False, display=True)
    encoded_df = rie.fit_transform(df)
    y_dtc = encoded_df[target] 
    X_dtc = encoded_df.drop(target,axis=1)

if Tree:
    print('\n****Choosing Decision Tree Model Parameters with 10 Fold CV****')
    score_list = ['precision', 'recall',  'f1']
    best = 0
    # 10-Fold Cross-Validation
    depths = [2, 4, 6, 7, 8, 9, 10]
    for d in depths:
        print("\nTree Depth: ", d)
        dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=2, 
                                     min_samples_split=2,random_state=12345)
        dtc = dtc.fit(X_dtc,y_dtc)
        scores = cross_validate(dtc, X_dtc, y_dtc, scoring=score_list, cv=10,
                                return_train_score=False, n_jobs=-1)
        
        print("\nDecision Tree with Best Maximum Depth=", d)
        print("{:.<18s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<18s}{:>7.4f}{:>10.4f}".format(s, mean, std))
            if s=='f1' and mean>best:
                best       = mean
                best_depth = d
    
    print("\nDecision Tree with Best Depth= ", best_depth)
    dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=2, 
                                     min_samples_split=2,random_state=12345)
    dtc = dtc.fit(X_dtc,y_dtc)
    tree_classifier.display_importance(dtc, X_dtc.columns, top=10, plot=True)
    tree_classifier.display_metrics(dtc, X_dtc, y_dtc)
    
    
    print("\n********* Best Decision Tree 70/30 Validation ************")
    Xt, Xv, yt, yv = \
        train_test_split(X_dtc,y_dtc, train_size = 0.7, random_state=12345)
    
    dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=2,   
                                 min_samples_split=2,  random_state=12345)
    dtc.fit(Xt, yt)
    tree_classifier.display_importance(dtc, Xt.columns, top=10, plot=True)
    tree_classifier.display_split_metrics(dtc, Xt, yt, Xv, yv)

if Forest:
    print('\n****Choosing Parameters for Random Forest with 4 Fold CV****')
        # Cross-Validation
    score_list      = ['precision', 'recall',  'f1']
    #n_list          = len(score_list)
    # 410 worked with all best size and such commented out
    estimators_list = [390, 395, 400, 405]
    best_d          = 0
    best_size       = 5
    best_split_size = 2*best_size
    depth_list      = [6, 7, 8, 9, 10, 25, None]
    features_list   = ['auto', None]
    best_score    = 0
    for e in estimators_list:
        for d in depth_list:
            for features in features_list:
                # Leaf size and split size is rule of thumb
                leaf_size  = round(X_dtc.shape[0]/1000)
                split_size = 2*leaf_size
                print("\nNumber of Trees: ", e, "Max_Depth: ", d,
                      "Max Features: ", features)
                rfc = RandomForestClassifier(n_estimators=e, criterion="gini",
                            min_samples_split=split_size, max_depth=d,
                            min_samples_leaf=leaf_size, max_features=features, 
                            n_jobs=-1, bootstrap=True, random_state=12345)
                scores = cross_validate(rfc, X_dtc, y_dtc, scoring=score_list, \
                                        return_train_score=False, cv=4)
                
                print("{:.<20s}{:>6s}{:>13s}".format("Metric","Mean", 
                                                     "Std. Dev."))
                for s in score_list:
                    var = "test_"+s
                    mean = scores[var].mean()
                    std  = scores[var].std()
                    print("{:.<20s}{:>7.4f}{:>10.4f}".format(s, mean, std))
                    if mean > best_score and s=='f1':
                        best_score      = mean
                        best_estimator  = e
                        best_depth      = d
                        best_features   = features
                        best_leaf_size  = leaf_size
                        best_split_size = split_size

    print("\nEvaluate Using Entire Dataset with Best Parameters")
    print("Best Number of Trees (estimators) = ", best_estimator)
    print("Best Depth = ", best_depth)
    print("Best Leaf Size = ", best_leaf_size)
    print("Best Split Size = ", best_split_size)
    print("Best Max Features = ", best_features)
    rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                            min_samples_split=best_split_size, 
                            max_depth=best_depth,
                            min_samples_leaf=best_leaf_size, 
                            max_features=best_features, 
                            n_jobs=-1, bootstrap=True, random_state=12345)
    rfc = rfc.fit(X_dtc, y_dtc)
    
    forest_classifier.display_metrics(rfc, X_dtc, y_dtc)
    forest_classifier.display_importance(rfc, X_dtc.columns, top =10, plot=True)
    
    # Evaluate the random forest with the best configuration
    print("\nEvaluating Using 70/30 Partition")
    print("Evaluating Best Random Forest")
    X_train, X_validate, y_train, y_validate = \
        train_test_split(X_dtc, y_dtc, train_size = 0.7, random_state=12345)
    print("Best Trees=", best_estimator)
    print("Best Depth=", best_depth)
    print("Best Leaf Size = ", best_leaf_size)
    print("Best Split Size = ", best_split_size)
    print("Best Max Features = ", best_features)
    rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                            min_samples_split=best_split_size, 
                            max_depth=best_depth,
                            min_samples_leaf=best_leaf_size, 
                            max_features=best_features, 
                            n_jobs=-1, bootstrap=True, random_state=12345)
    rfc= rfc.fit(X_train, y_train)
    
    forest_classifier.display_split_metrics(rfc, X_train, y_train, \
                                            X_validate, y_validate)
    forest_classifier.display_importance(rfc, X_dtc.columns, top = 10, plot=True)
