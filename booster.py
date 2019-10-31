#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package booster
Package to train boosted trees to isolate conversion electrons for the Mu2e experiment.

Takes as input TrkAna TTrees converted into pandas dataframes.
"""

from operator import itemgetter

import pandas as pd
import xgboost as xgb
import shap

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, average_precision_score


labels = [
    "dio",
    "cr"
]


titles = [
    r"Decay-in-orbit",
    r"Cosmic ray"
    # r"RMC",
    # r"RPC",
]

bkg_queries = [
    "demcgen_gen == 7",
    "demcgen_gen == 38"
    # "(demcgen_gen == 41) | (demcgen_gen == 42)",
    # "(demcgen_gen == 11) | (demcgen_gen == 22)"
]

variables = [
    "is_signal",
    "de_t0",
    "deent_td",
    "deent_d0",
    "deent_om",
    "dequal_TrkQual",
    "dequal_TrkPID",
    "ue_status",
    "crvinfo__timeWindowStart_best",
    "is_triggered"
]


class Booster:
    """Main Booster class

    Args:
        samples (dict): Dictionary of pandas dataframes.
        training_vars (list): List of variables used for training.
        random_state: seed for splitting sample. Default is 0.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       random_state: seed for splitting sample.
       variables (list): List of variables used for training.
       params (dict): XGBoost parameters.
    """

    def __init__(self, samples, training_vars, random_state=0):
        self.samples = samples
        self.random_state = random_state
        self.variables = training_vars

        eta = 0.1
        max_depth = 10
        subsample = 1
        colsample_bytree = 1
        min_child_weight = 1
        self.params = {
            "objective": "binary:logistic",
            "booster": "gbtree",
            "eval_metric": "auc",
            "eta": eta,
            "tree_method": 'exact',
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "silent": 1,
            "min_child_weight": min_child_weight,
            "seed": random_state,
            #"num_class" : 22,
        }
        print(
            'XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'
            .format(eta, max_depth, subsample, colsample_bytree))

    def _run_single(self, train, test, features, target, ax, title=''):
        num_boost_round = 1000
        early_stopping_rounds = 50

        y_train = train[target]
        y_valid = test[target]
        dtrain = xgb.DMatrix(train[features], y_train)
        dvalid = xgb.DMatrix(test[features], y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(
            self.params,
            dtrain,
            num_boost_round,
            evals=watchlist,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False)

        gbm.save_model("pickles/%s.model" % title.replace(" ", "_"))
        gbm.dump_model("pickles/%s.txt" % title.replace(" ", "_"))

        print("Validating...")
        check = gbm.predict(
            xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration + 1)

        #area under the precision-recall curve
        score = average_precision_score(test[target].values, check)
        print('area under the precision-recall curve: {:.6f}'.format(score))

        check2 = check.round()
        score = precision_score(test[target].values, check2)
        print('precision score: {:.6f}'.format(score))

        score = recall_score(test[target].values, check2)
        print('recall score: {:.6f}'.format(score))

        imp = self.get_importance(gbm, features)
        # print('Importance array: ', imp)

        ############################################ ROC Curve

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(test[target].values, check)
        roc_auc = auc(fpr, tpr)
        # xgb.plot_importance(gbm)
        # explainer = shap.TreeExplainer(gbm)
        # shap_values = explainer.shap_values(train[features])
        # shap.force_plot(explainer.expected_value, shap_values, train[features])
        # shap.summary_plot(shap_values, train[features], max_display=5)

        ax.plot(fpr, tpr, lw=2, label='%s (area = %g)' % (title, roc_auc))
        # ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_ylabel("Background rejection")
        ax.set_xlabel("1 - Signal efficiency")
        return gbm, imp, gbm.best_iteration + 1

    def get_importance(self, gbm, features):
        self.create_feature_map(features)
        importance = gbm.get_fscore(fmap='pickles/xgb.fmap')
        importance = sorted(importance.items(),
                            key=itemgetter(1), reverse=True)
        return importance

    def train_booster(self, ax, bkg_query=""):

        plt_title = 'Global'

        if bkg_query in bkg_queries:
            print("Training %s..." % titles[bkg_queries.index(bkg_query)])
            plt_title = r"%s background" % titles[bkg_queries.index(bkg_query)]
            bkg_query = "(%s | is_signal == 1)" % bkg_query

        train_dio = self.samples["dio"][0].query("deent_mom > 95 & " + bkg_query)[self.variables]
        test_dio = self.samples["dio"][1].query("deent_mom > 95 & " + bkg_query)[self.variables]

        train_ce = self.samples["ce"][0].query("deent_mom > 95 & " + bkg_query)[self.variables]
        test_ce = self.samples["ce"][1].query("deent_mom > 95 & " + bkg_query)[self.variables]

        train = pd.concat([train_dio, train_ce])
        test = pd.concat([test_dio, test_ce])

        features = list(train.columns.values)
        features.remove('is_signal')
        # features.remove('shr_energy_tot_cali')
        # features.remove('trk_energy_tot')

        preds, imp, num_boost_rounds = self._run_single(
            train,
            test,
            features,
            'is_signal',
            ax,
            title=plt_title)

        return preds

    @staticmethod
    def get_features(train):
        trainval = list(train.columns.values)
        output = trainval
        return sorted(output)

    @staticmethod
    def create_feature_map(features):
        outfile = open('pickles/xgb.fmap', 'w')
        for i, feat in enumerate(features):
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        outfile.close()
