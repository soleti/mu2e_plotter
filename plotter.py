#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""@package plotter
Plotter for searchingfornues TrkAna TTree

This module produces plot from the TTree produced by the
searchingfornues framework (https://github.com/ubneutrinos/searchingfornues)

Example:
    my_plotter = plotter.Plotter(samples, weights)
    fig, ax = my_plotter.plot_variable("deent_mom",
                                       title="Reco. momentum [GeV]",
                                       cat_var="demcgen_gen",
                                       range=(95,110),
                                       bins=30)

Attributes:
    category_labels (dict): Description of event categories
    pdg_labels (dict): Labels for PDG codes
    category_colors (dict): Color scheme for event categories
    pdg_colors (dict): Colors scheme for PDG codes
"""

from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """Main plotter class

    Args:
        samples (dict): Dictionary of pandas dataframes.
        weights (dict): Dictionary of global dataframes weights.
            One for each entry in the samples dict.

    Attributes:
       samples (dict): Dictionary of pandas dataframes.
       weights (dict): Dictionary of global dataframes weights.
       pot (int): Number of protons-on-target.
    """

    genid_labels = {
        7: r"Decay-in-orbit",
        11: r"RPC (external)",
        22: r"RPC (internal)",
        38: r"Cosmic ray",
        41: r"RMC (external)",
        42: r"RMC (internal)",
        43: r"Conversion electron",
    }

    genid_colors = {
        7: "xkcd:cobalt",
        11: "xkcd:cerulean",
        22: "xkcd:sky blue",
        38: "xkcd:cyan",
        41: "xkcd:lime green",
        42: "xkcd:green",
        43: "xkcd:light red",
    }

    pdg_labels = {
        -211: r"$\pi^{+}$",
        -13: r"$\mu^{+}$",
        -11: r"$e^{+}$",
        11: r"$e^{-}$",
        13: r"$\mu^{-}$",
        22: r"$\gamma$",
        211: r"$\pi^{-}$",
        2112: r"$n$",
        2212: r"$p$"
    }

    pdg_colors = {
        -211: "#ff7f00",
        -13: "#33a02c",
        -11: "#1f78b4",
        11: "#a6cee3",
        13: "#b2df8a",
        22: "#cab2d6",
        211: "#fdbf6f",
        2112: "#137e6d",
        2212: "#e31a1c"
    }

    colors = {
        'demcgen_gen': genid_colors,
        'demcgen_pdg': pdg_colors
    }

    labels = {
        'demcgen_gen': genid_labels,
        'demcgen_pdg': pdg_labels
    }

    def __init__(self, samples, weights):
        self.samples = samples
        self.weights = weights

        for s in self.samples:
            self.samples[s] = self.samples[s].assign(
                weight=[self.weights[s]] * len(self.samples[s]))


    def _categorize(self, df, cat_var, var_name):
        var_dict = OrderedDict()
        weight_dict = OrderedDict()
        grouped = df.groupby(cat_var)
        for cat in sorted(df[cat_var].unique()):
            var_dict[cat] = grouped.get_group(cat).eval(var_name)
            weight_dict[cat] = grouped.get_group(cat)['weight']

        return var_dict, weight_dict


    def plot_variable(self,
                      variable,
                      query="deent_mom>0",
                      title="",
                      cat_var="demcgen_gen",
                      x_range=[0, 0],
                      bins=100):
        """It plots the variable from the TTree, after applying an eventual query

        Args:
            variable (str): name of the variable.
            query (str): pandas query. Default is ``selected``.
            title (str, optional): title of the plot. Default is ``variable``.
            cat_var (str, optional): Categorization of the plot. Can be ``demcgen_pdg``
                or ``demcgen_gen``
            x_range: Range of the plot on the x-axis
            bins: Number of histogram bins

        Returns:
            Figure and subplots

        """
        if not title:
            title = variable

        mc = self.samples["mc"].query(query)

        if x_range == [0, 0]:
            x_range[0] = min(mc.eval(variable))
            x_range[1] = max(mc.eval(variable))

        # pandas bug https://github.com/pandas-dev/pandas/issues/16363
        if x_range[0] >= 0 and x_range[1] >= 0:
            if query:
                query += "&"
            query += "%s <= %g & %s >= %g" % (variable,
                                              x_range[1],
                                              variable,
                                              x_range[0])

        var_dict, weight_dict = self._categorize(mc, cat_var, variable)

        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

        cat_labels = [
            "%s: %.1f" % (self.labels[cat_var][c], sum(weight_dict[c]))
            for c in sorted(mc[cat_var].unique())
        ]

        cat_colors = [self.colors[cat_var].get(c) for c in sorted(mc[cat_var].unique())]
        if not var_dict.values():
            raise ValueError("No entries selected")

        total_array = np.concatenate(list(var_dict.values()))
        total_weight = np.concatenate(list(weight_dict.values()))

        ax1.hist(var_dict.values(),
                 stacked=True,
                 edgecolor=None,
                 linewidth=0,
                 label=cat_labels,
                 color=cat_colors,
                 weights=list(weight_dict.values()),
                 range=x_range,
                 bins=bins)

        n_tot, tot_bins, patches = ax1.hist(total_array,
                                            weights=total_weight,
                                            histtype="step",
                                            edgecolor="black",
                                            range=x_range,
                                            bins=bins)

        bincenters = 0.5 * (tot_bins[1:] + tot_bins[:-1])
        mc_err = np.sqrt(n_tot * self.weights["mc"] * self.weights["mc"])

        ax1.bar(bincenters,
                n_tot,
                linewidth=0,
                edgecolor=None,
                width=0,
                yerr=mc_err)

        leg = ax1.legend(title=r'Mu2e preliminary')
        plt.setp(leg.get_title(), fontweight='bold')
        ax1.set_ylim(0., max(n_tot)*1.5)
        ax1.set_xlim(x_range[0], x_range[1])

        unit = title[title.find("[") + 1:title.find("]")
                     ] if "[" and "]" in title else ""

        x_range_size = x_range[1] - x_range[0]

        if isinstance(bins, Iterable):
            ax1.set_ylabel("N. Entries")
        else:
            ax1.set_ylabel("N. Entries / %g %s" % (x_range_size / bins, unit))

        ax1.set_xlabel(title)

        return fig, ax1