# Mu2e TrkAna plotter

This class allows you to produce plots of the variable stored in TrkAna TTrees.
The instantion requires a dictionary with a pandas dataframe. You can convert a ROOT TTree into a pandas dataframe using [uproot](https://github.com/scikit-hep/uproot):

```python
import uproot
file = uproot.open("trkana-mc.root")
trkananeg = file["TrkAnaNeg"]["trkana"]

df = trkananeg.pandas.df(flatname=default_flatname, flatten=False)
```

The plotter class is then instantied as:
```python
samples = {'mc': df}
weights = {'mc': 1}

my_plotter = plotter.Plotter(samples, weights)
```

The main method is `plot_variable` which can manipulate different variables and apply queries to the dataframe. It also possible to categorize the events according to the PDG code (`demcgen_pdg`) or the GenID code (`demcgen_gen`).

This example shows how to plot the reconstructed momentum categorized by GenID code:
```python
fig, ax = my_plotter.plot_variable("deent_mom",
                                   title="Reco. momentum [GeV]",
                                   cat_var="demcgen_gen",
                                   x_range=(95,110),
                                   bins=30)
ax.set_yscale('log')
ax.set_ylim(bottom=0.5, top=5e5)
```

More information can be found in `Example.ipynb`.
