There are two notebooks in this folder
- `pypi_end-to-end.ipynb`
- `LSTM.ipynb`

The `pypi_end-to-end.ipynb` runs the following steps:
- process the raw event log and generate triplet and split files;
- generate models from the triplet;
- run cynet prediction with models and split files;
- generate csv files for running machine learning algorithms;
- run machine learning prediction with the csvs;
- generate simulation files and plot snapshot.

After we run `pypi_end-to-end.ipynb` upto the point when triplet and split files are generated, we can run the `LSTM.ipynb` to get RNN prediction.
