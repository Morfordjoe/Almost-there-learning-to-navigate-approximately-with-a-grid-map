# Almost-there-learning-to-navigate-approximately-with-a-grid-map

WORKFLOW

gradients_script.m
    Models defined and training data, test data and model predictions data outputted
    Requires functions files: function_allcomb.m, function_mod_fig.m, function_standardise_range.m,
    Outputs files: All_model_predictions.csv, Testing_data.csv, Training_data.csv
    Outputs various figures (outputted into Write up folder)

Neural_networks_2023_categoric_output.py
Neural_networks_2023_continuous_output.py
    Both scripts train and test neural networks with the data from the Testing_data.csv and Training_data.csv
    Both scripts require functions file: NN_functions_2023.py
    Output csvs of NN testing performance into NN_continuous_2023_out and NN_categ_2023_out folders

NN_error_analysis_stats.R
    Requires functions file: NN_error_analysis_stats_functions.R
    Analyses NN performance data with respect to the navigational models
    Uses the output of the python scripts and the matlab script
    Produces graphs and involves stats tests

ABSTRACT

Grid map navigation, in which animals judge their location using intersecting environmental gradients, has been proposed to account for impressive navigational abilities across various taxa. However, the precise mechanisms by which animals navigate using environmental gradients are obscure: first, how do animals extrapolate the spatial distribution of gradients, and second, how do they combine spatial information from multiple gradients? Various models of the extrapolation and combination of spatial gradients have been proposed, but the ontogeny of these mechanisms is little considered. Animals might be predisposed to utilise particular navigational strategies, with these fixed through development; alternatively, mechanisms might arise and change through learning. To investigate this, we trained artificial neural networks, as simple computational learning models, to navigate in virtual bicoordinate grid environments, and tested their outputs against previously proposed models. We found neural networks initially adopted ‘the approximate model’: determining their displacement in each gradient independently and summing these to approximate goalward directions. This supports the suggestion that this model represents a relatively simple mechanism to adopt in complex environments. However, by the end of training, neural networks no longer conformed to the model predictions, hence adopting this mechanism for a limited period only. Thus, the predictions of these models might be met only in certain developmental stages as animals learn. Conversely, the neural networks extrapolated gradients differently depending on the environment. These results facilitate more nuanced predictions of how animal navigation might develop through learning. These predictions should be tested as large tracking datasets of animal movements accumulate.

