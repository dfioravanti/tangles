# Tangles

This is the code to run the program associated with the paper: "Tangles: from weak to strong clustering".

The program can be used in two ways:

1. Set the paramenters in `settings.yml` and then run `python main.py`. This is the prefered way the program is intended to be used.
2. Pass a set of parameter to `python main.py` that overrides the parameters in `settings.yml`. Paramenters that are not set default to the one specified in `settings.yml`. This is how we can run bulk experiments in parallel.

Example: `python main.py --id 1591047201 -t mindsets -p features -a 10 -s 10 -o 10 --mind_sizes 40 40 --mind_questions 40 --mind_useless 40 --mind_noise 0.1 --no_plots
`

--------

1. **Specification of dependencies**:
   We provide a `requirements.txt` file in the zip.

2. **Training / Evaluation code**:
   We provide such code in the `src` folder.

3. **Pre-trained models**:
   Not applicable, our model trains in less than a minute.

4. **Table of results with paramentes**: The majority of our experiments are comparisons among different sets of paramenter in order to investigate the properties of the new algorithm that we are introducing.  Each experiment was created using the `make_experiment.py` file.  Each experiment was averaged over 10 runs with seeds 10-100. By using `make_experiments.py` you can recreate the list of parameters for each experiment. The parameters are given in the paper near every figure.