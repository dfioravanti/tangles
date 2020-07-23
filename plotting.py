from src.plotting_experiments import plot_ideal_cuts_distribution, plot_quality_of_initial_cuts, \
    plot_comparison_of_initial_cuts, plot_interplay_a_and_psi, plot_interplay_cut_quality_and_psi, \
    plot_interplay_cut_quality_and_a, plot_choice_of_psi, plot_influence_of_cut_quality, plot_number_of_cuts, \
    plot_choice_of_a, plot_choice_of_cost_function, plot_psi_tree


experiments = ['two', 'unbalanced', 'three']
datasets = ['knn_gauss_blobs', 'sbm']

for name in experiments:
    for dataset in datasets:
        plot_choice_of_cost_function(dataset)

        plot_interplay_a_and_psi(dataset, name)

        plot_interplay_cut_quality_and_a(dataset, name)

        plot_interplay_cut_quality_and_psi(dataset, name)

        plot_influence_of_cut_quality(dataset, name)
        pass

# comment
plot_quality_of_initial_cuts()

plot_choice_of_a()

plot_choice_of_psi()

plot_psi_tree()

