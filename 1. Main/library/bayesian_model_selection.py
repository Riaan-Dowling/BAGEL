"""
Bayesian model selection
"""

import math
import numpy as np
import pandas as pd

from scipy.stats import dirichlet, invgamma, invwishart, multivariate_normal, norm
from tqdm import tqdm


# Constant seed
np.random.seed(1)


def gibbs_sampler_2_gaussians(
    previous_proximity,
    USED_IN_2d_APPROACH,
    window_3d,
    itterations,
    burn_period,
    window_number,
    window_removed_bagel_loop_data,
    bagel_loop_data_terminal_state,
):
    """
    gibbs_sampler_2_gaussians

    Note:
    *M1: Model 1
    *M2: Model 2
    *G1: Gaussian 1
    *G2: Gaussian 2
    """
    N = len(window_3d)  # Determine total number of cells

    ran_out_of_data_split = False  # If ran out of data is true

    d = {
        "pseudo_time_normal": window_3d.pseudo_time_normal,
        "pca_1": window_3d.pca_1,
        "pca_2": window_3d.pca_2,
    }
    df = pd.DataFrame(d)

    d = {
        "pseudo_time_normal": window_3d.pseudo_time_normal,
        "pca_1": window_3d.pca_1,
        "pca_2": window_3d.pca_2,
        "cell_id_number": window_3d["cell_id_number"].values,
    }
    df_cell_id = pd.DataFrame(d)

    # Return what model
    model_1 = None
    model_2 = None
    # """
    # -----------------------------------------------------------------
    # Prior beliefs for all models
    # -----------------------------------------------------------------
    # """
    mu_0 = df.mean(axis=0)
    tau = 0.001  # #Confidence in mean
    m_degrees_of_freedom = 3  # Confidence in covariance

    # -Prior opinion of covariance

    # Addaptive statistical distance as prior

    # Statistical distance
    mean_tsne_1 = window_3d.iloc[:, 1:2].mean(axis=0)
    mean_tsne_2 = window_3d.iloc[:, 2:3].mean(axis=0)

    # Data frame of mean meseure distance to mean
    mean_data = df.iloc[:, 0:3]
    mean = mean_data.mean(axis=0)
    # mean.columns = ["pseudo_time_normal", "pca_1", "pca_2"]
    # Calculate euclidean distance as prior
    euclidean_distance_1 = (
        (mean_data.iloc[:, 0:1] - mean.iloc[0]).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_mean_1 = euclidean_distance_1.mean(axis=0)

    euclidean_distance_2 = (
        (mean_data.iloc[:, 1:2] - mean.iloc[1]).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_mean_2 = euclidean_distance_2.mean(axis=0)

    euclidean_distance_3 = (
        (mean_data.iloc[:, 2:3] - mean.iloc[2]).pow(2).sum(1).pow(0.5)
    )
    euclidean_distance_mean_3 = euclidean_distance_3.mean(axis=0)

    df_prior_opinion_cov = pd.DataFrame(
        np.array(
            [
                [euclidean_distance_mean_1, 0, 0],
                [0, euclidean_distance_mean_2, 0],
                [0, 0, euclidean_distance_mean_3],
            ]
        )
    )

    dirichlet_prior = 5  # Ensure convergence

    # """
    # -----------------------------------------------------------------
    # Prior information Model 1
    # -----------------------------------------------------------------
    # """
    # Priors already set

    # """
    # -----------------------------------------------------------------
    # Gibbs sampler output Model 1
    # -----------------------------------------------------------------
    # """

    # Sample parameters
    model_1_mu_sample_array = []
    model_1_iw_sample_array = []

    # Posterior distrebution
    model_1_log_posterior = 0

    # """
    # -----------------------------------------------------------------
    # Prior information Model 2
    # -----------------------------------------------------------------
    # """

    # Prior
    model_2_start_cov = df_prior_opinion_cov
    model_2_zeta_hat_start_1 = df.mean(axis=0)

    # Randomized start values
    # model_2_iw_sample_g1 = invwishart.rvs(df= m_degrees_of_freedom,scale = model_2_start_cov , size=1, random_state=None)
    # model_2_mu_sample_g1 = np.random.multivariate_normal(model_2_zeta_hat_start_1, model_2_iw_sample_g1/(tau), 1).T
    # model_2_mu_sample_g1 = model_2_mu_sample_g1.tolist()
    # model_2_mu_sample_g1 = [i[0] for i in model_2_mu_sample_g1]

    # model_2_iw_sample_g2 = invwishart.rvs(df= m_degrees_of_freedom,scale = model_2_start_cov , size=1, random_state=None)
    # model_2_mu_sample_g2 = np.random.multivariate_normal(model_2_zeta_hat_start_1, model_2_iw_sample_g2/(tau), 1).T
    # model_2_mu_sample_g2 = model_2_mu_sample_g2.tolist()
    # model_2_mu_sample_g2 = [i[0] for i in model_2_mu_sample_g2]

    # model_2_mu_sample_g2 = model_2_mu_sample_g1 #Model 2 G2 mu
    # model_2_iw_sample_g2 = model_2_iw_sample_g1 #Model 2 G1 var

    model_2_omega_dirichlet = np.random.dirichlet(
        (dirichlet_prior, dirichlet_prior), 1
    ).transpose()
    model_2_omega_dirichlet_samples = []

    # """
    # -----------------------------------------------------------------
    # Gibbs sampler output Model 2
    # -----------------------------------------------------------------
    # """
    # Multinomial samples sum
    y_line = np.zeros(N)
    d = {"G1": y_line, "G2": y_line}
    model_2_sum_multinomial_samples = pd.DataFrame(d)

    # Sum of all of the components belonging to Gaussian 'x'
    model_2_df_g1_sum_total = 0
    model_2_df_g2_sum_total = 0

    # Sample parameters
    model_2_mu_sample_array_g1 = []
    model_2_iw_sample_array_g1 = []

    model_2_mu_sample_array_g2 = []
    model_2_iw_sample_array_g2 = []

    # Posterior distrebution
    model_2_log_posterior = 0

    # ALl of the scale matrix values of Gibbs sampler
    model_2_scale_matrix_1_samples = []
    model_2_scale_matrix_2_samples = []

    # """
    # -----------------------------------------------------------------
    # Gibbs sampler
    # -----------------------------------------------------------------
    # """

    # Model 1 paramere calulations
    # Mean
    model_1_y_hat = df.mean(axis=0)
    model_1_zeta_hat = (np.multiply(tau, mu_0) + N * model_1_y_hat) / (tau + N)

    # -Scale matrix
    mean_minus = df - model_1_y_hat
    df_s_yy = mean_minus.cov() * N

    # the deviation between prior and estimated mean values
    step_1 = model_1_y_hat - mu_0
    step_2 = step_1.T
    zeta_calculation = (N * tau / (N + tau)) * (step_1.dot(step_2))

    # -Resulting scale matrix
    model_1_scale_matrix = (
        df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
    )

    # Random start values
    iw_sample = invwishart.rvs(
        df=m_degrees_of_freedom + N,
        scale=model_1_scale_matrix,
        size=1,
        random_state=None,
    )
    mu_sample = np.random.multivariate_normal(
        model_1_zeta_hat, iw_sample / (N + tau), 1
    ).T

    model_2_iw_sample_g1 = iw_sample
    model_2_mu_sample_g1 = mu_sample
    model_2_mu_sample_g1 = model_2_mu_sample_g1.tolist()
    model_2_mu_sample_g1 = [i[0] for i in model_2_mu_sample_g1]

    model_2_mu_sample_g2 = model_2_mu_sample_g1  # Model 2 G2 mu
    model_2_iw_sample_g2 = model_2_iw_sample_g1  # Model 2 G1 var

    print("Gibbs sampler start.")
    i = 0

    for i in tqdm(range(itterations), desc="Gibbs sampler progress"):

        # """
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # Model 1 Gaussian samples
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # """
        iw_sample = invwishart.rvs(
            df=m_degrees_of_freedom + N,
            scale=model_1_scale_matrix,
            size=1,
            random_state=None,
        )
        mu_sample = np.random.multivariate_normal(
            model_1_zeta_hat, iw_sample / (N + tau), 1
        ).T

        # """
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # Model 2 Gaussian samples
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # """

        # """
        # -----------------------------------------------------------------
        # Model 2 Multinomial samples
        # -----------------------------------------------------------------
        # """
        min_cells = 1
        model_2_min_cells_g1 = 1
        model_2_min_cells_g2 = 1

        total_searches = 0
        while (model_2_min_cells_g1 <= min_cells) or (
            model_2_min_cells_g2 <= min_cells
        ):

            # Multinomal distrebution for Variables
            model_2_observation_belongs_1 = []
            model_2_observation_belongs_2 = []

            model_2_observation_belongs_1 = model_2_omega_dirichlet[
                0
            ] * multivariate_normal.pdf(
                df, mean=model_2_mu_sample_g1, cov=model_2_iw_sample_g1
            )
            model_2_observation_belongs_1 = np.nan_to_num(model_2_observation_belongs_1)

            model_2_observation_belongs_2 = model_2_omega_dirichlet[
                1
            ] * multivariate_normal.pdf(
                df, mean=model_2_mu_sample_g2, cov=model_2_iw_sample_g2
            )
            model_2_observation_belongs_2 = np.nan_to_num(model_2_observation_belongs_2)

            model_2_observation_belong_sum = (
                model_2_observation_belongs_1 + model_2_observation_belongs_2
            )

            model_2_multinomial_samples = []
            z = 0
            for z in range(N):
                model_2_multinomial_samples.append(
                    np.random.multinomial(
                        1,
                        [
                            model_2_observation_belongs_1[z]
                            / model_2_observation_belong_sum[z],
                            model_2_observation_belongs_2[z]
                            / model_2_observation_belong_sum[z],
                        ],
                    )
                )
            model_2_multinomial_samples = pd.DataFrame(model_2_multinomial_samples)
            model_2_multinomial_samples.columns = ["G1", "G2"]

            # Assign sampled data to Guassian
            model_2_df_g1 = df[model_2_multinomial_samples["G1"] is 1]
            # model_2_df_g1 = np.nan_to_num(model_2_df_g1)

            model_2_df_g2 = df[model_2_multinomial_samples["G2"] is 1]
            # model_2_df_g1 = np.nan_to_num(model_2_df_g1)

            model_2_df_g1_sum = model_2_multinomial_samples["G1"].sum()

            model_2_df_g2_sum = model_2_multinomial_samples["G2"].sum()

            model_2_min_cells_g1 = model_2_df_g1_sum
            model_2_min_cells_g2 = model_2_df_g2_sum

            # """
            # -----------------------------------------------------------------
            # Model 2 dirichlet samples
            # -----------------------------------------------------------------
            # """
            model_2_omega_dirichlet = np.random.dirichlet(
                (
                    dirichlet_prior + model_2_df_g1_sum,
                    dirichlet_prior + model_2_df_g2_sum,
                ),
                1,
            ).transpose()

            # """
            # -----------------------------------------------------------------
            # Gibbs sampler help
            # -----------------------------------------------------------------
            # """
            if total_searches >= 1:
                if model_2_df_g1_sum > model_2_df_g2_sum:
                    model_2_df_g1 = df[model_2_multinomial_samples["G1"] is 1]
                    model_2_df_g2 = df.sample(n=2)

                    model_2_df_g1_sum = model_2_multinomial_samples["G1"].sum()
                    model_2_df_g2_sum = 2

                    model_2_min_cells_g1 = model_2_df_g1_sum
                    model_2_min_cells_g2 = model_2_df_g2_sum

                    # """
                    # -----------------------------------------------------------------
                    # Model 2 dirichlet samples
                    # -----------------------------------------------------------------
                    # """
                    model_2_omega_dirichlet = np.random.dirichlet(
                        (
                            dirichlet_prior + model_2_df_g1_sum,
                            dirichlet_prior + model_2_df_g2_sum,
                        ),
                        1,
                    ).transpose()

                else:
                    model_2_df_g1 = df.sample(n=2)
                    model_2_df_g2 = df[model_2_multinomial_samples["G2"] is 1]

                    model_2_df_g1_sum = 2
                    model_2_df_g2_sum = model_2_multinomial_samples["G2"].sum()

                    model_2_min_cells_g1 = model_2_df_g1_sum
                    model_2_min_cells_g2 = model_2_df_g2_sum

                    # """
                    # -----------------------------------------------------------------
                    # Model 2 dirichlet samples
                    # -----------------------------------------------------------------
                    # """
                    model_2_omega_dirichlet = np.random.dirichlet(
                        (
                            dirichlet_prior + model_2_df_g1_sum,
                            dirichlet_prior + model_2_df_g2_sum,
                        ),
                        1,
                    ).transpose()

            total_searches = total_searches + 1  # increment total searches

        # """
        # -----------------------------------------------------------------
        # Model 2 Gaussian 1 samples
        # -----------------------------------------------------------------
        # """
        model_2_y_hat_g1 = model_2_df_g1.mean(axis=0)

        # Mean
        model_2_zeta_hat_g1 = (
            np.multiply(tau, mu_0) + model_2_df_g1_sum * model_2_y_hat_g1
        ) / (tau + model_2_df_g1_sum)
        # Covariance

        mean_minus_1 = model_2_df_g1 - model_2_y_hat_g1
        df_s_yy = pd.DataFrame(np.dot(mean_minus_1.cov(), model_2_df_g1_sum))
        # the deviation between prior and estimated mean values
        step_1 = model_2_y_hat_g1 - mu_0
        step_2 = step_1.T
        zeta_calculation = (model_2_df_g1_sum * tau / (model_2_df_g1_sum + tau)) * (
            step_1.dot(step_2)
        )
        if math.isnan(zeta_calculation) is True:
            model_2_scale_matrix_g1 = df_prior_opinion_cov.values
            # print(df_s_yy.values)
        else:
            model_2_scale_matrix_g1 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        model_2_iw_sample_g1 = invwishart.rvs(
            df=m_degrees_of_freedom + model_2_df_g1_sum,
            scale=model_2_scale_matrix_g1,
            size=1,
            random_state=None,
        )
        model_2_mu_sample_g1 = np.random.multivariate_normal(
            model_2_zeta_hat_g1, model_2_iw_sample_g1 / (model_2_df_g1_sum + tau), 1
        ).T
        model_2_mu_sample_g1 = model_2_mu_sample_g1.tolist()
        model_2_mu_sample_g1 = [i[0] for i in model_2_mu_sample_g1]

        # """
        # -----------------------------------------------------------------
        # Model 2 Gaussian 2 samples
        # -----------------------------------------------------------------
        # """

        model_2_y_hat_g2 = model_2_df_g2.mean(axis=0)

        # Mean
        model_2_zeta_hat_g2 = (
            np.multiply(tau, mu_0) + model_2_df_g2_sum * model_2_y_hat_g2
        ) / (tau + model_2_df_g2_sum)
        # Covariance

        mean_minus_2 = model_2_df_g2 - model_2_y_hat_g2
        df_s_yy = mean_minus_2.cov() * model_2_df_g2_sum
        # the deviation between prior and estimated mean values
        step_1 = model_2_y_hat_g2 - mu_0
        step_2 = step_1.T
        zeta_calculation = (model_2_df_g2_sum * tau / (model_2_df_g2_sum + tau)) * (
            step_2.dot(step_2)
        )

        if math.isnan(zeta_calculation) is True:
            model_2_scale_matrix_g2 = df_s_yy.values + df_prior_opinion_cov.values
        else:
            model_2_scale_matrix_g2 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        model_2_iw_sample_g2 = invwishart.rvs(
            df=m_degrees_of_freedom + model_2_df_g2_sum,
            scale=model_2_scale_matrix_g2,
            size=1,
            random_state=None,
        )
        model_2_mu_sample_g2 = np.random.multivariate_normal(
            model_2_zeta_hat_g2, model_2_iw_sample_g2 / (model_2_df_g2_sum + tau), 1
        ).T
        model_2_mu_sample_g2 = model_2_mu_sample_g2.tolist()
        model_2_mu_sample_g2 = [i[0] for i in model_2_mu_sample_g2]

        # """
        # -----------------------------------------------------------------
        # Burn period samples
        # -----------------------------------------------------------------
        # """

        if i > burn_period - 1:

            # """
            # -----------------------------------------------------------------
            # Model 1
            # -----------------------------------------------------------------
            # """
            # All parameter samples array
            mu_sample_to_list = mu_sample.tolist()
            mu_sample_to_list = [i[0] for i in mu_sample_to_list]
            model_1_mu_sample_array.append(mu_sample_to_list)
            iw_sample_to_list = iw_sample.tolist()
            model_1_iw_sample_array.append(iw_sample_to_list)

            # Calculate log posterior
            model_1_post_mu_1 = multivariate_normal.pdf(
                mu_sample_to_list, mean=model_1_zeta_hat, cov=iw_sample / (N + tau)
            )
            model_1_post_cov_1 = invwishart.pdf(
                iw_sample, df=m_degrees_of_freedom + N, scale=model_1_scale_matrix
            )
            model_1_log_posterior = (
                model_1_log_posterior
                + np.log(model_1_post_mu_1)
                + np.log(model_1_post_cov_1)
            )

            # """
            # -----------------------------------------------------------------
            # Model 2
            # -----------------------------------------------------------------
            # """
            # Scale matrix
            model_2_scale_matrix_1_samples.append(model_2_scale_matrix_g1)
            model_2_scale_matrix_2_samples.append(model_2_scale_matrix_g2)
            # All parameter samples array
            model_2_mu_sample_array_g1.append(model_2_mu_sample_g1)
            model_2_iw_sample_g1_tolist = model_2_iw_sample_g1.tolist()
            model_2_iw_sample_array_g1.append(model_2_iw_sample_g1_tolist)

            model_2_mu_sample_array_g2.append(model_2_mu_sample_g2)
            model_2_iw_sample_g2_tolist = model_2_iw_sample_g2.tolist()
            model_2_iw_sample_array_g2.append(model_2_iw_sample_g2_tolist)

            model_2_omega_dirichlet_tolist = model_2_omega_dirichlet.tolist()
            model_2_omega_dirichlet_tolist = model_2_omega_dirichlet_tolist[0:]
            model_2_omega_dirichlet_tolist = [
                i[0] for i in model_2_omega_dirichlet_tolist
            ]
            model_2_omega_dirichlet_samples.append(model_2_omega_dirichlet_tolist)

            # Estimate what data point correpsonds to which gaussian
            model_2_sum_multinomial_samples = (
                model_2_sum_multinomial_samples.values + model_2_multinomial_samples
            )

            # Calculate log posterior
            model_2_post_mu_g1 = multivariate_normal.pdf(
                model_2_mu_sample_g1,
                mean=model_2_zeta_hat_g1,
                cov=model_2_iw_sample_g1 / (model_2_df_g1_sum + tau),
            )
            model_2_post_cov_g1 = invwishart.pdf(
                model_2_iw_sample_g1_tolist,
                df=m_degrees_of_freedom + model_2_df_g1_sum,
                scale=model_2_scale_matrix_g1,
            )

            model_2_post_mu_g2 = multivariate_normal.pdf(
                model_2_mu_sample_g2,
                mean=model_2_zeta_hat_g2,
                cov=model_2_iw_sample_g2 / (model_2_df_g2_sum + tau),
            )
            model_2_post_cov_g2 = invwishart.pdf(
                model_2_iw_sample_g2_tolist,
                df=m_degrees_of_freedom + model_2_df_g2_sum,
                scale=model_2_scale_matrix_g2,
            )

            model_2_post_dirchlet = dirichlet.pdf(
                model_2_omega_dirichlet_tolist,
                (
                    dirichlet_prior + model_2_df_g1_sum,
                    dirichlet_prior + model_2_df_g2_sum,
                ),
            )

            # model_2_log_posterior = model_2_log_posterior + np.log(model_2_post_mu_g1*model_2_post_cov_g1*model_2_post_mu_g2*model_2_post_cov_g2*model_2_post_dirchlet)
            model_2_log_posterior = (
                model_2_log_posterior
                + np.log(model_2_post_mu_g1)
                + np.log(model_2_post_cov_g1)
                + np.log(model_2_post_mu_g2)
                + np.log(model_2_post_cov_g2)
                + np.log(model_2_post_dirchlet)
            )

            # DF sum of components
            model_2_df_g1_sum_total = model_2_df_g1_sum_total + model_2_df_g1_sum
            model_2_df_g2_sum_total = model_2_df_g2_sum_total + model_2_df_g2_sum

        # if show_itteation is True:
        #     print('Itteration: ' + str(i))

    print("Gibbs sampler end.")

    # """
    # -----------------------------------------------------------------
    # Model 1 Maximum a posteriori
    # -----------------------------------------------------------------
    # """
    # mu
    df_model_1_mu_sample_array_1 = pd.DataFrame(
        model_1_mu_sample_array[1:],
        columns=["pseudo_time_normal", "pca_1", "pca_2"],
        dtype="float64",
    )
    model_1_map_mean = df_model_1_mu_sample_array_1.mean(axis=0)

    # Covariance
    model_1_map_cov = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period - 1):
        model_1_map_cov = model_1_map_cov + np.matrix(model_1_iw_sample_array[z])
    model_1_map_cov = model_1_map_cov / (itterations - burn_period)

    print("Model evidence:")
    # """
    # -----------------------------------------------------------------
    # Model 1 Evidence / Marginal likelihood
    # -----------------------------------------------------------------
    # """
    # print('model_1_log_evidence start')
    # Log Prior
    model_1_prior_mu = multivariate_normal.pdf(
        model_1_map_mean, mean=mu_0, cov=df_prior_opinion_cov
    )
    model_1_prior_cov = invwishart.pdf(
        model_1_map_cov, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    model_1_log_prior = np.log(model_1_prior_mu) + np.log(model_1_prior_cov)
    # model_1_log_prior = np.log(model_1_prior_mu*model_1_prior_cov)

    # Log Likelihood

    model_1_log_likelihood = 0
    for row in df.itertuples(index=False):
        qwerty = np.log(
            multivariate_normal.pdf(row, mean=model_1_map_mean, cov=model_1_map_cov)
        )
        model_1_log_likelihood = model_1_log_likelihood + qwerty

    # Log posterior
    model_1_log_posterior_divide = model_1_log_posterior / (itterations - burn_period)

    model_1_log_evidence = (
        model_1_log_likelihood + model_1_log_prior - model_1_log_posterior_divide
    )

    # print('model_1_log_evidence End')
    print("Model 1 (No bifurcation point) log evidence: " + str(model_1_log_evidence))

    # """
    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    # """

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # """
    # Model 1 Map result
    # """
    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Cr
    x, y = np.mgrid[-1:0.1:0.01, -1:1:0.01]
    pos = np.dstack((x, y))

    model_1_predict_distrebution = multivariate_normal(
        model_1_map_mean.T, model_1_map_cov
    )

    model_1_df_post = df
    # TODO test this command
    # model_1_plot_FLAG = True
    # plots.model_1_plot(x, y, model_1_predict_distrebution, pos, df, model_1_plot_FLAG)

    # """
    # -----------------------------------------------------------------
    # Model 2 Maximum a posteriori
    # -----------------------------------------------------------------
    # """
    # mu
    model_2_df_mu_samples_g1 = pd.DataFrame(
        model_2_mu_sample_array_g1[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )
    model_2_df_mu_samples_g2 = pd.DataFrame(
        model_2_mu_sample_array_g2[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )

    model_2_map_mean_g1 = model_2_df_mu_samples_g1.mean(axis=0)
    model_2_map_mean_g2 = model_2_df_mu_samples_g2.mean(axis=0)

    # Covariance
    model_2_map_cov_g1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_cov_g1 = model_2_map_cov_g1 + np.matrix(
            model_2_iw_sample_array_g1[z]
        )
    model_2_map_cov_g1 = model_2_map_cov_g1 / (itterations - burn_period)

    model_2_map_cov_g2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_cov_g2 = model_2_map_cov_g2 + np.matrix(
            model_2_iw_sample_array_g2[z]
        )
    model_2_map_cov_g2 = model_2_map_cov_g2 / (itterations - burn_period)

    # model_2_omega_dirichlet_samples
    model_2_df_omega_dirichlet_samples = pd.DataFrame(
        model_2_omega_dirichlet_samples[1:], columns=["Omega_1", "Omega_2"]
    )
    model_2_map_omega_dirichlet = model_2_df_omega_dirichlet_samples.mean(axis=0)

    # Scater matrix
    model_2_map_scale_matrix_g1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_scale_matrix_g1 = model_2_map_scale_matrix_g1 + np.matrix(
            model_2_scale_matrix_1_samples[z]
        )
    model_2_map_scale_matrix_g1 = model_2_map_scale_matrix_g1 / (
        itterations - burn_period
    )

    model_2_map_scale_matrix_g2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_scale_matrix_g2 = model_2_map_scale_matrix_g2 + np.matrix(
            model_2_scale_matrix_2_samples[z]
        )
    model_2_map_scale_matrix_g2 = model_2_map_scale_matrix_g2 / (
        itterations - burn_period
    )

    # """
    # -----------------------------------------------------------------
    # model_2_log_evidence / Marginal likelihood
    # -----------------------------------------------------------------
    # """
    # Log Prior
    model_2_prior_mu_g1 = multivariate_normal.pdf(
        model_2_map_mean_g1, mean=mu_0, cov=df_prior_opinion_cov
    )
    prior_cov_1 = invwishart.pdf(
        model_2_map_cov_g1, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    model_2_prior_mu_g2 = multivariate_normal.pdf(
        model_2_map_mean_g2, mean=mu_0, cov=df_prior_opinion_cov
    )
    model_2_prior_cov_g2 = invwishart.pdf(
        model_2_map_cov_g2, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    alpha = np.array([dirichlet_prior, dirichlet_prior])
    model_2_prior_dirchlet = dirichlet.pdf(model_2_map_omega_dirichlet, alpha)

    # model_2_log_prior = np.log(model_2_prior_mu_g1*prior_cov_1*model_2_prior_mu_g2*model_2_prior_cov_g2*model_2_prior_dirchlet)

    model_2_log_prior = (
        np.log(model_2_prior_mu_g1)
        + np.log(prior_cov_1)
        + np.log(model_2_prior_mu_g2)
        + np.log(model_2_prior_cov_g2)
        + np.log(model_2_prior_dirchlet)
    )

    # Log Likelihood

    model_2_log_likelihood = 0
    for row in df.itertuples(index=False):
        model_2_log_likelihood_data_point = np.log(
            model_2_map_omega_dirichlet["Omega_1"]
            * multivariate_normal.pdf(
                row, mean=model_2_map_mean_g1, cov=model_2_map_cov_g1
            )
            + model_2_map_omega_dirichlet["Omega_2"]
            * multivariate_normal.pdf(
                row, mean=model_2_map_mean_g2, cov=model_2_map_cov_g2
            )
        )
        model_2_log_likelihood = (
            model_2_log_likelihood + model_2_log_likelihood_data_point
        )

    # Posterior
    model_2_log_posterior_divide = model_2_log_posterior / (itterations - burn_period)

    model_2_log_evidence = (
        model_2_log_prior + model_2_log_likelihood - model_2_log_posterior_divide
    )
    print("Model 2 (Bifurcation point) log evidence: " + str(model_2_log_evidence))
    # """
    # -----------------------------------------------------------------
    # Model 2 Plots
    # -----------------------------------------------------------------
    # """

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # """
    # Model 2 Map result
    # """
    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL

    model_2_sum_multinomial_samples = model_2_sum_multinomial_samples.div(
        itterations - burn_period
    )

    model_2_df_g1_post = df[model_2_sum_multinomial_samples["G1"] > 0.5]
    model_2_df_g2_post = df[model_2_sum_multinomial_samples["G2"] > 0.5]

    # test = df[(model_2_sum_multinomial_samples['G1'] > 0.5) & (model_2_sum_multinomial_samples['G2'] > 0.5)]

    # if test.empty is  False:
    #     print(test)

    # 2 Gaussians plot
    # TODO test model_2_plot
    # plots.model_2_plot(
    #     model_2_map_mean_g1,
    #     model_2_map_cov_g1,
    #     model_2_map_mean_g2,
    #     model_2_map_cov_g2,
    #     pos,
    #     x,
    #     y,
    #     model_2_df_g1_post,
    #     model_2_df_g2_post,
    #     model_2_plot_FLAG,
    # )

    bayes_factor_12 = (
        model_1_log_evidence - model_2_log_evidence
    )  # + model 1 / - model 2

    print("Bayes Factor: " + str(bayes_factor_12))

    if bayes_factor_12 >= 0:
        model_1 = True
        model_2 = False

        # original window data
        original_window_1 = (
            window_3d  # .loc[(window_3d['tsne_1'] is model_1_df_post['tsne_1'])]
        )
        # Return data
        d = {
            "g1_pseudo_time_normal": original_window_1["pseudo_time_normal"],
            "g1_pca_1": original_window_1["pca_1"],
            "g1_pca_2": original_window_1["pca_2"],
        }
        data_output_gibbs_original = pd.DataFrame(d)

        # Return data
        d = {
            "g1_pseudo_time_normal": window_3d["pseudo_time_normal"],
            "g1_pca_1": window_3d["pca_1"],
            "g1_pca_2": window_3d["pca_2"],
        }
        data_output_gibbs_projected = pd.DataFrame(d)

    else:

        # False positive test
        # -----(-)/(-)------
        # Pseudo time
        min_pt_g1 = min(model_2_df_g1_post["pseudo_time_normal"])
        max_pt_g1 = max(model_2_df_g1_post["pseudo_time_normal"])

        min_pt_g2 = min(model_2_df_g2_post["pseudo_time_normal"])
        max_pt_g2 = max(model_2_df_g2_post["pseudo_time_normal"])

        min_pt_window = min(window_3d["pseudo_time_normal"])
        max_pt_window = max(window_3d["pseudo_time_normal"])

        # Minimum and Maximum values rows

        min_g1_row_pos = model_2_df_g1_post["pseudo_time_normal"].isin([min_pt_g1])
        min_g1_row = model_2_df_g1_post[min_g1_row_pos]

        max_g1_row_pos = model_2_df_g1_post["pseudo_time_normal"].isin([max_pt_g1])
        max_g1_row = model_2_df_g1_post[max_g1_row_pos]

        min_g2_row_pos = model_2_df_g2_post["pseudo_time_normal"].isin([min_pt_g2])
        min_g2_row = model_2_df_g2_post[min_g2_row_pos]

        max_g2_row_pos = model_2_df_g2_post["pseudo_time_normal"].isin([max_pt_g2])
        max_g2_row = model_2_df_g2_post[max_g2_row_pos]

        # percentage = (currentValue - minValue) / (maxValue - minValue);

        false_positive_flag = None
        min_contains_terminal_state = None
        # Test #1 pt
        # |----->
        # |----->
        test_pt_start = abs(
            (((min_pt_g1 - min_pt_g2)) / (max_pt_window - min_pt_window)) * 100
        )

        # Calculate at maximum pt
        #       |----->
        # |----->

        if max_pt_g1 > max_pt_g2:
            test_pt_2 = abs(
                (((min_pt_g1 - max_pt_g2)) / (max_pt_window - min_pt_window)) * 100
            )
            # |----------------->
            # |--->

            true_false = model_2_df_g2_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_g2_post.reset_index(drop=True, inplace=True)
            test_terminal_state = model_2_df_g2_post[true_false]
            test_terminal_state.reset_index(drop=True, inplace=True)
            if test_terminal_state.empty:
                min_contains_terminal_state = False
            else:
                min_contains_terminal_state = True

        else:
            test_pt_2 = abs(
                (((min_pt_g2 - max_pt_g1)) / (max_pt_window - min_pt_window)) * 100
            )

            # |----------------->
            # |--->

            true_false = model_2_df_g1_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_g1_post.reset_index(drop=True, inplace=True)
            test_terminal_state = model_2_df_g1_post[true_false]
            test_terminal_state.reset_index(drop=True, inplace=True)
            if test_terminal_state.empty:
                min_contains_terminal_state = False
            else:
                min_contains_terminal_state = True

        # if window_number >=40:
        #     import matplotlib
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter( model_2_df_g1_post.pseudo_time_normal, model_2_df_g1_post.pca_1,  model_2_df_g1_post.pca_2, c='b', marker='1', s = 20)
        #     ax.scatter( model_2_df_g2_post.pseudo_time_normal, model_2_df_g2_post.pca_1,  model_2_df_g2_post.pca_2, c='k', marker='1', s = 20)
        #     ax.scatter( bagel_loop_data_terminal_state.pseudo_time_normal, bagel_loop_data_terminal_state.pca_1,  bagel_loop_data_terminal_state.pca_2, c='r', marker='o', s = 40)
        #     ax.set_zlabel('t-SNE z')
        #     ax.set_ylabel('t-SNE y')
        #     ax.set_xlabel('Pseudo time')
        #     plt.show()

        if window_number >= 35:
            qwerty = 1

        if test_pt_start < 50:
            if (test_pt_2 < 60) and (min_contains_terminal_state is True):
                false_positive_flag = False
            elif (test_pt_2 < 60) and (min_contains_terminal_state is False):
                false_positive_flag = True
            else:
                false_positive_flag = False
        elif (test_pt_start > 50) and (min_contains_terminal_state is True):
            false_positive_flag = False
        else:
            false_positive_flag = True

        # Mean distance between Gaussians
        # Test if an adrupt jump in Gaussian means proximity to each other
        # Data frame of mean meseure distance to mean
        mean_g1 = model_2_df_g1_post.mean(axis=0)
        mean_g1.columns = ["pseudo_time_normal", "pca_1", "pca_2"]

        mean_g2 = model_2_df_g2_post.mean(axis=0)
        mean_g2.columns = ["pseudo_time_normal", "pca_1", "pca_2"]
        # Calculate euclidean distance
        euclidean_distance = np.linalg.norm(mean_g1 - mean_g2)

        proximity_ratio = (euclidean_distance / previous_proximity) * 100
        if proximity_ratio < 50:
            false_positive_flag = True
        else:
            previous_proximity = euclidean_distance

        if false_positive_flag is True:
            print("False positive")

            previous_proximity = 0.00001

            model_1 = True
            model_2 = False

            # original window data
            original_window_1 = (
                window_3d  # .loc[(window_3d['tsne_1'] is model_1_df_post['tsne_1'])]
            )
            # Return data
            d = {
                "g1_pseudo_time_normal": original_window_1["pseudo_time_normal"],
                "g1_pca_1": original_window_1["pca_1"],
                "g1_pca_2": original_window_1["pca_2"],
            }
            data_output_gibbs_original = pd.DataFrame(d)

            # Return data
            d = {
                "g1_pseudo_time_normal": window_3d["pseudo_time_normal"],
                "g1_pca_1": window_3d["pca_1"],
                "g1_pca_2": window_3d["pca_2"],
            }
            data_output_gibbs_projected = pd.DataFrame(d)

        else:

            # """
            # Test if either Gausians contains a terminal state
            # """
            true_false = model_2_df_g2_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_g2_post.reset_index(drop=True, inplace=True)
            test_terminal_state_g2 = model_2_df_g2_post[true_false]
            test_terminal_state_g2.reset_index(drop=True, inplace=True)

            true_false = model_2_df_g1_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_g1_post.reset_index(drop=True, inplace=True)
            test_terminal_state_g1 = model_2_df_g1_post[true_false]
            test_terminal_state_g1.reset_index(drop=True, inplace=True)

            true_false = window_removed_bagel_loop_data["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            window_removed_bagel_loop_data.reset_index(drop=True, inplace=True)
            test_terminal_state_window_removed_bagel_loop_data = (
                window_removed_bagel_loop_data[true_false]
            )
            test_terminal_state_window_removed_bagel_loop_data.reset_index(
                drop=True, inplace=True
            )

            if (test_terminal_state_g2.empty is False) or (
                test_terminal_state_g1.empty is False
            ):
                if test_terminal_state_window_removed_bagel_loop_data.empty is False:
                    ran_out_of_data_split = True
            else:
                ran_out_of_data_split = False

            model_1 = False
            model_2 = True
            # original window data
            # Gaussian 2
            # Link data to original pseudo time
            con1 = df_cell_id["pseudo_time_normal"].isin(
                model_2_df_g1_post["pseudo_time_normal"].values
            )
            con1 = window_3d[con1]
            con1.reset_index(drop=True, inplace=True)

            # Link data to original t-sne values
            con2 = window_3d["cell_id_number"].isin(con1["cell_id_number"].values)
            con2.reset_index(drop=True, inplace=True)
            original_window_1 = window_3d[con2]
            original_window_1.reset_index(drop=True, inplace=True)

            projected_window_1 = window_3d[con2]
            projected_window_1.reset_index(drop=True, inplace=True)
            # Gaussian 2
            # Link data to original pseudo time
            con3 = df_cell_id["pseudo_time_normal"].isin(
                model_2_df_g2_post["pseudo_time_normal"].values
            )  # (window_3d['tsne_1'].isin(model_2_df_g2_post['tsne_1'].values))
            con3 = window_3d[con3]
            con3.reset_index(drop=True, inplace=True)

            # Link data to original t-sne values
            con4 = window_3d["cell_id_number"].isin(
                con3["cell_id_number"].values
            )  # (window_3d['pseudo_time_normal'].isin(con3['pseudo_time_normal'].values))
            con4.reset_index(drop=True, inplace=True)
            original_window_2 = window_3d[con4]
            original_window_2.reset_index(drop=True, inplace=True)

            projected_window_2 = window_3d[con4]
            projected_window_2.reset_index(drop=True, inplace=True)

            # Ensure that Gaussian 1 alwas has the most data points.

            if len(original_window_1["pseudo_time_normal"]) > len(
                original_window_2["pseudo_time_normal"]
            ):

                # original window data
                d = {
                    "g1_pseudo_time_normal": original_window_1["pseudo_time_normal"],
                    "g1_pca_1": original_window_1["pca_1"],
                    "g1_pca_2": original_window_1["pca_2"],
                    "g2_pseudo_time_normal": original_window_2["pseudo_time_normal"],
                    "g2_pca_1": original_window_2["pca_1"],
                    "g2_pca_2": original_window_2["pca_2"],
                }
                data_output_gibbs_original = pd.DataFrame(d)

                # Plane projected data
                d = {
                    "g1_pseudo_time_normal": projected_window_1["pseudo_time_normal"],
                    "g1_pca_1": projected_window_1["pca_1"],
                    "g1_pca_2": projected_window_1["pca_2"],
                    "g2_pseudo_time_normal": projected_window_2["pseudo_time_normal"],
                    "g2_pca_1": projected_window_2["pca_1"],
                    "g2_pca_2": projected_window_2["pca_2"],
                }
                data_output_gibbs_projected = pd.DataFrame(d)

            else:
                # original window data
                d = {
                    "g1_pseudo_time_normal": original_window_2["pseudo_time_normal"],
                    "g1_pca_1": original_window_2["pca_1"],
                    "g1_pca_2": original_window_2["pca_2"],
                    "g2_pseudo_time_normal": original_window_1["pseudo_time_normal"],
                    "g2_pca_1": original_window_1["pca_1"],
                    "g2_pca_2": original_window_1["pca_2"],
                }
                data_output_gibbs_original = pd.DataFrame(d)

                # Plane projected data
                d = {
                    "g1_pseudo_time_normal": projected_window_2["pseudo_time_normal"],
                    "g1_pca_1": projected_window_2["pca_1"],
                    "g1_pca_2": projected_window_2["pca_2"],
                    "g2_pseudo_time_normal": projected_window_1["pseudo_time_normal"],
                    "g2_pca_1": projected_window_1["pca_1"],
                    "g2_pca_2": projected_window_1["pca_2"],
                }
                data_output_gibbs_projected = pd.DataFrame(d)

    return (
        previous_proximity,
        ran_out_of_data_split,
        model_1,
        model_2,
        data_output_gibbs_original,
        data_output_gibbs_projected,
        model_1_map_mean,
        model_1_map_cov,
        model_2_map_mean_g1,
        model_2_map_cov_g1,
        model_2_map_mean_g2,
        model_2_map_cov_g2,
    )
