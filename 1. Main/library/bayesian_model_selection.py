"""
Bayesian model selection
"""

import numpy as np
import pandas as pd

from scipy.stats import dirichlet, invgamma, invwishart, multivariate_normal, norm

import math
import joblib
from tqdm import tqdm


# Constant seed
np.random.seed(1)


"""
Note:
*M1: Model 1
*M2: Model 2
*G1: Gaussian 1
*G2: Gaussian 2
"""


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
    """
    N = len(window_3d)  # Determine total number of cells

    ran_out_of_data_split = False  # If ran out of data is true

    pseudo_time = window_3d.pseudo_time_normal
    tsne_1 = window_3d.pca_1
    tsne_2 = window_3d.pca_2

    d = {"pseudo_time_normal": pseudo_time, "pca_1": tsne_1, "pca_2": tsne_2}
    df = pd.DataFrame(d)

    d = {
        "pseudo_time_normal": pseudo_time,
        "pca_1": tsne_1,
        "pca_2": tsne_2,
        "cell_id_number": window_3d["cell_id_number"].values,
    }
    df_cell_ID = pd.DataFrame(d)

    # Return what model
    Model_1 = None
    Model_2 = None
    """
    -----------------------------------------------------------------
    Prior beliefs for all models
    -----------------------------------------------------------------
    """
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
    mean.columns = ["pseudo_time_normal", "pca_1", "pca_2"]
    # Calculate euclidean distance as prior
    euclidean_distance_1 = (mean_data.iloc[:, 0:1] - mean[0]).pow(2).sum(1).pow(0.5)
    euclidean_distance_mean_1 = euclidean_distance_1.mean(axis=0)

    euclidean_distance_2 = (mean_data.iloc[:, 1:2] - mean[1]).pow(2).sum(1).pow(0.5)
    euclidean_distance_mean_2 = euclidean_distance_2.mean(axis=0)

    euclidean_distance_3 = (mean_data.iloc[:, 2:3] - mean[2]).pow(2).sum(1).pow(0.5)
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

    Dirichlet_prior = 5  # Ensure convergence

    """
    -----------------------------------------------------------------
    Prior information Model 1
    -----------------------------------------------------------------
    """
    # Priors already set

    """
    -----------------------------------------------------------------
    Gibbs sampler output Model 1
    -----------------------------------------------------------------
    """

    # Sample parameters
    model_1_mu_sample_array = []
    model_1_iw_sample_array = []

    # Posterior distrebution
    model_1_log_posterior = 0

    """
    -----------------------------------------------------------------
    Prior information Model 2
    -----------------------------------------------------------------
    """

    # Prior
    model_2_start_cov = df_prior_opinion_cov
    model_2_zeta_hat_start_1 = df.mean(axis=0)

    # Randomized start values
    # model_2_iw_sample_G1 = invwishart.rvs(df= m_degrees_of_freedom,scale = model_2_start_cov , size=1, random_state=None)
    # model_2_mu_sample_G1 = np.random.multivariate_normal(model_2_zeta_hat_start_1, model_2_iw_sample_G1/(tau), 1).T
    # model_2_mu_sample_G1 = model_2_mu_sample_G1.tolist()
    # model_2_mu_sample_G1 = [i[0] for i in model_2_mu_sample_G1]

    # model_2_iw_sample_G2 = invwishart.rvs(df= m_degrees_of_freedom,scale = model_2_start_cov , size=1, random_state=None)
    # model_2_mu_sample_G2 = np.random.multivariate_normal(model_2_zeta_hat_start_1, model_2_iw_sample_G2/(tau), 1).T
    # model_2_mu_sample_G2 = model_2_mu_sample_G2.tolist()
    # model_2_mu_sample_G2 = [i[0] for i in model_2_mu_sample_G2]

    # model_2_mu_sample_G2 = model_2_mu_sample_G1 #Model 2 G2 mu
    # model_2_iw_sample_G2 = model_2_iw_sample_G1 #Model 2 G1 var

    model_2_Omega_Dirichlet = np.random.dirichlet(
        (Dirichlet_prior, Dirichlet_prior), 1
    ).transpose()
    model_2_Omega_Dirichlet_samples = []

    """
    -----------------------------------------------------------------
    Gibbs sampler output Model 2
    -----------------------------------------------------------------
    """
    # Multinomial samples sum
    y_line = np.zeros(N)
    d = {"G1": y_line, "G2": y_line}
    model_2_sum_multinomial_samples = pd.DataFrame(d)

    # Sum of all of the components belonging to Gaussian 'x'
    model_2_df_G1_sum_total = 0
    model_2_df_G2_sum_total = 0

    # Sample parameters
    model_2_mu_sample_array_G1 = []
    model_2_iw_sample_array_G1 = []

    model_2_mu_sample_array_G2 = []
    model_2_iw_sample_array_G2 = []

    # Posterior distrebution
    model_2_log_posterior = 0

    # ALl of the scale matrix values of Gibbs sampler
    model_2_scale_matrix_1_samples = []
    model_2_scale_matrix_2_samples = []

    """
    -----------------------------------------------------------------
    Gibbs sampler
    -----------------------------------------------------------------
    """

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

    model_2_iw_sample_G1 = iw_sample
    model_2_mu_sample_G1 = mu_sample
    model_2_mu_sample_G1 = model_2_mu_sample_G1.tolist()
    model_2_mu_sample_G1 = [i[0] for i in model_2_mu_sample_G1]

    model_2_mu_sample_G2 = model_2_mu_sample_G1  # Model 2 G2 mu
    model_2_iw_sample_G2 = model_2_iw_sample_G1  # Model 2 G1 var

    print("Gibbs sampler start.")
    i = 0

    for i in tqdm(range(itterations), desc="Gibbs sampler progress"):

        """
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Model 1 Gaussian samples
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        """
        iw_sample = invwishart.rvs(
            df=m_degrees_of_freedom + N,
            scale=model_1_scale_matrix,
            size=1,
            random_state=None,
        )
        mu_sample = np.random.multivariate_normal(
            model_1_zeta_hat, iw_sample / (N + tau), 1
        ).T

        """
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        Model 2 Gaussian samples
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        """

        """
        -----------------------------------------------------------------
        Model 2 Multinomial samples
        -----------------------------------------------------------------
        """
        min_cells = 1
        model_2_min_cells_G1 = 1
        model_2_min_cells_G2 = 1

        total_searches = 0
        while (model_2_min_cells_G1 <= min_cells) or (
            model_2_min_cells_G2 <= min_cells
        ):

            # Multinomal distrebution for Variables
            model_2_observation_belongs_1 = []
            model_2_observation_belongs_2 = []

            model_2_observation_belongs_1 = model_2_Omega_Dirichlet[
                0
            ] * multivariate_normal.pdf(
                df, mean=model_2_mu_sample_G1, cov=model_2_iw_sample_G1
            )
            model_2_observation_belongs_1 = np.nan_to_num(model_2_observation_belongs_1)

            model_2_observation_belongs_2 = model_2_Omega_Dirichlet[
                1
            ] * multivariate_normal.pdf(
                df, mean=model_2_mu_sample_G2, cov=model_2_iw_sample_G2
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
            model_2_df_G1 = df[model_2_multinomial_samples["G1"] == 1]
            # model_2_df_G1 = np.nan_to_num(model_2_df_G1)

            model_2_df_G2 = df[model_2_multinomial_samples["G2"] == 1]
            # model_2_df_G1 = np.nan_to_num(model_2_df_G1)

            model_2_df_G1_sum = model_2_multinomial_samples["G1"].sum()

            model_2_df_G2_sum = model_2_multinomial_samples["G2"].sum()

            model_2_min_cells_G1 = model_2_df_G1_sum
            model_2_min_cells_G2 = model_2_df_G2_sum

            """
            -----------------------------------------------------------------
            Model 2 Dirichlet samples
            -----------------------------------------------------------------
            """
            model_2_Omega_Dirichlet = np.random.dirichlet(
                (
                    Dirichlet_prior + model_2_df_G1_sum,
                    Dirichlet_prior + model_2_df_G2_sum,
                ),
                1,
            ).transpose()

            """
            -----------------------------------------------------------------
            Gibbs sampler help
            -----------------------------------------------------------------
            """
            if total_searches >= 1:
                if model_2_df_G1_sum > model_2_df_G2_sum:
                    model_2_df_G1 = df[model_2_multinomial_samples["G1"] == 1]
                    model_2_df_G2 = df.sample(n=2)

                    model_2_df_G1_sum = model_2_multinomial_samples["G1"].sum()
                    model_2_df_G2_sum = 2

                    model_2_min_cells_G1 = model_2_df_G1_sum
                    model_2_min_cells_G2 = model_2_df_G2_sum

                    """
                    -----------------------------------------------------------------
                    Model 2 Dirichlet samples
                    -----------------------------------------------------------------
                    """
                    model_2_Omega_Dirichlet = np.random.dirichlet(
                        (
                            Dirichlet_prior + model_2_df_G1_sum,
                            Dirichlet_prior + model_2_df_G2_sum,
                        ),
                        1,
                    ).transpose()

                else:
                    model_2_df_G1 = df.sample(n=2)
                    model_2_df_G2 = df[model_2_multinomial_samples["G2"] == 1]

                    model_2_df_G1_sum = 2
                    model_2_df_G2_sum = model_2_multinomial_samples["G2"].sum()

                    model_2_min_cells_G1 = model_2_df_G1_sum
                    model_2_min_cells_G2 = model_2_df_G2_sum

                    """
                    -----------------------------------------------------------------
                    Model 2 Dirichlet samples
                    -----------------------------------------------------------------
                    """
                    model_2_Omega_Dirichlet = np.random.dirichlet(
                        (
                            Dirichlet_prior + model_2_df_G1_sum,
                            Dirichlet_prior + model_2_df_G2_sum,
                        ),
                        1,
                    ).transpose()

            total_searches = total_searches + 1  # increment total searches

        """
        -----------------------------------------------------------------
        Model 2 Gaussian 1 samples
        -----------------------------------------------------------------
        """
        model_2_y_hat_G1 = model_2_df_G1.mean(axis=0)

        # Mean
        model_2_zeta_hat_G1 = (
            np.multiply(tau, mu_0) + model_2_df_G1_sum * model_2_y_hat_G1
        ) / (tau + model_2_df_G1_sum)
        # Covariance

        mean_minus_1 = model_2_df_G1 - model_2_y_hat_G1
        df_s_yy = pd.DataFrame(np.dot(mean_minus_1.cov(), model_2_df_G1_sum))
        # the deviation between prior and estimated mean values
        step_1 = model_2_y_hat_G1 - mu_0
        step_2 = step_1.T
        zeta_calculation = (model_2_df_G1_sum * tau / (model_2_df_G1_sum + tau)) * (
            step_1.dot(step_2)
        )
        if math.isnan(zeta_calculation) == True:
            model_2_scale_matrix_G1 = df_prior_opinion_cov.values
            # print(df_s_yy.values)
        else:
            model_2_scale_matrix_G1 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        model_2_iw_sample_G1 = invwishart.rvs(
            df=m_degrees_of_freedom + model_2_df_G1_sum,
            scale=model_2_scale_matrix_G1,
            size=1,
            random_state=None,
        )
        model_2_mu_sample_G1 = np.random.multivariate_normal(
            model_2_zeta_hat_G1, model_2_iw_sample_G1 / (model_2_df_G1_sum + tau), 1
        ).T
        model_2_mu_sample_G1 = model_2_mu_sample_G1.tolist()
        model_2_mu_sample_G1 = [i[0] for i in model_2_mu_sample_G1]

        """
        -----------------------------------------------------------------
        Model 2 Gaussian 2 samples
        -----------------------------------------------------------------
        """

        model_2_y_hat_G2 = model_2_df_G2.mean(axis=0)

        # Mean
        model_2_zeta_hat_G2 = (
            np.multiply(tau, mu_0) + model_2_df_G2_sum * model_2_y_hat_G2
        ) / (tau + model_2_df_G2_sum)
        # Covariance

        mean_minus_2 = model_2_df_G2 - model_2_y_hat_G2
        df_s_yy = mean_minus_2.cov() * model_2_df_G2_sum
        # the deviation between prior and estimated mean values
        step_1 = model_2_y_hat_G2 - mu_0
        step_2 = step_1.T
        zeta_calculation = (model_2_df_G2_sum * tau / (model_2_df_G2_sum + tau)) * (
            step_2.dot(step_2)
        )

        if math.isnan(zeta_calculation) == True:
            model_2_scale_matrix_G2 = df_s_yy.values + df_prior_opinion_cov.values
        else:
            model_2_scale_matrix_G2 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        model_2_iw_sample_G2 = invwishart.rvs(
            df=m_degrees_of_freedom + model_2_df_G2_sum,
            scale=model_2_scale_matrix_G2,
            size=1,
            random_state=None,
        )
        model_2_mu_sample_G2 = np.random.multivariate_normal(
            model_2_zeta_hat_G2, model_2_iw_sample_G2 / (model_2_df_G2_sum + tau), 1
        ).T
        model_2_mu_sample_G2 = model_2_mu_sample_G2.tolist()
        model_2_mu_sample_G2 = [i[0] for i in model_2_mu_sample_G2]

        """
        -----------------------------------------------------------------
        Burn period samples
        -----------------------------------------------------------------
        """

        if i > burn_period - 1:

            """
            -----------------------------------------------------------------
            Model 1
            -----------------------------------------------------------------
            """
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

            """
            -----------------------------------------------------------------
            Model 2
            -----------------------------------------------------------------
            """
            # Scale matrix
            model_2_scale_matrix_1_samples.append(model_2_scale_matrix_G1)
            model_2_scale_matrix_2_samples.append(model_2_scale_matrix_G2)
            # All parameter samples array
            model_2_mu_sample_array_G1.append(model_2_mu_sample_G1)
            model_2_iw_sample_G1_tolist = model_2_iw_sample_G1.tolist()
            model_2_iw_sample_array_G1.append(model_2_iw_sample_G1_tolist)

            model_2_mu_sample_array_G2.append(model_2_mu_sample_G2)
            model_2_iw_sample_G2_tolist = model_2_iw_sample_G2.tolist()
            model_2_iw_sample_array_G2.append(model_2_iw_sample_G2_tolist)

            model_2_Omega_Dirichlet_tolist = model_2_Omega_Dirichlet.tolist()
            model_2_Omega_Dirichlet_tolist = model_2_Omega_Dirichlet_tolist[0:]
            model_2_Omega_Dirichlet_tolist = [
                i[0] for i in model_2_Omega_Dirichlet_tolist
            ]
            model_2_Omega_Dirichlet_samples.append(model_2_Omega_Dirichlet_tolist)

            # Estimate what data point correpsonds to which gaussian
            model_2_sum_multinomial_samples = (
                model_2_sum_multinomial_samples.values + model_2_multinomial_samples
            )

            # Calculate log posterior
            model_2_post_mu_G1 = multivariate_normal.pdf(
                model_2_mu_sample_G1,
                mean=model_2_zeta_hat_G1,
                cov=model_2_iw_sample_G1 / (model_2_df_G1_sum + tau),
            )
            model_2_post_cov_G1 = invwishart.pdf(
                model_2_iw_sample_G1_tolist,
                df=m_degrees_of_freedom + model_2_df_G1_sum,
                scale=model_2_scale_matrix_G1,
            )

            model_2_post_mu_G2 = multivariate_normal.pdf(
                model_2_mu_sample_G2,
                mean=model_2_zeta_hat_G2,
                cov=model_2_iw_sample_G2 / (model_2_df_G2_sum + tau),
            )
            model_2_post_cov_G2 = invwishart.pdf(
                model_2_iw_sample_G2_tolist,
                df=m_degrees_of_freedom + model_2_df_G2_sum,
                scale=model_2_scale_matrix_G2,
            )

            model_2_post_dirchlet = dirichlet.pdf(
                model_2_Omega_Dirichlet_tolist,
                (
                    Dirichlet_prior + model_2_df_G1_sum,
                    Dirichlet_prior + model_2_df_G2_sum,
                ),
            )

            # model_2_log_posterior = model_2_log_posterior + np.log(model_2_post_mu_G1*model_2_post_cov_G1*model_2_post_mu_G2*model_2_post_cov_G2*model_2_post_dirchlet)
            model_2_log_posterior = (
                model_2_log_posterior
                + np.log(model_2_post_mu_G1)
                + np.log(model_2_post_cov_G1)
                + np.log(model_2_post_mu_G2)
                + np.log(model_2_post_cov_G2)
                + np.log(model_2_post_dirchlet)
            )

            # DF sum of components
            model_2_df_G1_sum_total = model_2_df_G1_sum_total + model_2_df_G1_sum
            model_2_df_G2_sum_total = model_2_df_G2_sum_total + model_2_df_G2_sum

        # if show_itteation == True:
        #     print('Itteration: ' + str(i))

    print("Gibbs sampler end.")

    """
    -----------------------------------------------------------------
    Model 1 Maximum a posteriori
    -----------------------------------------------------------------
    """
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
    """
    -----------------------------------------------------------------
    Model 1 Evidence / Marginal likelihood
    -----------------------------------------------------------------
    """
    # print('model_1_log_Evidence start')
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

    model_1_log_Evidence = (
        model_1_log_likelihood + model_1_log_prior - model_1_log_posterior_divide
    )

    # print('model_1_log_Evidence End')
    print("Model 1 (No bifurcation point) log evidence: " + str(model_1_log_Evidence))

    """
    -----------------------------------------------------------------
    Plots
    -----------------------------------------------------------------
    """

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    """
    Model 1 Map result
    """
    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    # Cr
    x, y = np.mgrid[-1:0.1:0.01, -1:1:0.01]
    pos = np.dstack((x, y))

    model_1_predict_distrebution = multivariate_normal(
        model_1_map_mean.T, model_1_map_cov
    )

    model_1_df_post = df
    # TODO test this command
    # Model_1_plot_FLAG = True
    # plots.Model_1_plot(x, y, model_1_predict_distrebution, pos, df, Model_1_plot_FLAG)

    """
    -----------------------------------------------------------------
    Model 2 Maximum a posteriori
    -----------------------------------------------------------------
    """
    # mu
    model_2_df_mu_samples_G1 = pd.DataFrame(
        model_2_mu_sample_array_G1[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )
    model_2_df_mu_samples_G2 = pd.DataFrame(
        model_2_mu_sample_array_G2[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )

    model_2_map_mean_G1 = model_2_df_mu_samples_G1.mean(axis=0)
    model_2_map_mean_G2 = model_2_df_mu_samples_G2.mean(axis=0)

    # Covariance
    model_2_map_cov_G1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_cov_G1 = model_2_map_cov_G1 + np.matrix(
            model_2_iw_sample_array_G1[z]
        )
    model_2_map_cov_G1 = model_2_map_cov_G1 / (itterations - burn_period)

    model_2_map_cov_G2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_cov_G2 = model_2_map_cov_G2 + np.matrix(
            model_2_iw_sample_array_G2[z]
        )
    model_2_map_cov_G2 = model_2_map_cov_G2 / (itterations - burn_period)

    # model_2_Omega_Dirichlet_samples
    model_2_df_Omega_Dirichlet_samples = pd.DataFrame(
        model_2_Omega_Dirichlet_samples[1:], columns=["Omega_1", "Omega_2"]
    )
    model_2_map_Omega_Dirichlet = model_2_df_Omega_Dirichlet_samples.mean(axis=0)

    # Scater matrix
    model_2_map_scale_matrix_G1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_scale_matrix_G1 = model_2_map_scale_matrix_G1 + np.matrix(
            model_2_scale_matrix_1_samples[z]
        )
    model_2_map_scale_matrix_G1 = model_2_map_scale_matrix_G1 / (
        itterations - burn_period
    )

    model_2_map_scale_matrix_G2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        model_2_map_scale_matrix_G2 = model_2_map_scale_matrix_G2 + np.matrix(
            model_2_scale_matrix_2_samples[z]
        )
    model_2_map_scale_matrix_G2 = model_2_map_scale_matrix_G2 / (
        itterations - burn_period
    )

    """
    -----------------------------------------------------------------
    model_2_log_Evidence / Marginal likelihood
    -----------------------------------------------------------------
    """
    # Log Prior
    model_2_prior_mu_G1 = multivariate_normal.pdf(
        model_2_map_mean_G1, mean=mu_0, cov=df_prior_opinion_cov
    )
    prior_cov_1 = invwishart.pdf(
        model_2_map_cov_G1, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    model_2_prior_mu_G2 = multivariate_normal.pdf(
        model_2_map_mean_G2, mean=mu_0, cov=df_prior_opinion_cov
    )
    model_2_prior_cov_G2 = invwishart.pdf(
        model_2_map_cov_G2, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    alpha = np.array([Dirichlet_prior, Dirichlet_prior])
    model_2_prior_dirchlet = dirichlet.pdf(model_2_map_Omega_Dirichlet, alpha)

    # model_2_log_prior = np.log(model_2_prior_mu_G1*prior_cov_1*model_2_prior_mu_G2*model_2_prior_cov_G2*model_2_prior_dirchlet)

    model_2_log_prior = (
        np.log(model_2_prior_mu_G1)
        + np.log(prior_cov_1)
        + np.log(model_2_prior_mu_G2)
        + np.log(model_2_prior_cov_G2)
        + np.log(model_2_prior_dirchlet)
    )

    # Log Likelihood

    model_2_log_likelihood = 0
    for row in df.itertuples(index=False):
        model_2_log_likelihood_data_point = np.log(
            model_2_map_Omega_Dirichlet["Omega_1"]
            * multivariate_normal.pdf(
                row, mean=model_2_map_mean_G1, cov=model_2_map_cov_G1
            )
            + model_2_map_Omega_Dirichlet["Omega_2"]
            * multivariate_normal.pdf(
                row, mean=model_2_map_mean_G2, cov=model_2_map_cov_G2
            )
        )
        model_2_log_likelihood = (
            model_2_log_likelihood + model_2_log_likelihood_data_point
        )

    # Posterior
    model_2_log_posterior_divide = model_2_log_posterior / (itterations - burn_period)

    model_2_log_Evidence = (
        model_2_log_prior + model_2_log_likelihood - model_2_log_posterior_divide
    )
    print("Model 2 (Bifurcation point) log evidence: " + str(model_2_log_Evidence))
    """
    -----------------------------------------------------------------
    Model 2 Plots
    -----------------------------------------------------------------
    """

    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL
    """
    Model 2 Map result
    """
    # LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL

    model_2_sum_multinomial_samples = model_2_sum_multinomial_samples.div(
        itterations - burn_period
    )

    model_2_df_G1_post = df[model_2_sum_multinomial_samples["G1"] > 0.5]
    model_2_df_G2_post = df[model_2_sum_multinomial_samples["G2"] > 0.5]

    # test = df[(model_2_sum_multinomial_samples['G1'] > 0.5) & (model_2_sum_multinomial_samples['G2'] > 0.5)]

    # if test.empty ==  False:
    #     print(test)

    # 2 Gaussians plot
    # TODO test Model_2_plot
    # plots.Model_2_plot(
    #     model_2_map_mean_G1,
    #     model_2_map_cov_G1,
    #     model_2_map_mean_G2,
    #     model_2_map_cov_G2,
    #     pos,
    #     x,
    #     y,
    #     model_2_df_G1_post,
    #     model_2_df_G2_post,
    #     Model_2_plot_FLAG,
    # )

    bayes_factor_12 = (
        model_1_log_Evidence - model_2_log_Evidence
    )  # + model 1 / - model 2

    print("Bayes Factor: " + str(bayes_factor_12))

    if bayes_factor_12 >= 0:
        Model_1 = True
        Model_2 = False

        # ORIGINAL window data
        ORIGINAL_window_1 = (
            window_3d  # .loc[(window_3d['tsne_1'] == model_1_df_post['tsne_1'])]
        )
        # Return data
        d = {
            "g1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
            "g1_pca_1": ORIGINAL_window_1["pca_1"],
            "g1_pca_2": ORIGINAL_window_1["pca_2"],
        }
        data_output_gibbs_ORIGNAL = pd.DataFrame(d)

        # Return data
        d = {
            "g1_pseudo_time_normal": window_3d["pseudo_time_normal"],
            "g1_pca_1": window_3d["pca_1"],
            "g1_pca_2": window_3d["pca_2"],
        }
        data_output_gibbs_PROJECTED = pd.DataFrame(d)

    else:

        # False positive test
        # -----(-)/(-)------
        # Pseudo time
        min_PT_G1 = min(model_2_df_G1_post["pseudo_time_normal"])
        max_PT_G1 = max(model_2_df_G1_post["pseudo_time_normal"])

        min_PT_G2 = min(model_2_df_G2_post["pseudo_time_normal"])
        max_PT_G2 = max(model_2_df_G2_post["pseudo_time_normal"])

        min_PT_window = min(window_3d["pseudo_time_normal"])
        max_PT_window = max(window_3d["pseudo_time_normal"])

        # Minimum and Maximum values rows

        min_G1_row_pos = model_2_df_G1_post["pseudo_time_normal"].isin([min_PT_G1])
        min_G1_row = model_2_df_G1_post[min_G1_row_pos]

        max_G1_row_pos = model_2_df_G1_post["pseudo_time_normal"].isin([max_PT_G1])
        max_G1_row = model_2_df_G1_post[max_G1_row_pos]

        min_G2_row_pos = model_2_df_G2_post["pseudo_time_normal"].isin([min_PT_G2])
        min_G2_row = model_2_df_G2_post[min_G2_row_pos]

        max_G2_row_pos = model_2_df_G2_post["pseudo_time_normal"].isin([max_PT_G2])
        max_G2_row = model_2_df_G2_post[max_G2_row_pos]

        # percentage = (currentValue - minValue) / (maxValue - minValue);

        false_positive_FLAG = None
        min_contains_TERMINAL_STATE = None
        # Test #1 PT
        # |----->
        # |----->
        test_PT_START = abs(
            (((min_PT_G1 - min_PT_G2)) / (max_PT_window - min_PT_window)) * 100
        )

        # Calculate at maximum pt
        #       |----->
        # |----->

        if max_PT_G1 > max_PT_G2:
            test_PT_2 = abs(
                (((min_PT_G1 - max_PT_G2)) / (max_PT_window - min_PT_window)) * 100
            )
            # |----------------->
            # |--->

            true_false = model_2_df_G2_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_G2_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE = model_2_df_G2_post[true_false]
            test_TERMINAL_STATE.reset_index(drop=True, inplace=True)
            if test_TERMINAL_STATE.empty:
                min_contains_TERMINAL_STATE = False
            else:
                min_contains_TERMINAL_STATE = True

        else:
            test_PT_2 = abs(
                (((min_PT_G2 - max_PT_G1)) / (max_PT_window - min_PT_window)) * 100
            )

            # |----------------->
            # |--->

            true_false = model_2_df_G1_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_G1_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE = model_2_df_G1_post[true_false]
            test_TERMINAL_STATE.reset_index(drop=True, inplace=True)
            if test_TERMINAL_STATE.empty:
                min_contains_TERMINAL_STATE = False
            else:
                min_contains_TERMINAL_STATE = True

        # if window_number >=40:
        #     import matplotlib
        #     import matplotlib.pyplot as plt
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter( model_2_df_G1_post.pseudo_time_normal, model_2_df_G1_post.pca_1,  model_2_df_G1_post.pca_2, c='b', marker='1', s = 20)
        #     ax.scatter( model_2_df_G2_post.pseudo_time_normal, model_2_df_G2_post.pca_1,  model_2_df_G2_post.pca_2, c='k', marker='1', s = 20)
        #     ax.scatter( bagel_loop_data_terminal_state.pseudo_time_normal, bagel_loop_data_terminal_state.pca_1,  bagel_loop_data_terminal_state.pca_2, c='r', marker='o', s = 40)
        #     ax.set_zlabel('t-SNE z')
        #     ax.set_ylabel('t-SNE y')
        #     ax.set_xlabel('Pseudo time')
        #     plt.show()

        if window_number >= 35:
            qwerty = 1

        if test_PT_START < 50:
            if (test_PT_2 < 60) and (min_contains_TERMINAL_STATE == True):
                false_positive_FLAG = False
            elif (test_PT_2 < 60) and (min_contains_TERMINAL_STATE == False):
                false_positive_FLAG = True
            else:
                false_positive_FLAG = False
        elif (test_PT_START > 50) and (min_contains_TERMINAL_STATE == True):
            false_positive_FLAG = False
        else:
            false_positive_FLAG = True

        # Mean distance between Gaussians
        # Test if an adrupt jump in Gaussian means proximity to each other
        # Data frame of mean meseure distance to mean
        mean_G1 = model_2_df_G1_post.mean(axis=0)
        mean_G1.columns = ["pseudo_time_normal", "pca_1", "pca_2"]

        mean_G2 = model_2_df_G2_post.mean(axis=0)
        mean_G2.columns = ["pseudo_time_normal", "pca_1", "pca_2"]
        # Calculate euclidean distance
        euclidean_distance = np.linalg.norm(mean_G1 - mean_G2)

        proximity_ratio = (euclidean_distance / previous_proximity) * 100
        if proximity_ratio < 50:
            false_positive_FLAG = True
        else:
            previous_proximity = euclidean_distance

        if false_positive_FLAG == True:
            print("False positive")

            previous_proximity = 0.00001

            Model_1 = True
            Model_2 = False

            # ORIGINAL window data
            ORIGINAL_window_1 = (
                window_3d  # .loc[(window_3d['tsne_1'] == model_1_df_post['tsne_1'])]
            )
            # Return data
            d = {
                "g1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                "g1_pca_1": ORIGINAL_window_1["pca_1"],
                "g1_pca_2": ORIGINAL_window_1["pca_2"],
            }
            data_output_gibbs_ORIGNAL = pd.DataFrame(d)

            # Return data
            d = {
                "g1_pseudo_time_normal": window_3d["pseudo_time_normal"],
                "g1_pca_1": window_3d["pca_1"],
                "g1_pca_2": window_3d["pca_2"],
            }
            data_output_gibbs_PROJECTED = pd.DataFrame(d)

        else:

            """
            Test if either Gausians contains a terminal state
            """
            true_false = model_2_df_G2_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_G2_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_G2 = model_2_df_G2_post[true_false]
            test_TERMINAL_STATE_G2.reset_index(drop=True, inplace=True)

            true_false = model_2_df_G1_post["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            model_2_df_G1_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_G1 = model_2_df_G1_post[true_false]
            test_TERMINAL_STATE_G1.reset_index(drop=True, inplace=True)

            true_false = window_removed_bagel_loop_data["pseudo_time_normal"].isin(
                bagel_loop_data_terminal_state["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            window_removed_bagel_loop_data.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_window_removed_bagel_loop_data = (
                window_removed_bagel_loop_data[true_false]
            )
            test_TERMINAL_STATE_window_removed_bagel_loop_data.reset_index(
                drop=True, inplace=True
            )

            if (test_TERMINAL_STATE_G2.empty == False) or (
                test_TERMINAL_STATE_G1.empty == False
            ):
                if test_TERMINAL_STATE_window_removed_bagel_loop_data.empty == False:
                    ran_out_of_data_split = True
            else:
                ran_out_of_data_split = False

            Model_1 = False
            Model_2 = True
            # ORIGINAL window data
            # Gaussian 2
            # Link data to ORIGINAL pseudo time
            con1 = df_cell_ID["pseudo_time_normal"].isin(
                model_2_df_G1_post["pseudo_time_normal"].values
            )
            con1 = window_3d[con1]
            con1.reset_index(drop=True, inplace=True)

            # Link data to ORIGINAL t-sne values
            con2 = window_3d["cell_id_number"].isin(con1["cell_id_number"].values)
            con2.reset_index(drop=True, inplace=True)
            ORIGINAL_window_1 = window_3d[con2]
            ORIGINAL_window_1.reset_index(drop=True, inplace=True)

            PROJECTED_window_1 = window_3d[con2]
            PROJECTED_window_1.reset_index(drop=True, inplace=True)
            # Gaussian 2
            # Link data to ORIGINAL pseudo time
            con3 = df_cell_ID["pseudo_time_normal"].isin(
                model_2_df_G2_post["pseudo_time_normal"].values
            )  # (window_3d['tsne_1'].isin(model_2_df_G2_post['tsne_1'].values))
            con3 = window_3d[con3]
            con3.reset_index(drop=True, inplace=True)

            # Link data to ORIGINAL t-sne values
            con4 = window_3d["cell_id_number"].isin(
                con3["cell_id_number"].values
            )  # (window_3d['pseudo_time_normal'].isin(con3['pseudo_time_normal'].values))
            con4.reset_index(drop=True, inplace=True)
            ORIGINAL_window_2 = window_3d[con4]
            ORIGINAL_window_2.reset_index(drop=True, inplace=True)

            PROJECTED_window_2 = window_3d[con4]
            PROJECTED_window_2.reset_index(drop=True, inplace=True)

            # Ensure that Gaussian 1 alwas has the most data points.

            if len(ORIGINAL_window_1["pseudo_time_normal"]) > len(
                ORIGINAL_window_2["pseudo_time_normal"]
            ):

                # Orignal window data
                d = {
                    "g1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                    "g1_pca_1": ORIGINAL_window_1["pca_1"],
                    "g1_pca_2": ORIGINAL_window_1["pca_2"],
                    "g2_pseudo_time_normal": ORIGINAL_window_2["pseudo_time_normal"],
                    "g2_pca_1": ORIGINAL_window_2["pca_1"],
                    "g2_pca_2": ORIGINAL_window_2["pca_2"],
                }
                data_output_gibbs_ORIGNAL = pd.DataFrame(d)

                # Plane PROJECTED data
                d = {
                    "g1_pseudo_time_normal": PROJECTED_window_1["pseudo_time_normal"],
                    "g1_pca_1": PROJECTED_window_1["pca_1"],
                    "g1_pca_2": PROJECTED_window_1["pca_2"],
                    "g2_pseudo_time_normal": PROJECTED_window_2["pseudo_time_normal"],
                    "g2_pca_1": PROJECTED_window_2["pca_1"],
                    "g2_pca_2": PROJECTED_window_2["pca_2"],
                }
                data_output_gibbs_PROJECTED = pd.DataFrame(d)

            else:
                # Orignal window data
                d = {
                    "g1_pseudo_time_normal": ORIGINAL_window_2["pseudo_time_normal"],
                    "g1_pca_1": ORIGINAL_window_2["pca_1"],
                    "g1_pca_2": ORIGINAL_window_2["pca_2"],
                    "g2_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                    "g2_pca_1": ORIGINAL_window_1["pca_1"],
                    "g2_pca_2": ORIGINAL_window_1["pca_2"],
                }
                data_output_gibbs_ORIGNAL = pd.DataFrame(d)

                # Plane PROJECTED data
                d = {
                    "g1_pseudo_time_normal": PROJECTED_window_2["pseudo_time_normal"],
                    "g1_pca_1": PROJECTED_window_2["pca_1"],
                    "g1_pca_2": PROJECTED_window_2["pca_2"],
                    "g2_pseudo_time_normal": PROJECTED_window_1["pseudo_time_normal"],
                    "g2_pca_1": PROJECTED_window_1["pca_1"],
                    "g2_pca_2": PROJECTED_window_1["pca_2"],
                }
                data_output_gibbs_PROJECTED = pd.DataFrame(d)

    return (
        previous_proximity,
        ran_out_of_data_split,
        Model_1,
        Model_2,
        data_output_gibbs_ORIGNAL,
        data_output_gibbs_PROJECTED,
        model_1_map_mean,
        model_1_map_cov,
        model_2_map_mean_G1,
        model_2_map_cov_G1,
        model_2_map_mean_G2,
        model_2_map_cov_G2,
    )
