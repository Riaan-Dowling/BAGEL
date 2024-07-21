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
    pov_data,
):
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
    M1_mu_sample_array = []
    M1_iw_sample_array = []

    # Posterior distrebution
    M1_log_posterior = 0

    """
    -----------------------------------------------------------------
    Prior information Model 2
    -----------------------------------------------------------------
    """

    # Prior
    M2_start_cov = df_prior_opinion_cov
    M2_zeta_hat_start_1 = df.mean(axis=0)

    # Randomized start values
    # M2_iw_sample_G1 = invwishart.rvs(df= m_degrees_of_freedom,scale = M2_start_cov , size=1, random_state=None)
    # M2_mu_sample_G1 = np.random.multivariate_normal(M2_zeta_hat_start_1, M2_iw_sample_G1/(tau), 1).T
    # M2_mu_sample_G1 = M2_mu_sample_G1.tolist()
    # M2_mu_sample_G1 = [i[0] for i in M2_mu_sample_G1]

    # M2_iw_sample_G2 = invwishart.rvs(df= m_degrees_of_freedom,scale = M2_start_cov , size=1, random_state=None)
    # M2_mu_sample_G2 = np.random.multivariate_normal(M2_zeta_hat_start_1, M2_iw_sample_G2/(tau), 1).T
    # M2_mu_sample_G2 = M2_mu_sample_G2.tolist()
    # M2_mu_sample_G2 = [i[0] for i in M2_mu_sample_G2]

    # M2_mu_sample_G2 = M2_mu_sample_G1 #Model 2 G2 mu
    # M2_iw_sample_G2 = M2_iw_sample_G1 #Model 2 G1 var

    M2_Omega_Dirichlet = np.random.dirichlet(
        (Dirichlet_prior, Dirichlet_prior), 1
    ).transpose()
    M2_Omega_Dirichlet_samples = []

    """
    -----------------------------------------------------------------
    Gibbs sampler output Model 2
    -----------------------------------------------------------------
    """
    # Multinomial samples sum
    y_line = np.zeros(N)
    d = {"G1": y_line, "G2": y_line}
    M2_sum_multinomial_samples = pd.DataFrame(d)

    # Sum of all of the components belonging to Gaussian 'x'
    M2_df_G1_sum_total = 0
    M2_df_G2_sum_total = 0

    # Sample parameters
    M2_mu_sample_array_G1 = []
    M2_iw_sample_array_G1 = []

    M2_mu_sample_array_G2 = []
    M2_iw_sample_array_G2 = []

    # Posterior distrebution
    M2_log_posterior = 0

    # ALl of the scale matrix values of Gibbs sampler
    M2_scale_matrix_1_samples = []
    M2_scale_matrix_2_samples = []

    """
    -----------------------------------------------------------------
    Gibbs sampler
    -----------------------------------------------------------------
    """

    # Model 1 paramere calulations
    # Mean
    M1_y_hat = df.mean(axis=0)
    M1_zeta_hat = (np.multiply(tau, mu_0) + N * M1_y_hat) / (tau + N)

    # -Scale matrix
    mean_minus = df - M1_y_hat
    df_s_yy = mean_minus.cov() * N

    # the deviation between prior and estimated mean values
    step_1 = M1_y_hat - mu_0
    step_2 = step_1.T
    zeta_calculation = (N * tau / (N + tau)) * (step_1.dot(step_2))

    # -Resulting scale matrix
    M1_scale_matrix = df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation

    # Random start values
    iw_sample = invwishart.rvs(
        df=m_degrees_of_freedom + N, scale=M1_scale_matrix, size=1, random_state=None
    )
    mu_sample = np.random.multivariate_normal(M1_zeta_hat, iw_sample / (N + tau), 1).T

    M2_iw_sample_G1 = iw_sample
    M2_mu_sample_G1 = mu_sample
    M2_mu_sample_G1 = M2_mu_sample_G1.tolist()
    M2_mu_sample_G1 = [i[0] for i in M2_mu_sample_G1]

    M2_mu_sample_G2 = M2_mu_sample_G1  # Model 2 G2 mu
    M2_iw_sample_G2 = M2_iw_sample_G1  # Model 2 G1 var

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
            scale=M1_scale_matrix,
            size=1,
            random_state=None,
        )
        mu_sample = np.random.multivariate_normal(
            M1_zeta_hat, iw_sample / (N + tau), 1
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
        M2_min_cells_G1 = 1
        M2_min_cells_G2 = 1

        total_searches = 0
        while (M2_min_cells_G1 <= min_cells) or (M2_min_cells_G2 <= min_cells):

            # Multinomal distrebution for Variables
            M2_observation_belongs_1 = []
            M2_observation_belongs_2 = []

            M2_observation_belongs_1 = M2_Omega_Dirichlet[0] * multivariate_normal.pdf(
                df, mean=M2_mu_sample_G1, cov=M2_iw_sample_G1
            )
            M2_observation_belongs_1 = np.nan_to_num(M2_observation_belongs_1)

            M2_observation_belongs_2 = M2_Omega_Dirichlet[1] * multivariate_normal.pdf(
                df, mean=M2_mu_sample_G2, cov=M2_iw_sample_G2
            )
            M2_observation_belongs_2 = np.nan_to_num(M2_observation_belongs_2)

            M2_observation_belong_sum = (
                M2_observation_belongs_1 + M2_observation_belongs_2
            )

            M2_multinomial_samples = []
            z = 0
            for z in range(N):
                M2_multinomial_samples.append(
                    np.random.multinomial(
                        1,
                        [
                            M2_observation_belongs_1[z] / M2_observation_belong_sum[z],
                            M2_observation_belongs_2[z] / M2_observation_belong_sum[z],
                        ],
                    )
                )
            M2_multinomial_samples = pd.DataFrame(M2_multinomial_samples)
            M2_multinomial_samples.columns = ["G1", "G2"]

            # Assign sampled data to Guassian
            M2_df_G1 = df[M2_multinomial_samples["G1"] == 1]
            # M2_df_G1 = np.nan_to_num(M2_df_G1)

            M2_df_G2 = df[M2_multinomial_samples["G2"] == 1]
            # M2_df_G1 = np.nan_to_num(M2_df_G1)

            M2_df_G1_sum = M2_multinomial_samples["G1"].sum()

            M2_df_G2_sum = M2_multinomial_samples["G2"].sum()

            M2_min_cells_G1 = M2_df_G1_sum
            M2_min_cells_G2 = M2_df_G2_sum

            """
            -----------------------------------------------------------------
            Model 2 Dirichlet samples
            -----------------------------------------------------------------
            """
            M2_Omega_Dirichlet = np.random.dirichlet(
                (Dirichlet_prior + M2_df_G1_sum, Dirichlet_prior + M2_df_G2_sum), 1
            ).transpose()

            """
            -----------------------------------------------------------------
            Gibbs sampler help
            -----------------------------------------------------------------
            """
            if total_searches >= 1:
                if M2_df_G1_sum > M2_df_G2_sum:
                    M2_df_G1 = df[M2_multinomial_samples["G1"] == 1]
                    M2_df_G2 = df.sample(n=2)

                    M2_df_G1_sum = M2_multinomial_samples["G1"].sum()
                    M2_df_G2_sum = 2

                    M2_min_cells_G1 = M2_df_G1_sum
                    M2_min_cells_G2 = M2_df_G2_sum

                    """
                    -----------------------------------------------------------------
                    Model 2 Dirichlet samples
                    -----------------------------------------------------------------
                    """
                    M2_Omega_Dirichlet = np.random.dirichlet(
                        (
                            Dirichlet_prior + M2_df_G1_sum,
                            Dirichlet_prior + M2_df_G2_sum,
                        ),
                        1,
                    ).transpose()

                else:
                    M2_df_G1 = df.sample(n=2)
                    M2_df_G2 = df[M2_multinomial_samples["G2"] == 1]

                    M2_df_G1_sum = 2
                    M2_df_G2_sum = M2_multinomial_samples["G2"].sum()

                    M2_min_cells_G1 = M2_df_G1_sum
                    M2_min_cells_G2 = M2_df_G2_sum

                    """
                    -----------------------------------------------------------------
                    Model 2 Dirichlet samples
                    -----------------------------------------------------------------
                    """
                    M2_Omega_Dirichlet = np.random.dirichlet(
                        (
                            Dirichlet_prior + M2_df_G1_sum,
                            Dirichlet_prior + M2_df_G2_sum,
                        ),
                        1,
                    ).transpose()

            total_searches = total_searches + 1  # increment total searches

        """
        -----------------------------------------------------------------
        Model 2 Gaussian 1 samples
        -----------------------------------------------------------------
        """
        M2_y_hat_G1 = M2_df_G1.mean(axis=0)

        # Mean
        M2_zeta_hat_G1 = (np.multiply(tau, mu_0) + M2_df_G1_sum * M2_y_hat_G1) / (
            tau + M2_df_G1_sum
        )
        # Covariance

        mean_minus_1 = M2_df_G1 - M2_y_hat_G1
        df_s_yy = pd.DataFrame(np.dot(mean_minus_1.cov(), M2_df_G1_sum))
        # the deviation between prior and estimated mean values
        step_1 = M2_y_hat_G1 - mu_0
        step_2 = step_1.T
        zeta_calculation = (M2_df_G1_sum * tau / (M2_df_G1_sum + tau)) * (
            step_1.dot(step_2)
        )
        if math.isnan(zeta_calculation) == True:
            M2_scale_matrix_G1 = df_prior_opinion_cov.values
            # print(df_s_yy.values)
        else:
            M2_scale_matrix_G1 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        M2_iw_sample_G1 = invwishart.rvs(
            df=m_degrees_of_freedom + M2_df_G1_sum,
            scale=M2_scale_matrix_G1,
            size=1,
            random_state=None,
        )
        M2_mu_sample_G1 = np.random.multivariate_normal(
            M2_zeta_hat_G1, M2_iw_sample_G1 / (M2_df_G1_sum + tau), 1
        ).T
        M2_mu_sample_G1 = M2_mu_sample_G1.tolist()
        M2_mu_sample_G1 = [i[0] for i in M2_mu_sample_G1]

        """
        -----------------------------------------------------------------
        Model 2 Gaussian 2 samples
        -----------------------------------------------------------------
        """

        M2_y_hat_G2 = M2_df_G2.mean(axis=0)

        # Mean
        M2_zeta_hat_G2 = (np.multiply(tau, mu_0) + M2_df_G2_sum * M2_y_hat_G2) / (
            tau + M2_df_G2_sum
        )
        # Covariance

        mean_minus_2 = M2_df_G2 - M2_y_hat_G2
        df_s_yy = mean_minus_2.cov() * M2_df_G2_sum
        # the deviation between prior and estimated mean values
        step_1 = M2_y_hat_G2 - mu_0
        step_2 = step_1.T
        zeta_calculation = (M2_df_G2_sum * tau / (M2_df_G2_sum + tau)) * (
            step_2.dot(step_2)
        )

        if math.isnan(zeta_calculation) == True:
            M2_scale_matrix_G2 = df_s_yy.values + df_prior_opinion_cov.values
        else:
            M2_scale_matrix_G2 = (
                df_s_yy.values + df_prior_opinion_cov.values + zeta_calculation
            )

        # Samples
        M2_iw_sample_G2 = invwishart.rvs(
            df=m_degrees_of_freedom + M2_df_G2_sum,
            scale=M2_scale_matrix_G2,
            size=1,
            random_state=None,
        )
        M2_mu_sample_G2 = np.random.multivariate_normal(
            M2_zeta_hat_G2, M2_iw_sample_G2 / (M2_df_G2_sum + tau), 1
        ).T
        M2_mu_sample_G2 = M2_mu_sample_G2.tolist()
        M2_mu_sample_G2 = [i[0] for i in M2_mu_sample_G2]

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
            M1_mu_sample_array.append(mu_sample_to_list)
            iw_sample_to_list = iw_sample.tolist()
            M1_iw_sample_array.append(iw_sample_to_list)

            # Calculate log posterior
            M1_post_mu_1 = multivariate_normal.pdf(
                mu_sample_to_list, mean=M1_zeta_hat, cov=iw_sample / (N + tau)
            )
            M1_post_cov_1 = invwishart.pdf(
                iw_sample, df=m_degrees_of_freedom + N, scale=M1_scale_matrix
            )
            M1_log_posterior = (
                M1_log_posterior + np.log(M1_post_mu_1) + np.log(M1_post_cov_1)
            )

            """
            -----------------------------------------------------------------
            Model 2
            -----------------------------------------------------------------
            """
            # Scale matrix
            M2_scale_matrix_1_samples.append(M2_scale_matrix_G1)
            M2_scale_matrix_2_samples.append(M2_scale_matrix_G2)
            # All parameter samples array
            M2_mu_sample_array_G1.append(M2_mu_sample_G1)
            M2_iw_sample_G1_tolist = M2_iw_sample_G1.tolist()
            M2_iw_sample_array_G1.append(M2_iw_sample_G1_tolist)

            M2_mu_sample_array_G2.append(M2_mu_sample_G2)
            M2_iw_sample_G2_tolist = M2_iw_sample_G2.tolist()
            M2_iw_sample_array_G2.append(M2_iw_sample_G2_tolist)

            M2_Omega_Dirichlet_tolist = M2_Omega_Dirichlet.tolist()
            M2_Omega_Dirichlet_tolist = M2_Omega_Dirichlet_tolist[0:]
            M2_Omega_Dirichlet_tolist = [i[0] for i in M2_Omega_Dirichlet_tolist]
            M2_Omega_Dirichlet_samples.append(M2_Omega_Dirichlet_tolist)

            # Estimate what data point correpsonds to which gaussian
            M2_sum_multinomial_samples = (
                M2_sum_multinomial_samples.values + M2_multinomial_samples
            )

            # Calculate log posterior
            M2_post_mu_G1 = multivariate_normal.pdf(
                M2_mu_sample_G1,
                mean=M2_zeta_hat_G1,
                cov=M2_iw_sample_G1 / (M2_df_G1_sum + tau),
            )
            M2_post_cov_G1 = invwishart.pdf(
                M2_iw_sample_G1_tolist,
                df=m_degrees_of_freedom + M2_df_G1_sum,
                scale=M2_scale_matrix_G1,
            )

            M2_post_mu_G2 = multivariate_normal.pdf(
                M2_mu_sample_G2,
                mean=M2_zeta_hat_G2,
                cov=M2_iw_sample_G2 / (M2_df_G2_sum + tau),
            )
            M2_post_cov_G2 = invwishart.pdf(
                M2_iw_sample_G2_tolist,
                df=m_degrees_of_freedom + M2_df_G2_sum,
                scale=M2_scale_matrix_G2,
            )

            M2_post_dirchlet = dirichlet.pdf(
                M2_Omega_Dirichlet_tolist,
                (Dirichlet_prior + M2_df_G1_sum, Dirichlet_prior + M2_df_G2_sum),
            )

            # M2_log_posterior = M2_log_posterior + np.log(M2_post_mu_G1*M2_post_cov_G1*M2_post_mu_G2*M2_post_cov_G2*M2_post_dirchlet)
            M2_log_posterior = (
                M2_log_posterior
                + np.log(M2_post_mu_G1)
                + np.log(M2_post_cov_G1)
                + np.log(M2_post_mu_G2)
                + np.log(M2_post_cov_G2)
                + np.log(M2_post_dirchlet)
            )

            # DF sum of components
            M2_df_G1_sum_total = M2_df_G1_sum_total + M2_df_G1_sum
            M2_df_G2_sum_total = M2_df_G2_sum_total + M2_df_G2_sum

        # if show_itteation == True:
        #     print('Itteration: ' + str(i))

    print("Gibbs sampler end.")

    """
    -----------------------------------------------------------------
    Model 1 Maximum a posteriori
    -----------------------------------------------------------------
    """
    # mu
    df_M1_mu_sample_array_1 = pd.DataFrame(
        M1_mu_sample_array[1:],
        columns=["pseudo_time_normal", "pca_1", "pca_2"],
        dtype="float64",
    )
    M1_map_mean = df_M1_mu_sample_array_1.mean(axis=0)

    # Covariance
    M1_map_cov = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period - 1):
        M1_map_cov = M1_map_cov + np.matrix(M1_iw_sample_array[z])
    M1_map_cov = M1_map_cov / (itterations - burn_period)

    print("Model evidence:")
    """
    -----------------------------------------------------------------
    Model 1 Evidence / Marginal likelihood
    -----------------------------------------------------------------
    """
    # print('M1_log_Evidence start')
    # Log Prior
    M1_prior_mu = multivariate_normal.pdf(
        M1_map_mean, mean=mu_0, cov=df_prior_opinion_cov
    )
    M1_prior_cov = invwishart.pdf(
        M1_map_cov, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    M1_log_prior = np.log(M1_prior_mu) + np.log(M1_prior_cov)
    # M1_log_prior = np.log(M1_prior_mu*M1_prior_cov)

    # Log Likelihood

    M1_log_likelihood = 0
    for row in df.itertuples(index=False):
        qwerty = np.log(multivariate_normal.pdf(row, mean=M1_map_mean, cov=M1_map_cov))
        M1_log_likelihood = M1_log_likelihood + qwerty

    # Log posterior
    M1_log_posterior_divide = M1_log_posterior / (itterations - burn_period)

    M1_log_Evidence = M1_log_likelihood + M1_log_prior - M1_log_posterior_divide

    # print('M1_log_Evidence End')
    print("Model 1 (No bifurcation point) log evidence: " + str(M1_log_Evidence))

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

    M1_predict_distrebution = multivariate_normal(M1_map_mean.T, M1_map_cov)

    M1_df_post = df
    # TODO test this command
    # Model_1_plot_FLAG = True
    # plots.Model_1_plot(x, y, M1_predict_distrebution, pos, df, Model_1_plot_FLAG)

    """
    -----------------------------------------------------------------
    Model 2 Maximum a posteriori
    -----------------------------------------------------------------
    """
    # mu
    M2_df_mu_samples_G1 = pd.DataFrame(
        M2_mu_sample_array_G1[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )
    M2_df_mu_samples_G2 = pd.DataFrame(
        M2_mu_sample_array_G2[1:], columns=["pseudo_time_normal", "pca_1", "pca_2"]
    )

    M2_map_mean_G1 = M2_df_mu_samples_G1.mean(axis=0)
    M2_map_mean_G2 = M2_df_mu_samples_G2.mean(axis=0)

    # Covariance
    M2_map_cov_G1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        M2_map_cov_G1 = M2_map_cov_G1 + np.matrix(M2_iw_sample_array_G1[z])
    M2_map_cov_G1 = M2_map_cov_G1 / (itterations - burn_period)

    M2_map_cov_G2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        M2_map_cov_G2 = M2_map_cov_G2 + np.matrix(M2_iw_sample_array_G2[z])
    M2_map_cov_G2 = M2_map_cov_G2 / (itterations - burn_period)

    # M2_Omega_Dirichlet_samples
    M2_df_Omega_Dirichlet_samples = pd.DataFrame(
        M2_Omega_Dirichlet_samples[1:], columns=["Omega_1", "Omega_2"]
    )
    M2_map_Omega_Dirichlet = M2_df_Omega_Dirichlet_samples.mean(axis=0)

    # Scater matrix
    M2_map_scale_matrix_G1 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        M2_map_scale_matrix_G1 = M2_map_scale_matrix_G1 + np.matrix(
            M2_scale_matrix_1_samples[z]
        )
    M2_map_scale_matrix_G1 = M2_map_scale_matrix_G1 / (itterations - burn_period)

    M2_map_scale_matrix_G2 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for z in range(itterations - burn_period):
        M2_map_scale_matrix_G2 = M2_map_scale_matrix_G2 + np.matrix(
            M2_scale_matrix_2_samples[z]
        )
    M2_map_scale_matrix_G2 = M2_map_scale_matrix_G2 / (itterations - burn_period)

    """
    -----------------------------------------------------------------
    M2_log_Evidence / Marginal likelihood
    -----------------------------------------------------------------
    """
    # Log Prior
    M2_prior_mu_G1 = multivariate_normal.pdf(
        M2_map_mean_G1, mean=mu_0, cov=df_prior_opinion_cov
    )
    prior_cov_1 = invwishart.pdf(
        M2_map_cov_G1, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    M2_prior_mu_G2 = multivariate_normal.pdf(
        M2_map_mean_G2, mean=mu_0, cov=df_prior_opinion_cov
    )
    M2_prior_cov_G2 = invwishart.pdf(
        M2_map_cov_G2, df=m_degrees_of_freedom, scale=df_prior_opinion_cov
    )

    alpha = np.array([Dirichlet_prior, Dirichlet_prior])
    M2_prior_dirchlet = dirichlet.pdf(M2_map_Omega_Dirichlet, alpha)

    # M2_log_prior = np.log(M2_prior_mu_G1*prior_cov_1*M2_prior_mu_G2*M2_prior_cov_G2*M2_prior_dirchlet)

    M2_log_prior = (
        np.log(M2_prior_mu_G1)
        + np.log(prior_cov_1)
        + np.log(M2_prior_mu_G2)
        + np.log(M2_prior_cov_G2)
        + np.log(M2_prior_dirchlet)
    )

    # Log Likelihood

    M2_log_likelihood = 0
    for row in df.itertuples(index=False):
        M2_log_likelihood_data_point = np.log(
            M2_map_Omega_Dirichlet["Omega_1"]
            * multivariate_normal.pdf(row, mean=M2_map_mean_G1, cov=M2_map_cov_G1)
            + M2_map_Omega_Dirichlet["Omega_2"]
            * multivariate_normal.pdf(row, mean=M2_map_mean_G2, cov=M2_map_cov_G2)
        )
        M2_log_likelihood = M2_log_likelihood + M2_log_likelihood_data_point

    # Posterior
    M2_log_posterior_divide = M2_log_posterior / (itterations - burn_period)

    M2_log_Evidence = M2_log_prior + M2_log_likelihood - M2_log_posterior_divide
    print("Model 2 (Bifurcation point) log evidence: " + str(M2_log_Evidence))
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

    M2_sum_multinomial_samples = M2_sum_multinomial_samples.div(
        itterations - burn_period
    )

    M2_df_G1_post = df[M2_sum_multinomial_samples["G1"] > 0.5]
    M2_df_G2_post = df[M2_sum_multinomial_samples["G2"] > 0.5]

    # test = df[(M2_sum_multinomial_samples['G1'] > 0.5) & (M2_sum_multinomial_samples['G2'] > 0.5)]

    # if test.empty ==  False:
    #     print(test)

    # 2 Gaussians plot
    # TODO test Model_2_plot
    # plots.Model_2_plot(
    #     M2_map_mean_G1,
    #     M2_map_cov_G1,
    #     M2_map_mean_G2,
    #     M2_map_cov_G2,
    #     pos,
    #     x,
    #     y,
    #     M2_df_G1_post,
    #     M2_df_G2_post,
    #     Model_2_plot_FLAG,
    # )

    bayes_factor_12 = M1_log_Evidence - M2_log_Evidence  # + model 1 / - model 2

    print("Bayes Factor: " + str(bayes_factor_12))

    if bayes_factor_12 >= 0:
        Model_1 = True
        Model_2 = False

        # ORIGINAL window data
        ORIGINAL_window_1 = (
            window_3d  # .loc[(window_3d['tsne_1'] == M1_df_post['tsne_1'])]
        )
        # Return data
        d = {
            "G1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
            "G1_tsne_1": ORIGINAL_window_1["pca_1"],
            "G1_tsne_2": ORIGINAL_window_1["pca_2"],
        }
        data_output_gibbs_ORIGNAL = pd.DataFrame(d)

        # Return data
        d = {
            "G1_pseudo_time_normal": window_3d["pseudo_time_normal"],
            "G1_tsne_1": window_3d["pca_1"],
            "G1_tsne_2": window_3d["pca_2"],
        }
        data_output_gibbs_PROJECTED = pd.DataFrame(d)

    else:

        # False positive test
        # -----(-)/(-)------
        # Pseudo time
        min_PT_G1 = min(M2_df_G1_post["pseudo_time_normal"])
        max_PT_G1 = max(M2_df_G1_post["pseudo_time_normal"])

        min_PT_G2 = min(M2_df_G2_post["pseudo_time_normal"])
        max_PT_G2 = max(M2_df_G2_post["pseudo_time_normal"])

        min_PT_window = min(window_3d["pseudo_time_normal"])
        max_PT_window = max(window_3d["pseudo_time_normal"])

        # Minimum and Maximum values rows

        min_G1_row_pos = M2_df_G1_post["pseudo_time_normal"].isin([min_PT_G1])
        min_G1_row = M2_df_G1_post[min_G1_row_pos]

        max_G1_row_pos = M2_df_G1_post["pseudo_time_normal"].isin([max_PT_G1])
        max_G1_row = M2_df_G1_post[max_G1_row_pos]

        min_G2_row_pos = M2_df_G2_post["pseudo_time_normal"].isin([min_PT_G2])
        min_G2_row = M2_df_G2_post[min_G2_row_pos]

        max_G2_row_pos = M2_df_G2_post["pseudo_time_normal"].isin([max_PT_G2])
        max_G2_row = M2_df_G2_post[max_G2_row_pos]

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
        wp_data_TSNE_ROW = joblib.load("wp_data_TSNE_ROW.pkl")

        if max_PT_G1 > max_PT_G2:
            test_PT_2 = abs(
                (((min_PT_G1 - max_PT_G2)) / (max_PT_window - min_PT_window)) * 100
            )
            # |----------------->
            # |--->

            true_false = M2_df_G2_post["pseudo_time_normal"].isin(
                wp_data_TSNE_ROW["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            M2_df_G2_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE = M2_df_G2_post[true_false]
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

            true_false = M2_df_G1_post["pseudo_time_normal"].isin(
                wp_data_TSNE_ROW["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            M2_df_G1_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE = M2_df_G1_post[true_false]
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
        #     ax.scatter( M2_df_G1_post.pseudo_time_normal, M2_df_G1_post.pca_1,  M2_df_G1_post.pca_2, c='b', marker='1', s = 20)
        #     ax.scatter( M2_df_G2_post.pseudo_time_normal, M2_df_G2_post.pca_1,  M2_df_G2_post.pca_2, c='k', marker='1', s = 20)
        #     ax.scatter( wp_data_TSNE_ROW.pseudo_time_normal, wp_data_TSNE_ROW.pca_1,  wp_data_TSNE_ROW.pca_2, c='r', marker='o', s = 40)
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
        mean_G1 = M2_df_G1_post.mean(axis=0)
        mean_G1.columns = ["pseudo_time_normal", "pca_1", "pca_2"]

        mean_G2 = M2_df_G2_post.mean(axis=0)
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
                window_3d  # .loc[(window_3d['tsne_1'] == M1_df_post['tsne_1'])]
            )
            # Return data
            d = {
                "G1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                "G1_tsne_1": ORIGINAL_window_1["pca_1"],
                "G1_tsne_2": ORIGINAL_window_1["pca_2"],
            }
            data_output_gibbs_ORIGNAL = pd.DataFrame(d)

            # Return data
            d = {
                "G1_pseudo_time_normal": window_3d["pseudo_time_normal"],
                "G1_tsne_1": window_3d["pca_1"],
                "G1_tsne_2": window_3d["pca_2"],
            }
            data_output_gibbs_PROJECTED = pd.DataFrame(d)

        else:

            """
            Test if either Gausians contains a terminal state
            """
            true_false = M2_df_G2_post["pseudo_time_normal"].isin(
                wp_data_TSNE_ROW["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            M2_df_G2_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_G2 = M2_df_G2_post[true_false]
            test_TERMINAL_STATE_G2.reset_index(drop=True, inplace=True)

            true_false = M2_df_G1_post["pseudo_time_normal"].isin(
                wp_data_TSNE_ROW["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            M2_df_G1_post.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_G1 = M2_df_G1_post[true_false]
            test_TERMINAL_STATE_G1.reset_index(drop=True, inplace=True)

            true_false = pov_data["pseudo_time_normal"].isin(
                wp_data_TSNE_ROW["pseudo_time_normal"].values
            )
            true_false.reset_index(drop=True, inplace=True)
            pov_data.reset_index(drop=True, inplace=True)
            test_TERMINAL_STATE_pov_data = pov_data[true_false]
            test_TERMINAL_STATE_pov_data.reset_index(drop=True, inplace=True)

            if (test_TERMINAL_STATE_G2.empty == False) or (
                test_TERMINAL_STATE_G1.empty == False
            ):
                if test_TERMINAL_STATE_pov_data.empty == False:
                    ran_out_of_data_split = True
            else:
                ran_out_of_data_split = False

            Model_1 = False
            Model_2 = True
            # ORIGINAL window data
            # Gaussian 2
            # Link data to ORIGINAL pseudo time
            con1 = df_cell_ID["pseudo_time_normal"].isin(
                M2_df_G1_post["pseudo_time_normal"].values
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
                M2_df_G2_post["pseudo_time_normal"].values
            )  # (window_3d['tsne_1'].isin(M2_df_G2_post['tsne_1'].values))
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
                    "G1_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                    "G1_tsne_1": ORIGINAL_window_1["pca_1"],
                    "G1_tsne_2": ORIGINAL_window_1["pca_2"],
                    "G2_pseudo_time_normal": ORIGINAL_window_2["pseudo_time_normal"],
                    "G2_tsne_1": ORIGINAL_window_2["pca_1"],
                    "G2_tsne_2": ORIGINAL_window_2["pca_2"],
                }
                data_output_gibbs_ORIGNAL = pd.DataFrame(d)

                # Plane PROJECTED data
                d = {
                    "G1_pseudo_time_normal": PROJECTED_window_1["pseudo_time_normal"],
                    "G1_tsne_1": PROJECTED_window_1["pca_1"],
                    "G1_tsne_2": PROJECTED_window_1["pca_2"],
                    "G2_pseudo_time_normal": PROJECTED_window_2["pseudo_time_normal"],
                    "G2_tsne_1": PROJECTED_window_2["pca_1"],
                    "G2_tsne_2": PROJECTED_window_2["pca_2"],
                }
                data_output_gibbs_PROJECTED = pd.DataFrame(d)

            else:
                # Orignal window data
                d = {
                    "G1_pseudo_time_normal": ORIGINAL_window_2["pseudo_time_normal"],
                    "G1_tsne_1": ORIGINAL_window_2["pca_1"],
                    "G1_tsne_2": ORIGINAL_window_2["pca_2"],
                    "G2_pseudo_time_normal": ORIGINAL_window_1["pseudo_time_normal"],
                    "G2_tsne_1": ORIGINAL_window_1["pca_1"],
                    "G2_tsne_2": ORIGINAL_window_1["pca_2"],
                }
                data_output_gibbs_ORIGNAL = pd.DataFrame(d)

                # Plane PROJECTED data
                d = {
                    "G1_pseudo_time_normal": PROJECTED_window_2["pseudo_time_normal"],
                    "G1_tsne_1": PROJECTED_window_2["pca_1"],
                    "G1_tsne_2": PROJECTED_window_2["pca_2"],
                    "G2_pseudo_time_normal": PROJECTED_window_1["pseudo_time_normal"],
                    "G2_tsne_1": PROJECTED_window_1["pca_1"],
                    "G2_tsne_2": PROJECTED_window_1["pca_2"],
                }
                data_output_gibbs_PROJECTED = pd.DataFrame(d)

    return (
        previous_proximity,
        ran_out_of_data_split,
        Model_1,
        Model_2,
        data_output_gibbs_ORIGNAL,
        data_output_gibbs_PROJECTED,
        M1_map_mean,
        M1_map_cov,
        M2_map_mean_G1,
        M2_map_cov_G1,
        M2_map_mean_G2,
        M2_map_cov_G2,
    )
