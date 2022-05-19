import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cosine
import argparse
import os
import matplotlib.pyplot as plt
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer

pd.options.mode.chained_assignment = None  # default='warn'


class XplotManager:
    def __init__(self):
        args = self.parse_args()
        self.hp = hpf.load_interp_configs(args.interpreting_params_name)

        self.num_samples_hx = (
            self.hp.analysis.agent_h.num_episodes
        )  # number of generated samples to use
        self.num_samples_hx_grad = 4000
        self.num_samples_env_h = self.hp.analysis.env_h.num_samples
        self.num_samples_clusters = self.hp.analysis.combined_agent_env_hx.num_samples
        self.ts_per_sample = self.hp.analysis.saliency.num_sim_steps
        self.direction_type = self.hp.analysis.saliency.direction_type
        self.num_ica_components = self.hp.analysis.agent_h.n_components_ica
        self.hx_projector = HiddenStateDimensionalityReducer(
            self.hp, self.direction_type, self.num_samples_hx, data_type=np.ndarray
        )
        self.grad_projector = self.hx_projector.project_gradients

        hx_precomp_data_path = os.path.join("analysis", "hx_analysis_precomp")
        # clusters_precomp_data_path = os.path.join("analysis", "jacob_analysis_precomp")
        clusters_precomp_data_path = os.path.join(
            "analysis", "combined_agent_env_hx_analysis_precomp"
        )

        self.env_h_precomp_path = os.path.join("analysis", "env_analysis_precomp")
        env_components = np.load(
            f"{self.env_h_precomp_path}/pcomponents_env_{self.num_samples_env_h}.npy"
        )
        self.cluster_ids = np.load(
            f"{clusters_precomp_data_path}/clusters_per_sample_{self.num_samples_clusters}.npy"
        )
        self.hx_pc_components = np.load(
            f"{hx_precomp_data_path}/pcomponents_{self.num_samples_hx}.npy"
        )
        self.hx_pc_variances = np.load(
            f"{hx_precomp_data_path}/pc_loading_variances_{self.num_samples_hx}.npy"
        )  # TODO put the extra
        self.hx_pc_components = self.hx_pc_components.transpose()
        self.hx_mu = np.load(hx_precomp_data_path + f"/hx_mu_{self.num_samples_hx}.npy")
        self.hx_std = np.load(
            hx_precomp_data_path + f"/hx_std_{self.num_samples_hx}.npy"
        )

        # Prepare load and save dirs
        self.generated_data_path = os.path.join(
            self.hp.generated_data_dir, self.hp.analysis.agent_h.informed_or_random_init
        )
        save_path = self.hp.analysis.xplots.save_dir  # "xcorr_matrices/"
        save_path_data = self.hp.analysis.xplots.save_dir_data
        self.save_path = os.path.join(os.getcwd(), "analysis", save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path_data = os.path.join(os.getcwd(), "analysis", save_path_data)
        os.makedirs(self.save_path_data, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="args for plotting")
        parser.add_argument("--interpreting_params_name", type=str, default="defaults")
        args = parser.parse_args()
        return args

    def plot_direction_xcausation_multi_timestep(self):
        sample_ids = range(0, self.num_samples_hx_grad)
        stacked_samples_grads = self.collect_grads(sample_ids)

        for ts in range(stacked_samples_grads.shape[0]):
            print("Timestep: %i " % ts)
            ts_slice = stacked_samples_grads[ts]
            ts_sub = self.hp.analysis.saliency.common_timesteps[0] - ts
            title_name = f"Cross-causation matrix: grads of hx_direction at t=0 (x-axis) \nwith respect to hx_directions at (t-{ts_sub}) (y-axis)"
            self.plot_heatmap(
                ts_slice,
                "xcaus_hx_t+%i" % int(ts),
                title_name,
                labels=None,
                clim=ts_slice.max() / 2,
            )

    def plot_extrema_xcaus_plots(self):
        """
        For each direction, extrema timestep, and each extrema type
        - Gathers the names of the samples for that d and type
        - plots xcaus plot for that
        """
        hx_ics = self.gather_hx_ic_data()
        extrema_types = ["high", "low"]
        extrema_ts = [0, 2, 4]
        for direction_id in range(0, self.num_ica_components):
            for ext_ts in extrema_ts:
                ext_ts_cond = hx_ics["timestep"] == ext_ts
                ts_sub_ext = ext_ts - self.hp.analysis.saliency.common_timesteps[0]

                for extr_type in extrema_types:
                    # First get extrema samples for this d
                    extr_value = self.extrema_values[extr_type][direction_id]
                    if extr_type == "high":
                        ic_cond = hx_ics[direction_id] > extr_value
                    if extr_type == "low":
                        ic_cond = hx_ics[direction_id] < extr_value

                    combo_cond = ic_cond & ext_ts_cond
                    sample_ids = list(hx_ics[combo_cond]["sample_id"])
                    sample_ids = [int(id) for id in sample_ids]

                    # Get rid of outliers using cluster. First find clusters
                    # that are strongly represented by this extrema group.
                    # Then throw away any samples that are not in the
                    # strongly represented cluster
                    sample_cluster_ids = list(self.cluster_ids[sample_ids])
                    sample_cluster_ids_set = list(set(sample_cluster_ids))
                    cluster_counts = np.array(
                        [sample_cluster_ids.count(i) for i in sample_cluster_ids_set]
                    )
                    kept_cluster_ids_inds = (cluster_counts > 2).nonzero()[
                        0
                    ]  # two is an arbitrary threshold for outlier clusters
                    kept_cluster_ids = [
                        sample_cluster_ids_set[i] for i in kept_cluster_ids_inds
                    ]
                    kept_sample_ids = [
                        sample_ids[i]
                        for i in range(len(sample_ids))
                        if sample_cluster_ids[i] in kept_cluster_ids
                    ]

                    num_samples = len(kept_sample_ids)
                    stacked_samples_grads = self.collect_grads(
                        kept_sample_ids,
                        standardize_scale_per_sample=False,
                        clip_grads=False,
                    )

                    for ts in range(stacked_samples_grads.shape[0]):
                        print("Timestep: %i " % ts)
                        ts_slice = stacked_samples_grads[ts]
                        ts_sub = ts - self.hp.analysis.saliency.common_timesteps[0]
                        if ts_sub > 0:
                            op_str = "+"
                        else:
                            op_str = ""
                        if ts_sub_ext >= 0:
                            op_str_ext = "+"
                        else:
                            op_str_ext = ""
                        title_name = f"Cross-causation matrix: grads of hx_direction at t=0 (x-axis) \nwith respect to hx_directions at (t{op_str}{ts_sub}) (y-axis)\n Only samples where direction {direction_id} is extremely {extr_type} at t{op_str_ext}{ts_sub_ext} (N={num_samples} samples)\n in clusters {kept_cluster_ids}"
                        heatmap_name = f"xcaus_hx_t{op_str}{ts_sub}_d{direction_id}_{extr_type}_at_t{op_str_ext}{ts_sub_ext}"
                        self.plot_heatmap(
                            ts_slice,
                            heatmap_name,
                            title_name,
                            labels=None,
                            clim=ts_slice.max() / 2,
                        )

    def gather_hx_ic_data(self):
        # Collect hx loadings
        hx_loadings = {}
        full_hx_loading_arrs = []
        for sample_id in range(self.num_samples_hx):
            sample_id_str = f"sample_{sample_id:05d}"
            hx = np.load(
                os.path.join(self.generated_data_path, f"{sample_id_str}/agent_hs.npy")
            )
            hx_vecs = self.hx_projector.transform(hx)
            hx_vecs = hx_vecs.transpose()
            hx_loadings[sample_id] = hx_vecs
            full_hx_loading_arrs.append(hx_vecs)
            if (sample_id + 1) % 100 == 0:
                print(f"Samples collected: {sample_id}/{self.num_samples_hx}")

        full_hx_loading_array = np.concatenate(full_hx_loading_arrs, axis=1)
        hx_sorted = np.sort(full_hx_loading_array, axis=1).transpose()
        threshold = self.hp.analysis.xplots.extrema_threshold
        n = hx_sorted.shape[0]
        self.extrema_values = {
            "high": hx_sorted[n - int(n * threshold) - 1],
            "middle_upper": hx_sorted[int(n / 2 + (n * (threshold / 2))) - 1],
            "middle_lower": hx_sorted[int(n / 2 - (n * (threshold / 2))) - 1],
            "low": hx_sorted[int(n * threshold) - 1],
        }

        # Find the samples with extrema
        col_names = ["sample_id", "timestep"]
        col_names.extend(list(range(0, self.num_ica_components)))
        flatten = lambda x: [item for sublist in x for item in sublist]
        ts_per_sample = self.hp.analysis.saliency.num_sim_steps
        sample_ids = [list(range(0, self.num_samples_hx))] * ts_per_sample
        sample_ids = sorted(flatten(sample_ids))
        sample_ids = [int(i) for i in sample_ids]
        timesteps_vals = [list(range(0, ts_per_sample))] * self.num_samples_hx
        timesteps_vals = flatten(timesteps_vals)
        col_values = np.array([sample_ids, timesteps_vals])
        col_values = np.concatenate(
            [col_values.transpose(), full_hx_loading_array.transpose()], axis=1
        )
        hx_ics = pd.DataFrame(col_values, columns=col_names)

        return hx_ics

    def plot_agent_hx_xcorrs_per_cluster(self):
        # Get hx data (Cluster data has already been gotten)

        samples_data_hx = []
        print("Collecting data together...")
        for ep in range(0, self.num_samples_hx):
            hx = np.load(
                os.path.join(self.generated_data_path, f"sample_{ep:05d}/agent_hs.npy")
            )
            hx_vecs = self.hx_projector.transform(hx)
            hx_vecs = hx_vecs.transpose()
            samples_data_hx.append(hx_vecs)
        samples_data_hx = np.stack(samples_data_hx)
        print("Done collecting data together...")

        # for each cluster...
        clusters = list(set(self.cluster_ids))
        for cluster in clusters:
            print(f"Calculating cross correlation plots for cluster {cluster}")
            # Filter hx data for that cluster
            cluster_episode_ids = (self.cluster_ids == cluster).nonzero()[0]
            cluster_episodes = samples_data_hx[cluster_episode_ids]
            cluster_episodes = [
                samples_data_hx[i] for i in range(cluster_episodes.shape[0])
            ]

            # Calc and plot xcorr plot for that cluster
            self.plot_xcorrs_for_sample_group(cluster_episodes, cluster)

        pass

    def plot_xcorrs_for_sample_group(self, samples_data_hx, cluster):
        K = 10
        m = 1e3
        num_samples = len(samples_data_hx)
        for k in np.arange(0, K + 1, 1):
            ## Collect together the pairs of vectors that we'll be using to make the xcorr matrix
            xcorrs_hx = []
            for hx_sample in samples_data_hx:
                # Clip samples so pathologically large PC loadings don't have as much influence
                hx_sample = np.clip(hx_sample, a_min=-m, a_max=m)
                if k == 0:
                    set_a_hx = hx_sample
                    set_b_hx = hx_sample
                else:
                    set_a_hx = hx_sample[:, :-k]
                    set_b_hx = hx_sample[:, k:]

                ## Calculate the xcorr matrices
                n = set_a_hx.shape[0]  # num elements in set
                xcorr_hx = (set_a_hx @ set_b_hx.transpose()) / n

                xcorrs_hx.append(xcorr_hx)

            xcorrs_hx = np.stack(xcorrs_hx)
            xcorrs_hx = np.mean(xcorrs_hx, axis=0)

            ## Plot and save the xcorr matrices
            print("Plotting for k=%i" % int(k))
            plot_name = f"xcorr_hx_clust{cluster}_t+{int(k)}"
            title_name = f"Cross-correlation matrix comparing t (y-axis) with t+{k} (x-axis) for cluster {cluster} (n={num_samples})"
            self.plot_heatmap(
                xcorrs_hx,
                plot_name,
                title_name,
            )
            array_savename = os.path.join(self.save_path_data, plot_name + '.npy')
            np.save(array_savename, xcorrs_hx)

    def plot_direction_xcorrs_multi_timestep_all_samples(self):

        # Collect env and hxs and put them into a list of arrays, where each
        # array is one sample.

        # Get hidden states the were produced by the generative model
        print("Collecting env data together...")
        samples_data_env = []
        samples_data_hx = []
        env_pca_data = np.load(
            self.env_h_precomp_path + f"/pca_data_env_{self.num_samples_env_h}.npy"
        )

        print("Collecting data together...")
        for ep in range(0, min(self.num_samples_hx, self.num_samples_env_h)):
            # Env
            start_ts = ep * self.ts_per_sample
            stop_ts = (ep * self.ts_per_sample) + self.ts_per_sample
            env_vecs = env_pca_data[start_ts:stop_ts]
            samples_data_env.append(env_vecs.transpose())

            # Agent hx
            hx = np.load(
                os.path.join(self.generated_data_path, f"sample_{ep:05d}/agent_hs.npy")
            )
            hx_vecs = self.hx_projector.transform(hx)
            hx_vecs = hx_vecs.transpose()
            samples_data_hx.append(hx_vecs)
        print("Done collecting data together...")

        # For k in {-k, ... , -1, 0, 1, ... , k}
        K = 10
        m = 1e3
        for k in np.arange(0, K + 1, 1):
            ## Collect together the pairs of vectors that we'll be using to make the xcorr matrix
            xcorrs_hx = []
            xcorrs_env = []
            xcorrs_both = []
            for hx_sample, env_sample in zip(samples_data_hx, samples_data_env):
                # Clip samples so pathologically large PC loadings don't have as much influence
                hx_sample = np.clip(hx_sample, a_min=-m, a_max=m)
                env_sample = np.clip(env_sample, a_min=-m, a_max=m)
                if k == 0:
                    set_a_hx = hx_sample
                    set_a_env = env_sample
                    set_b_hx = hx_sample
                    set_b_env = env_sample
                else:
                    set_a_hx = hx_sample[:, :-k]
                    set_a_env = env_sample[:, :-k]
                    set_b_hx = hx_sample[:, k:]
                    set_b_env = env_sample[:, k:]
                set_a_hx_env = np.concatenate([set_a_hx, set_a_env])
                set_b_hx_env = np.concatenate([set_b_hx, set_b_env])

                ## Calculate the xcorr matrices
                n = set_a_hx.shape[0]  # num elements in set
                xcorr_hx = (set_a_hx @ set_b_hx.transpose()) / n
                xcorr_env = (set_a_env @ set_b_env.transpose()) / n
                xcorr_both = (set_a_hx_env @ set_b_hx_env.transpose()) / n

                xcorrs_hx.append(xcorr_hx)
                xcorrs_env.append(xcorr_env)
                xcorrs_both.append(xcorr_both)

            xcorrs_hx = np.stack(xcorrs_hx)
            xcorrs_env = np.stack(xcorrs_env)
            xcorrs_both = np.stack(xcorrs_both)

            xcorrs_hx = np.mean(xcorrs_hx, axis=0)
            xcorrs_env = np.mean(xcorrs_env, axis=0)
            xcorrs_both = np.mean(xcorrs_both, axis=0)

            ## Plot and save the xcorr matrices
            if k == 0:
                ordered_dist_mat, res_order, res_linkage = self.compute_serial_matrix(
                    squareform(pdist(xcorrs_hx)), "ward"
                )
                self.plot_dendrogram(res_linkage, f"dendrogram_{self.direction_type}")

            print("Plotting for k=%i" % int(k))
            title_name = (
                f"Cross-correlation matrix comparing t (y-axis) with t+{k} (x-axis)"
            )

            self.plot_heatmap(
                xcorrs_hx, "xcorr_hx_t+%i" % int(k), title_name, res_order
            )
            self.plot_heatmap(xcorrs_hx, "unordered_xcorr_hx_t+%i" % int(k), title_name)

    def collect_grads(
        self, sample_ids, standardize_scale_per_sample=False, clip_grads=False
    ):
        # Get hidden states gradients that were produced by the generative model
        samples_dir_grads = []

        print("Collecting data together...")
        for ep in sample_ids:
            sample_dir_grads = []
            for direction_id in range(self.num_ica_components):
                grads_name = (
                    f"grad_hx_hx_direction_{direction_id}_{self.direction_type}"
                )
                grads = np.load(
                    os.path.join(
                        self.generated_data_path, f"sample_{ep:05d}/{grads_name}.npy"
                    )
                )  # (T x hx)
                # project grads from hx_space into hx_direction_space
                dir_grads = self.grad_projector(grads)
                sample_dir_grads.append(dir_grads)  # forms each column of a slice
            stacked_sample_grads = np.stack(
                sample_dir_grads, axis=-1
            )  # forms the rows (by sticking columns together)
            samples_dir_grads.append(
                stacked_sample_grads
            )  # So the columns are 'grads_wrt, the rows are 'grads_of'
        stacked_samples_grads = np.stack(samples_dir_grads, axis=-1)

        if clip_grads:
            stacked_samples_grads = stacked_samples_grads.clip(-10, 10)

        if standardize_scale_per_sample:
            # per_sample AND per_timesteps
            shp = stacked_samples_grads.shape
            reshaped_grads = stacked_samples_grads.reshape(
                [shp[0], shp[1] * shp[2], shp[3]]
            )
            std_per_sample = reshaped_grads.std(axis=1)
            std_per_sample = std_per_sample.reshape(shp[0] * shp[3])
            reshaped_grads = stacked_samples_grads.reshape(
                shp[0] * shp[3], shp[1] * shp[2]
            )
            std_per_sample = np.where(
                std_per_sample != 0, std_per_sample, np.ones_like(std_per_sample)
            )  # make safe for div
            reshaped_grads = reshaped_grads / (np.expand_dims(std_per_sample, axis=-1))
            stacked_samples_grads = reshaped_grads.reshape(
                shp[0], shp[1], shp[2], shp[3]
            )

        # Average over samples
        stacked_samples_grads = stacked_samples_grads.mean(-1)
        print("Done collecting grad data together...")
        return stacked_samples_grads

    def plot_heatmap(
        self,
        matrix,
        name,
        plot_title,
        labels=None,
        clim=1.0,
        clim_low=None,
    ):

        plt.rcParams["figure.figsize"] = (10, 10)
        if labels is None:
            plt.imshow(matrix, cmap="seismic_r", interpolation="nearest")
            plt.xticks(ticks=np.arange(0, matrix.shape[0], 1.0))
            plt.yticks(ticks=np.arange(0, matrix.shape[0], 1.0))
        else:
            plt.imshow(matrix[:, labels], cmap="seismic_r", interpolation="nearest")
            plt.xticks(ticks=np.arange(0, matrix.shape[0], 1.0), labels=labels)
            plt.yticks(ticks=np.arange(0, matrix.shape[0], 1.0))
        plt.colorbar()
        if clim_low is None:
            plt.clim(-clim, clim)
        else:
            plt.clim(clim_low, clim)

        plt.tight_layout()
        plt.title(plot_title)
        plt.savefig(os.path.join(self.save_path, "%s.png" % name))
        plt.clf()
        plt.close()

    def plot_dendrogram(self, Z, name):
        plt.figure(figsize=(10, 10))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("ICA direction name")
        plt.ylabel("Similarity of correlations with a timestep")
        dendrogram(
            Z,
            leaf_rotation=0.0,  # rotates the x axis labels
            leaf_font_size=13,  # font size for the x axis labels
        )
        plt.title(f"Dendrogram for ica direction")
        plt.savefig(os.path.join(self.save_path, "%s.png" % name))
        plt.clf()
        plt.close()

    def seriation(self, Z, N, cur_index):
        """
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
        """
        if cur_index < N:
            return [cur_index]
        else:
            left = int(Z[cur_index - N, 0])
            right = int(Z[cur_index - N, 1])
            return self.seriation(Z, N, left) + self.seriation(Z, N, right)

    def compute_serial_matrix(self, dist_mat, method="ward"):
        """
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
        """
        N = len(dist_mat)
        flat_dist_mat = squareform(dist_mat)
        res_linkage = linkage(flat_dist_mat, method=method)
        res_order = self.seriation(res_linkage, N, N + N - 2)
        seriated_dist = np.zeros((N, N))
        a, b = np.triu_indices(N, k=1)
        seriated_dist[a, b] = dist_mat[
            [res_order[i] for i in a], [res_order[j] for j in b]
        ]
        seriated_dist[b, a] = seriated_dist[a, b]

        return seriated_dist, res_order, res_linkage

    def plot_cosine_sim_heatmap(self):
        # Get IC directions in hx space
        hx_dim = self.hp.gen_model.agent_hidden_size
        id_mat = np.eye(hx_dim, hx_dim)
        id_mat_mod = (id_mat * self.hx_projector.hx_std) + self.hx_projector.hx_mu
        ic_directions_in_hx = self.hx_projector.transform(id_mat_mod)
        ic_directions_in_hx = ic_directions_in_hx / np.linalg.norm(
            ic_directions_in_hx, axis=0
        )
        num_ic_dirs = ic_directions_in_hx.shape[1]
        assert num_ic_dirs == self.hp.analysis.agent_h.n_components_ica

        # Calc and plot cosine similarities between each direction
        cosine_sim_mat = np.zeros((num_ic_dirs, num_ic_dirs))
        for direction_id_i in range(num_ic_dirs):
            dir_i = ic_directions_in_hx[:, direction_id_i]
            for direction_id_j in range(num_ic_dirs):
                dir_j = ic_directions_in_hx[:, direction_id_j]
                cos_sim_ij = cosine(dir_i, dir_j)
                cosine_sim_mat[direction_id_i, direction_id_j] = cos_sim_ij
        cosine_sim_mat = np.ones_like(cosine_sim_mat) - cosine_sim_mat
        self.plot_heatmap(
            cosine_sim_mat,
            "Cosine_sim_between_IC_directions_in_hx_space",
            "Cosine similarity between IC directions in hx space",
            clim=1.0,
        )


if __name__ == "__main__":
    xpm = XplotManager()
    xpm.plot_agent_hx_xcorrs_per_cluster()
    # xpm.plot_extrema_xcaus_plots()
    # xcm.plot_direction_xcausation_multi_timestep()
    # xcm.plot_direction_xcorrs_multi_timestep_all_samples()
    # xpm.plot_cosine_sim_heatmap()
