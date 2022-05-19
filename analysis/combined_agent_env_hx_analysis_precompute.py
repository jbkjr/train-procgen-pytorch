import pandas as pd
import numpy as np
from precomput_analysis_funcs import scale_then_pca_then_save, plot_variance_expl_plot, clustering_after_pca, \
    tsne_after_pca, clustering_after_tsne
import argparse
import os
import hyperparam_functions as hpf
from dimred_projector import HiddenStateDimensionalityReducer

pd.options.mode.chained_assignment = None  # default='warn'


class ComboAnalysisManager:
    def __init__(self):
        args = self.parse_args()
        self.hp = hpf.load_interp_configs(args.interpreting_params_name)

        self.num_samples_hx = (
            self.hp.analysis.agent_h.num_episodes
        )  # number of generated samples to use
        self.num_samples_env_h = self.hp.analysis.env_h.num_samples
        self.num_samples = self.hp.analysis.combined_agent_env_hx.num_samples
        self.n_components_pca = self.hp.analysis.combined_agent_env_hx.n_components_pca
        self.n_components_tsne = self.hp.analysis.combined_agent_env_hx.n_components_tsne
        self.n_clusters = self.hp.analysis.combined_agent_env_hx.n_clusters
        self.n_ic_directions = self.hp.analysis.agent_h.n_components_ica
        self.n_hx_dims = self.hp.gen_model.agent_hidden_size
        self.n_env_dims = self.hp.gen_model.deter_dim
        self.ts_per_sample = self.hp.analysis.saliency.num_sim_steps
        self.timestep_differences = self.hp.analysis.combined_agent_env_hx.timestep_differences
        self.num_ts_to_keep = self.ts_per_sample - max(self.timestep_differences)
        self.n_env_dims_to_keep = 16#self.n_hx_dims

        # transformer = HiddenStateDimensionalityReducer(self.hp, 'ica', num_samples, data_type=np.ndarray)

        # Prepare load and save dirs
        generated_data_path = os.path.join(self.hp.generated_data_dir, 'informed_init')
        save_path = 'combined_agent_env_hx_analysis_precomp/'
        self.save_path = os.path.join(os.getcwd(), "analysis", save_path)
        plot_save_path = 'combined_agent_env_hx_analysis_plots'
        self.plot_save_path = os.path.join(os.getcwd(), "analysis", plot_save_path)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.plot_save_path, exist_ok=True)
        self.generated_data_path = os.path.join(self.hp.generated_data_dir,
                                                self.hp.analysis.agent_h.informed_or_random_init)
        self.env_h_precomp_path = os.path.join("analysis", "env_analysis_precomp")


    def parse_args(self):
        parser = argparse.ArgumentParser(
            description='args for plotting')
        parser.add_argument(
            '--interpreting_params_name', type=str,
            default='defaults')
        args = parser.parse_args()
        return args

    def collect_hx_and_env(self):
        env_pca_data = np.load(
            self.env_h_precomp_path + f"/pca_data_env_{self.num_samples_env_h}.npy"
        )
        # Collect env and hxs and put them into a list of arrays, where each
        # array is one sample.

        # Get hidden states the were produced by the generative model
        print("Collecting env data together...")
        samples_data_env = []
        samples_data_hx = []

        print("Collecting data together...")
        for ep in range(0, min(self.num_samples_hx, self.num_samples_env_h)):
            print(ep)
            # Env
            start_ts = ep * self.ts_per_sample
            stop_ts = (ep * self.ts_per_sample) + self.ts_per_sample
            env_vecs = env_pca_data[start_ts:stop_ts]
            env_vecs = env_vecs[:, :self.n_env_dims_to_keep]
            samples_data_env.append(env_vecs.transpose())
            # env_vecs = np.load(
            #     os.path.join(self.generated_data_path, f"sample_{ep:05d}/env_hs.npy")
            # )
            # samples_data_env.append(env_vecs.transpose())

            # Agent hx
            hx = np.load(
                os.path.join(self.generated_data_path, f"sample_{ep:05d}/agent_hs.npy")
            )
            samples_data_hx.append(hx.transpose())
        print("Done collecting data together...")
        return samples_data_hx, samples_data_env

    def collect_state_pairs_k_timesteps_apart(self, k, samples_data_hx, samples_data_env):
        """Returns pairs of vectors that are k timesteps apart within an episode"""

        # For k in {-k, ... , -1, 0, 1, ... , k}
        # K = 10
        m = 1e3
        # for k in np.arange(0, K + 1, 1):
        ## Collect together the pairs of vectors that we'll be using to make the xcorr matrix
        set_t = []
        set_t_plus_k = []
        for hx_sample, env_sample in zip(samples_data_hx, samples_data_env):
            # Clip samples so pathologically large PC loadings don't have as much influence
            hx_sample = np.clip(hx_sample, a_min=-m, a_max=m)
            env_sample = np.clip(env_sample, a_min=-m, a_max=m)  # TODO figure out a better outlier prevention mechanism
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

            set_t.append(set_a_hx_env)
            set_t_plus_k.append(set_b_hx_env)

        return set_t, set_t_plus_k

    def calculate_difference_data(self):
        hxs, env_hxs = self.collect_hx_and_env()
        difference_sets = []
        for k in self.timestep_differences:
            set_t, set_t_plus_k = self.collect_state_pairs_k_timesteps_apart(k, hxs, env_hxs)
            differences_k = [t - t_plus_k for (t, t_plus_k) in zip(set_t, set_t_plus_k)]
            differences_k_cut = [vecs[:, :self.num_ts_to_keep] for vecs in differences_k]
            difference_sets.append(differences_k_cut)

        # Clip off the last few ts so that they're the same length as the
        # difference tensors.
        hxs_cut = [hx[:, :self.num_ts_to_keep] for hx in hxs]
        env_hxs_cut = [env_hx[:, :self.num_ts_to_keep] for env_hx in env_hxs]

        # Combine everything together into one big dataset
        hxs_env_hxs = [np.concatenate([hx, env_hx]) for hx, env_hx in zip(hxs_cut, env_hxs_cut)]
        data_sets = difference_sets
        del difference_sets
        data_sets = [hxs_env_hxs] + data_sets
        combined_data_sets = [np.concatenate(list(data), axis=0) for data in zip(*data_sets)]
        combined_data_sets = np.concatenate(combined_data_sets, axis=1).transpose()
        return combined_data_sets

    def analyse_data(self):
        data = self.calculate_difference_data()
        # clustering
        print('Starting clustering...')
        clustering_after_tsne(data, self.n_components_pca, self.n_components_tsne, self.n_clusters, self.save_path,
                              "combined_agent_env_hx", self.num_samples)
        print("Clustering finished.")

        # # tSNE
        print('Starting tSNE...')
        tsne_after_pca(data, self.n_components_pca, self.n_components_tsne,
                       self.save_path, "combined_agent_env_hx", self.num_samples)
        print("tSNE finished.")

        self.save_per_sample_clusters(data)

    def save_per_sample_clusters(self, data):
        assert data.shape[0] / self.num_ts_to_keep == self.num_samples
        # Load clusters
        cluster_ids = np.load(os.path.join(self.save_path, f'clusters_combined_agent_env_hx_{self.num_samples}.npy'))

        # Break up list of clusters into samples
        cluster_ids_per_sample = \
            [cluster_ids[(i * self.num_ts_to_keep):
                         ((i * self.num_ts_to_keep) + self.num_ts_to_keep)]
             for i in range(self.num_samples)]

        # Throw away all but the first K ts
        trunc_length = min(5, cluster_ids_per_sample[0].shape[0])
        cluster_ids_per_sample = [sample[:trunc_length] for sample in cluster_ids_per_sample]

        # find the modal cluster for each sample
        sample_modal_cluster = [max(set(list(sample)), key=list(sample).count) for sample in cluster_ids_per_sample]

        # save
        np.save(
            os.path.join(self.save_path, f'clusters_per_sample_{self.num_samples}'),
            np.array(sample_modal_cluster)
        )


def run():
    cam = ComboAnalysisManager()
    cam.analyse_data()


if __name__ == "__main__":
    run()