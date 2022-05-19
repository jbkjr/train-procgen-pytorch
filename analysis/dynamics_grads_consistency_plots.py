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


class DynGradManager:
    def __init__(self):
        args = self.parse_args()
        self.hp = hpf.load_interp_configs(args.interpreting_params_name)

        self.num_samples_hx = (
            self.hp.analysis.agent_h.num_episodes
        )  # number of generated samples to use
        self.num_samples_hx_grad = 4000
        # self.num_samples_env_h = self.hp.analysis.env_h.num_samples
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

        # self.env_h_precomp_path = os.path.join("analysis", "env_analysis_precomp")
        # env_components = np.load(
        #     f"{self.env_h_precomp_path}/pcomponents_env_{self.num_samples_env_h}.npy"
        # )
        self.cluster_ids = np.load(
            f"{clusters_precomp_data_path}/clusters_per_sample_{self.num_samples_clusters}.npy"
        )
        self.hx_pc_components = np.load(
            f"{hx_precomp_data_path}/pcomponents_{self.num_samples_hx}.npy"
        )
        self.hx_pc_variances = np.load(
            f"{hx_precomp_data_path}/pc_loading_variances_{self.num_samples_hx}.npy"
        )  # TODO put the extra
        # self.hx_pc_components = self.hx_pc_components.transpose()
        # self.hx_mu = np.load(hx_precomp_data_path + f"/hx_mu_{self.num_samples_hx}.npy")
        # self.hx_std = np.load(
        #     hx_precomp_data_path + f"/hx_std_{self.num_samples_hx}.npy"
        # )

        self.xcorr_data_path = os.path.join(
            os.getcwd(), "analysis", "cross_corr_and_causation_plot_data"
        )

        # Prepare load and save dirs
        self.generated_data_path = os.path.join(
            self.hp.generated_data_dir, self.hp.analysis.agent_h.informed_or_random_init
        )
        save_path = self.hp.analysis.dyn_grad_comparison.save_dir  # "xcorr_matrices/"
        save_path_data = self.hp.analysis.dyn_grad_comparison.save_dir_data
        self.save_path = os.path.join(os.getcwd(), "analysis", save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path_data = os.path.join(os.getcwd(), "analysis", save_path_data)
        os.makedirs(self.save_path_data, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser(description="args for plotting")
        parser.add_argument("--interpreting_params_name", type=str, default="defaults")
        args = parser.parse_args()
        return args

    def plot_consistencies(self):

        # Get arrays for xcorr plots for all clusters and time differences
        K = self.ts_per_sample
        num_clusters = len(list(set(self.cluster_ids)))
        per_cluster_xcorrs = []
        for cluster in list(set(self.cluster_ids)):
            per_ts_xcorrs = []
            for k in np.arange(0, K + 1, 1):
                data_name = f"xcorr_hx_clust{cluster}_t+{int(k)}.npy"
                data_path = os.path.join(self.xcorr_data_path, data_name)
                data = np.load(data_path)
                per_ts_xcorrs.append(data)
            per_ts_xcorrs = np.stack(per_ts_xcorrs)
            per_cluster_xcorrs.append(per_ts_xcorrs)
        per_cluster_xcorrs = np.stack(per_cluster_xcorrs) # (Clusters, Timesteps, IC_d, IC_d)

        # Get arrays for Jacobians for all clusters and time differences


        # Compare sign, create mask

        # Compare magnitude
        pass

    def collect_sample_jacobian(self, generated_data_path, transformer, sample_id):
        n_ic_directions = self.hp.analysis.agent_h.n_components_ica
        ts_id = self.hp.analysis.jacobian.grad_timestep
        jacob_vecs = []
        for direction_id in range(n_ic_directions):  # D
            hx_grad_vecs = np.load(os.path.join(generated_data_path,
                                                f'sample_{sample_id:05d}/grad_hx_hx_direction_{direction_id}_ica.npy'))  # (T x H)
            hx_grad_vec = hx_grad_vecs[ts_id]  # (H,)
            # hx_grad_vec = transformer.project_gradients(hx_grad_vec)  # (I,)

            env_grad_vecs = np.load(os.path.join(generated_data_path,
                                                 f'sample_{sample_id:05d}/grad_env_h_hx_direction_{direction_id}_ica.npy'))
            env_grad_vec = env_grad_vecs[ts_id]
            grad_vec = np.concatenate([hx_grad_vec, env_grad_vec])
            jacob_vecs.append(grad_vec)
        jacobian = np.stack(jacob_vecs, axis=-1)  # (I, D)
        return jacobian


if __name__ == "__main__":
    dgm = DynGradManager()
    dgm.plot_consistencies()
