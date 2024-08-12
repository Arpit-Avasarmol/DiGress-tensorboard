import os
import time
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import PredefinedNoiseSchedule
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLoss
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, NLL
from src import utils

writer = SummaryWriter()

class LiftedDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features=None,
                 domain_features=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.norm_values = cfg.model.normalize_factors
        self.norm_biases = cfg.model.norm_biases
        self.gamma = PredefinedNoiseSchedule(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)
        diffusion_utils.check_issues_norm_values(self.gamma, self.norm_values[1], self.norm_values[2])

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.val_nll = NLL()
        self.val_X_mse = SumExceptBatchMSE()
        self.val_E_mse = SumExceptBatchMSE()
        self.val_y_mse = SumExceptBatchMSE()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMSE()

        self.test_nll = NLL()
        self.test_X_mse = SumExceptBatchMSE()
        self.test_E_mse = SumExceptBatchMSE()
        self.test_y_mse = SumExceptBatchMSE()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMSE()

        self.train_loss = TrainLoss()
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.save_hyperparameters()

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        
        # TensorBoard writer
        self.writer = SummaryWriter()

    def training_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        mse = self.train_loss(masked_pred_epsX=pred.X,
                              masked_pred_epsE=pred.E,
                              pred_y=pred.y,
                              true_epsX=noisy_data['epsX'],
                              true_epsE=noisy_data['epsE'],
                              true_y=noisy_data['epsy'],
                              log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_epsX=pred.X,
                           masked_pred_epsE=pred.E,
                           pred_y=pred.y,
                           true_epsX=noisy_data['epsX'],
                           true_epsE=noisy_data['epsE'],
                           true_y=noisy_data['epsy'], log=i % self.log_every_steps == 0)

        return {'loss': mse}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_mse: {to_log['train_epoch/epoch_X_mse'] :.3f}"
                      f" -- E mse: {to_log['train_epoch/epoch_E_mse'] :.3f} --"
                      f" y_mse: {to_log['train_epoch/epoch_y_mse'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        
        # TensorBoard logging
        for key, value in to_log.items():
            self.writer.add_scalar(key, value, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_mse.reset()
        self.val_E_mse.reset()
        self.val_y_mse.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        nll = self.compute_val_loss(pred, noisy_data, normalized_data.X, normalized_data.E, normalized_data.y,
                                    node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_mse.compute(), self.val_E_mse.compute(),
                   self.val_y_mse.compute(), self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        log_dict = {
            "val/epoch_NLL": metrics[0],
            "val/X_mse": metrics[1],
            "val/E_mse": metrics[2],
            "val/y_mse": metrics[3],
            "val/X_logp": metrics[4],
            "val/E_logp": metrics[5],
            "val/y_logp": metrics[6]
        }

        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, self.current_epoch)

        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident,
                                                 batch_size=to_generate,
                                                 num_nodes=None, save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_mse.reset()
        self.test_E_mse.reset()
        self.test_y_mse.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, normalized_data.X, normalized_data.E,
                                    normalized_data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_mse.compute(), self.test_E_mse.compute(),
                   self.test_y_mse.compute(), self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        log_dict = {
            "test/epoch_NLL": metrics[0],
            "test/X_mse": metrics[1],
            "test/E_mse": metrics[2],
            "test/y_mse": metrics[3],
            "test/X_logp": metrics[4],
            "test/E_logp": metrics[5],
            "test/y_logp": metrics[6]
        }

        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, self.current_epoch)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type MSE {metrics[1] :.2f} -- ",
              f"Test Edge type MSE: {metrics[2] :.2f} -- Test Global feat. MSE {metrics[3] :.2f}",
              f"-- Test X Reconstruction loss {metrics[4] :.2f} -- Test E Reconstruction loss {metrics[5] :.2f}",
              f"-- Test y Reconstruction loss {metrics[6] : .2f}\n")

        test_nll = metrics[0]
        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()

    def kl_prior(self, X, E, y, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1))
        ones = ones.type_as(X)
        gamma_T = self.gamma(ones)
        alpha_T = diffusion_utils.alpha(gamma_T, X.size())

        # Compute means.
        mu_T_X = alpha_T * X
        mu_T_E = alpha_T.unsqueeze(1) * E
        mu_T_y = alpha_T.squeeze(1) * y

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_X = diffusion_utils.sigma(gamma_T, mu_T_X.size())
        sigma_T_E = diffusion_utils.sigma(gamma_T, mu_T_E.size())
        sigma_T_y = diffusion_utils.sigma(gamma_T, mu_T_y.size())

        # Compute KL for h-part.
        kl_distance_X = diffusion_utils.gaussian_KL(mu_T_X, sigma_T_X)
        kl_distance_E = diffusion_utils.gaussian_KL(mu_T_E, sigma_T_E)
        kl_distance_y = diffusion_utils.gaussian_KL(mu_T_y, sigma_T_y)

        return kl_distance_X + kl_distance_E + kl_distance_y

    def log_constants_p_y_given_z0(self, batch_size):
        """ Computes p(y|z0)= -0.5 ydim (log(2pi) + gamma(0)).
            sigma_y = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
            output size: (batch_size)
        """
        if self.ydim_output == 0:
            return 0.0

        zeros = torch.zeros((batch_size, 1))
        gamma_0 = self.gamma(zeros).squeeze(1)
        return -0.5 * self.ydim * (gamma_0 + np.log(2 * np.pi))

    def reconstruction_logp(self, data, data_0, gamma_0, eps, pred_0, node_mask, epsilon=1e-10, test=False):
        """ Reconstruction loss.
            output size: (1).
        """
        X, E, y = data.values()
        X_0, E_0, y_0 = data_0.values()

        _, _, eps_y0 = eps.values()
        predy = pred_0.y

        if test:
            error_y = -0.5 * self.test_y_logp(predy, eps_y0)
        else:
            error_y = -0.5 * self.val_y_logp(predy, eps_y0)
        neg_log_constants = - self.log_constants_p_y_given_z0(y.shape[0])
        log_py = error_y + neg_log_constants

        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=X_0.size())
        sigma_0_X = sigma_0 * self.norm_values[0]
        sigma_0_E = (sigma_0 * self.norm_values[1]).unsqueeze(-1)

        unnormalized_data = utils.unnormalize(X, E, y, self.norm_values, self.norm_biases, node_mask, collapse=False)
        unnormalized_0 = utils.unnormalize(X_0, E_0, y_0, self.norm_values, self.norm_biases, node_mask, collapse=False)
        X_0, E_0, _ = unnormalized_0.X, unnormalized_0.E, unnormalized_0.y
        assert unnormalized_data.X.size() == X_0.size()

        E_0_centered = E_0 - 1
        X_0_centered = X_0 - 1

        log_pE_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((E_0_centered + 0.5) / sigma_0_E)
            - diffusion_utils.cdf_std_gaussian((E_0_centered - 0.5) / sigma_0_E)
            + epsilon)

        log_pX_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((X_0_centered + 0.5) / sigma_0_X)
            - diffusion_utils.cdf_std_gaussian((X_0_centered - 0.5) / sigma_0_X)
            + epsilon)

        norm_cst_E = torch.logsumexp(log_pE_proportional, dim=-1, keepdim=True)
        norm_cst_X = torch.logsumexp(log_pX_proportional, dim=-1, keepdim=True)

        log_probabilities_E = log_pE_proportional - norm_cst_E
        log_probabilities_X = log_pX_proportional - norm_cst_X

        logps = utils.PlaceHolder(X=log_probabilities_X * unnormalized_data.X,
                                  E=log_probabilities_E * unnormalized_data.E,
                                  y=None).mask(node_mask)

        if test:
            log_pE = - self.test_E_logp(-logps.E)
            log_pX = - self.test_X_logp(-logps.X)
        else:
            log_pE = - self.val_E_logp(-logps.E)
            log_pX = - self.val_X_logp(-logps.X)
        return log_pE + log_pX + log_py

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1))
        t_int = t_int.type_as(X).float()
        s_int = t_int - 1
        s_normalized = s_int / self.T
        t_normalized = t_int / self.T
        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_normalized), X.size())
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_normalized), X.size())
        alpha_t = diffusion_utils.alpha(gamma_t, X.size())
        sigma_t = diffusion_utils.sigma(gamma_t, X.size())
        eps = diffusion_utils.sample_feature_noise(X.size(), E.size(), y.size(), node_mask).type_as(X)
        X_t = alpha_t * X + sigma_t * eps.X
        E_t = alpha_t.unsqueeze(1) * E + sigma_t.unsqueeze(1) * eps.E
        y_t = alpha_t.squeeze(1) * y + sigma_t.squeeze(1) * eps.y

        noisy_data = {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
                      'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
                      'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}

        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        s = noisy_data['s']
        gamma_s = noisy_data['gamma_s']
        gamma_t = noisy_data['gamma_t']
        epsX = noisy_data['epsX']
        epsE = noisy_data['epsE']
        epsy = noisy_data['epsy']
        X_t = noisy_data['X_t']
        E_t = noisy_data['E_t']
        y_t = noisy_data['y_t']

        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)
        kl_prior_without_prefactor = self.kl_prior(X, E, y, node_mask)

        delta_log_py = -self.ydim_output * np.log(self.norm_values[2])
        delta_log_px = -self.Xdim_output * N * np.log(self.norm_values[0])
        delta_log_pE = -self.Edim_output * 0.5 * N * (N-1) * np.log(self.norm_values[1])
        kl_prior = kl_prior_without_prefactor - delta_log_px - delta_log_py - delta_log_pE

        SNR_weight = - (1 - diffusion_utils.SNR(gamma_s - gamma_t))
        sqrt_SNR_weight = torch.sqrt(SNR_weight)
        weighted_predX_diffusion = sqrt_SNR_weight * pred.X
        weighted_epsX_diffusion = sqrt_SNR_weight * epsX
        weighted_predE_diffusion = sqrt_SNR_weight.unsqueeze(1) * pred.E
        weighted_epsE_diffusion = sqrt_SNR_weight.unsqueeze(1) * epsE
        weighted_predy_diffusion = sqrt_SNR_weight.squeeze(1) * pred.y
        weighted_epsy_diffusion = sqrt_SNR_weight.squeeze(1) * epsy

        if test:
            diffusion_error = (self.test_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.test_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion) +
                               self.test_y_mse(weighted_predy_diffusion, weighted_epsy_diffusion))
        else:
            diffusion_error = (self.val_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.val_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion) +
                               self.val_y_mse(weighted_predy_diffusion, weighted_epsy_diffusion))
        loss_all_t = 0.5 * self.T * diffusion_error

        t_zeros = torch.zeros_like(s)
        gamma_0 = diffusion_utils.inflate_batch_array(self.gamma(t_zeros), X_t.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, X_t.size())
        sigma_0 = diffusion_utils.sigma(gamma_0, X_t.size())
        eps0 = diffusion_utils.sample_feature_noise(X_t.size(), E_t.size(), y_t.size(), node_mask).type_as(X_t)

        X_0 = alpha_0 * X_t + sigma_0 * eps0.X
        E_0 = alpha_0.unsqueeze(1) * E_t + sigma_0.unsqueeze(1) * eps0.E
        y_0 = alpha_0.squeeze(1) * y_t + sigma_0.squeeze(1) * eps0.y

        noisy_data0 = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': t_zeros}
        extra_data = self.compute_extra_data(noisy_data)
        pred_0 = self.forward(noisy_data0, extra_data, node_mask)

        loss_term_0 = - self.reconstruction_logp(data={'X': X, 'E': E, 'y': y},
                                                 data_0={'X_0': X_0, 'E_0': E_0, 'y_0': y_0},
                                                 gamma_0=gamma_0,
                                                 eps={'eps_X0': eps0.X, 'eps_E0': eps0.E, 'eps_y0': eps0.y},
                                                 pred_0=pred_0,
                                                 node_mask=node_mask,
                                                 test=test)

        nlls = - log_pN + kl_prior + loss_all_t + loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'
        nll = self.test_nll(nlls) if test else self.val_nll(nlls)

        # TensorBoard logging
        self.writer.add_scalar("kl prior", kl_prior.mean(), self.current_epoch)
        self.writer.add_scalar("Estimator loss terms", loss_all_t.mean(), self.current_epoch)
        self.writer.add_scalar("Loss term 0", loss_term_0, self.current_epoch)
        self.writer.add_scalar("log_pn", log_pN.mean(), self.current_epoch)
        self.writer.add_scalar('test_nll' if test else 'val_nll', nll, self.current_epoch)
        
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2)
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3)
        y = torch.hstack((noisy_data['y_t'], extra_data.y))
        return self.model(X, E, y, node_mask)

    def log_info(self):
        gamma_0 = self.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {'log_SNR_max': log_SNR_max.item(), 'log_SNR_min': log_SNR_min.item()}
        print("", info, "\n")

        return info

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, save_final: int, number_chain_steps: int,
                      num_nodes=None):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_nodes_max = torch.max(n_nodes).item()

        arange = torch.arange(n_nodes_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask = node_mask.float()

        z_T = diffusion_utils.sample_feature_noise(X_size=(batch_size, n_nodes_max, self.Xdim_output),
                                                   E_size=(batch_size, n_nodes_max, n_nodes_max, self.Edim_output),
                                                   y_size=(batch_size, self.ydim_output),
                                                   node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        average_X_coord = []
        average_E_coord = []
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            z_s = self.sample_p_zs_given_zt(s=s_norm, t=t_norm, X_t=X, E_t=E, y_t=y, node_mask=node_mask)
            X, E, y = z_s.X, z_s.E, z_s.y
            write_index = (s_int * number_chain_steps) // self.T
            unnormalized = utils.unnormalize(X=X[:keep_chain], E=E[:keep_chain], y=y[:keep_chain],
                                             norm_values=self.norm_values,
                                             norm_biases=self.norm_biases,
                                             node_mask=node_mask[:keep_chain],
                                             collapse=True)

            chain_X[write_index] = unnormalized.X
            chain_E[write_index] = unnormalized.E
            average_X_coord.append(X.abs().mean().item())
            average_E_coord.append(E.abs().mean().item())

        print(f"Average X coordinate at each step {[int(c) for i, c in enumerate(average_X_coord) if i % 10 == 0]}")
        print(f"Average E coordinate at each step {[int(c) for i, c in enumerate(average_E_coord) if i % 10 == 0]}")

        final_graph = self.sample_discrete_graph_given_z0(X, E, y, node_mask)
        X, E, y = final_graph.X, final_graph.E, final_graph.y
        assert (E == torch.transpose(E, 1, 2)).all()

        print("Examples of generated graphs:")
        for i in range(min(5, X.shape[0])):
            print("E", E[i])
            print("X: ", X[i])

        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]
            chain_X[0] = final_X_chain
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        if self.visualization_tools is not None:
            print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)

            print("Visualizing molecules...")
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final, log='graph')
            print("Done.")

        return molecule_list

    def sample_discrete_graph_given_z0(self, X_0, E_0, y_0, node_mask):
        zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
        gamma_0 = self.gamma(zeros)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        noisy_data = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': torch.zeros(y_0.shape[0], 1).type_as(y_0)}
        extra_data = self.compute_extra_data(noisy_data)
        eps0 = self.forward(noisy_data, extra_data, node_mask)

        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)
        pred_E = 1. / alpha_0.unsqueeze(1) * (E_0 - sigma_0.unsqueeze(1) * eps0.E)
        pred_y = 1. / alpha_0.squeeze(1) * (y_0 - sigma_0.squeeze(1) * eps0.y)
        assert (pred_E == torch.transpose(pred_E, 1, 2)).all()

        sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, node_mask).type_as(pred_X)
        assert (sampled.E == torch.transpose(sampled.E, 1, 2)).all()

        sampled = utils.unnormalize(sampled.X, sampled.E, sampled.y, self.norm_values,
                                    self.norm_biases, node_mask, collapse=True)
        return sampled

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        E_t = (E_t + E_t.transpose(1, 2)) / 2
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t}
        extra_data = self.compute_extra_data(noisy_data)
        eps = self.forward(noisy_data, extra_data, node_mask)

        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X
        mu_E = E_t / alpha_t_given_s.unsqueeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).unsqueeze(1) * eps.E
        mu_y = y_t / alpha_t_given_s.squeeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).squeeze(1) * eps.y
        sigma = sigma_t_given_s * sigma_s / sigma_t

        z_s = diffusion_utils.sample_normal(mu_X, mu_E, mu_y, sigma, node_mask).type_as(mu_X)

        return z_s

    def compute_extra_data(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        extra_x = torch.zeros((*X.shape[:-1], 0)).type_as(X)
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
        t = noisy_data['t']
        return utils.PlaceHolder(X=extra_x, E=extra_edge_attr, y=t)

writer.close()