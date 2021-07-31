from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer


class DMelodiesCNNVAETrainer(DMelodiesVAETrainer):
    def __init__(
            self,
            dataset,
            model: DMelodiesCNNVAE,
            model_type='beta-VAE',
            lr=1e-4,
            # reg_type: Tuple[str] = None,
            beta=0.001,
            gamma=1.0,
            delta=10.0,
            capacity=0.0,
            device=0,
            rand=0,
    ):
        super(DMelodiesCNNVAETrainer, self).__init__(
            dataset, model, model_type, lr, beta, gamma, delta, capacity, device, rand
        )
