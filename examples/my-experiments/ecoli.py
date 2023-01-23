import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tensorboardX import SummaryWriter
import uuid

from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from datasets.datasets import get_ecoil_dataset
from evaluation import Evaluator

import pdb

class Ecoil(Dataset):
	def __init__(self, cuda, batch_size, testing_mode=False):
		self.ds, self.data_shape = get_ecoil_dataset(batch_size)
		self.cuda = cuda
		self.testing_mode = testing_mode
		self._cache = dict()

	def __getitem__(self, index: int) -> torch.Tensor:
		if index not in self._cache:
			self._cache[index] = list(self.ds[index])
			if self.cuda:
				self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
				self._cache[index][1] = torch.tensor(self._cache[index][1], dtype=torch.long).cuda(non_blocking=True)
		return self._cache[index]

	def __len__(self) -> int:
		return 128 if self.testing_mode else len(self.ds)


@click.command()

@click.option("--cuda", help="whether to use CUDA (default True).", type=bool, default=True)

@click.option("--batch-size", help="training batch size (default 256).", type=int, default=56)

@click.option(
	"--pretrain-epochs",
	help="number of pretraining epochs (default 300).",
	type=int,
	default=150,
)

@click.option(
	"--finetune-epochs",
	help="number of finetune epochs (default 500).",
	type=int,
	default=100,
)

@click.option(
	"--testing-mode",
	help="whether to run in testing mode (default False).",
	type=bool,
	default=False,
)

def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
	writer = SummaryWriter()  # create the TensorBoard object
	# callback function to call during training, uses writer from the scope

	def training_callback(epoch, lr, loss, validation_loss):
		writer.add_scalars(	"data/autoencoder",	{"lr": lr, "loss": loss, "validation_loss": validation_loss,}, epoch,)

	ds_train = Ecoil(cuda=cuda, batch_size=batch_size, testing_mode=testing_mode)  # training dataset

	latent_dim = 4
	cluster_number = 8
	data_shape = ds_train.data_shape
	# TODO AutoEncoder
	autoencoder = StackedDenoisingAutoEncoder([data_shape, 14, latent_dim], final_activation=None)
	
	if cuda:
		autoencoder.cuda()

	print("Pretraining stage.")
	ae.pretrain(
		ds_train,
		autoencoder,
		cuda=cuda,
		validation=None,
		epochs=pretrain_epochs,
		batch_size=batch_size,
		optimizer=lambda model: Adam(model.parameters(), lr=0.01),
		scheduler=lambda x: StepLR(x, 100, gamma=0.1),
		corruption=None,
	)

	print("Training stage.")
	ae_optimizer = Adam(params=autoencoder.parameters(), lr=0.01)
	ae.train(
		ds_train,
		autoencoder,
		cuda=cuda,
		validation=None,
		epochs=finetune_epochs,
		batch_size=batch_size,
		optimizer=ae_optimizer,
		scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
		corruption=None,
		update_callback=training_callback,
	)

	print("DEC stage.")
	model = DEC(cluster_number=cluster_number, hidden_dimension=latent_dim, encoder=autoencoder.encoder)
	if cuda:
		model.cuda()
	dec_optimizer = Adam(model.parameters(), lr=0.01)
	train(
		dataset=ds_train,
		model=model,
		epochs=50,
		batch_size=batch_size,
		optimizer=dec_optimizer,
		stopping_delta=None,
		cuda=cuda,
	)

	predicted, actual = predict(
		ds_train, model, 336, silent=True, return_actual=True, cuda=cuda
	)

	actual = actual.cpu().numpy()
	predicted = predicted.cpu().numpy()

	evaluator = Evaluator()
	evaluator.evaluate_clustering(actual, predicted)
	evaluator.print_evaluation()

	reassignment, accuracy = cluster_accuracy(actual, predicted)
	#greedy_pred_labels = transform_clusters_to_labels(actual, predicted)
	#purity = accuracy_score(actual, greedy_pred_labels)
	#nmi = normalized_mutual_info_score(actual, predicted)
	#ari = adjusted_rand_score(actual, predicted)

	#print("Final DEC ACC: {:.2f} PURITY: {:.2f} NMI: {:.2f} ARI: {:.2f}".format(accuracy, purity, nmi, ari))

	if not testing_mode:
		predicted_reassigned = [
			reassignment[item] for item in predicted
		]  # TODO numpify
		confusion = confusion_matrix(actual, predicted_reassigned)
		normalised_confusion = (
			confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
		)
		confusion_id = uuid.uuid4().hex
		sns.heatmap(normalised_confusion).get_figure().savefig(
			"confusion_%s.png" % confusion_id
		)
		print("Writing out confusion diagram with UUID: %s" % confusion_id)
		writer.close()


if __name__ == "__main__":
	main()
