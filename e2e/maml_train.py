import torch
import learn2learn as l2l
import os
import time
from model import MLP
from dataloader import MAML_Dataset
from utils import parse_opts
import pandas as pd

import sys
sys.path.append('heuristic')
from methods.common import ruler

class MAML_learner(object):
    def __init__(self,opt):
        self.device = opt.device
        self.path = f'{opt.exact_solution}'
        self.model = MLP(756, 256, 450, dropout=0.1).to(self.device) #x_train.shape[1]
        self.ways = opt.ways
    
    def build_tasks(self, mode='train', ways=10, shots=5, num_tasks=100, filter_labels=None):
        loader = MAML_Dataset(mode = mode, path = self.path, ood = True) 
        dataset = l2l.data.MetaDataset(loader)
        new_ways = len(filter_labels) if filter_labels is not None else ways
        assert shots * 2 * new_ways <= dataset.__len__()//ways*new_ways, "Reduce the number of shots!"
        tasks = l2l.data.TaskDataset(dataset, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dataset, new_ways, 2 * shots, filter_labels=filter_labels),
            l2l.data.transforms.LoadData(dataset),
            l2l.data.transforms.RemapLabels(dataset, shuffle=True),
            l2l.data.transforms.ConsecutiveLabels(dataset),
        ], num_tasks=num_tasks)
        return tasks, loader.label_dict

    @staticmethod
    def fast_adapt(batch, label, learner, loss, adaptation_steps, shots, ways):
        opt = parse_opts()
        data, labels = batch
        data, labels = data.to(opt.device), labels.to(opt.device)

        # Create data indices
        indices = torch.arange(data.size(0))
        adaptation_indices = indices[::2][:shots * ways]
        evaluation_indices = indices[1::2]

        # Adaptation data and labels
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        adaptation_labels = torch.tensor([label[i.item()] for i in adaptation_labels]).to(opt.device)
        # adaptation_labels.copy_(torch.tensor([train_data_1_dict[i.item()] for i in adaptation_labels]))

        # Evaluation data and labels
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        evaluation_labels = torch.tensor([label[i.item()] for i in evaluation_labels]).to(opt.device)
        # evaluation_labels.copy_(torch.tensor([train_data_1_dict[i.item()] for i in evaluation_labels]))

        # Adapt the model
        for step in range(adaptation_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(evaluation_data)

        product_list = [ruler(x.reshape(150,3),75) for x in predictions.detach().cpu().numpy()]  # 产线还原

        min_val, _ = torch.min(evaluation_labels, dim=1, keepdim=True)
        max_val, _ = torch.max(evaluation_labels, dim=1, keepdim=True)

        scaled_data = (evaluation_labels - min_val) * 2 / (max_val - min_val) - 1

        valid_error = loss(predictions, scaled_data)
        return valid_error

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, opt, shots=5):
        # Set hyperparameters
        meta_lr = opt.meta_lr
        fast_lr = opt.fast_lr
        meta_batch_size = opt.meta_batch_size
        adaptation_steps = 1 if shots == 5 else 3
        epochs = opt.epochs

        # Initialize MAML algorithm and optimizer
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)
        optim = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = torch.nn.L1Loss()

        # Print training details
        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")

        # Build train and validation tasks
        train_tasks, label_dict = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks, _ = self.build_tasks('validation', valid_ways, shots, 1000, None)

        # train log
        train_error_list = []
        valid_error_list = []

        # Train loop
        for ep in range(epochs):
            # Initialize metrics
            meta_train_error = 0.0
            meta_valid_error = 0.0

            # Meta-Training
            optim.zero_grad()
            for _ in range(meta_batch_size):
                # Clone the MAML algorithm for each task
                learner = maml.clone()

                # Sample a train task and compute meta-training loss
                task = train_tasks.sample()
                evaluation_error = self.fast_adapt(task, label_dict, learner, loss, adaptation_steps, shots, train_ways)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()

                # Sample a validation task and compute meta-validation loss
                task = valid_tasks.sample()
                evaluation_error = self.fast_adapt(task, label_dict, learner, loss, adaptation_steps, shots, valid_ways)
                meta_valid_error += evaluation_error.item()

            # Print metrics
            print(f'Epoch {ep + 1}:')
            print(f'Meta-Train Error: {meta_train_error / meta_batch_size:.4f}')
            print(f'Meta-Valid Error: {meta_valid_error / meta_batch_size:.4f}')
            train_error_list.append(meta_train_error / meta_batch_size)
            valid_error_list.append(meta_valid_error / meta_batch_size)

            # Take the meta-learning step
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            optim.step()

            # Save model if needed
            if (ep + 1) % 50 == 0:
                new_save_path = opt.maml_save + rf'new_ep{ep + 1}'
                self.model_save(new_save_path)
        pd.DataFrame([train_error_list,valid_error_list],index=['train_error','valid_error']).T.to_csv(f'{opt.maml_save}{opt.epochs}_epoch_log.csv',index=False)

    def test(self, load_path, inner_steps=10, shots=5):
        # Load model and print details
        self.model.load_state_dict(torch.load(load_path))
        print(f'Loaded model from {load_path}')
        test_ways = self.ways
        print(f'{test_ways}-ways, {shots}-shots for testing ...')

        # Set hyperparameters
        fast_lr = 0.05
        meta_batch_size = 100
        adaptation_steps = inner_steps
        loss = torch.nn.L1Loss()

        # Build test tasks and initialize MAML algorithm
        test_tasks,label_dict = self.build_tasks('test', test_ways, shots, 1000, None)
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)

        # Test loop
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        t0 = time.time()
        for _ in range(meta_batch_size):
            # Clone MAML algorithm for each task and compute meta-testing loss
            learner = maml.clone()
            task = test_tasks.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(task, None, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f'Time taken for {meta_batch_size*shots} samples: {t1-t0:.4f} seconds.')
        print(f'Meta-Test Error: {meta_test_error / meta_batch_size:.4f}')
        print(f'Meta-Test Accuracy: {meta_test_accuracy / meta_batch_size:.4f}\n')


if __name__ == "__main__":
    opt = parse_opts()
    Net = MAML_learner(opt)  # T2
    Net.train(opt, shots=1)
    # Net.test(load_path='e2e/checkpoints/new_ep400',shots=1)

