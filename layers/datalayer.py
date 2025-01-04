# Required imports for data processing, neural networks, and audio handling
from functools import partial
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import logging
from nemo.backends.pytorch import DataLayerNM
from nemo.collections.asr.parts.dataset import AudioLabelDataset, seq_collate_fn
from nemo.collections.asr.parts.features import WaveformFeaturizer
from nemo.collections.asr.parts.perturb import AudioAugmentor, perturbation_types
from nemo.core.neural_types import *
from torch.utils.data.sampler import BatchSampler

# Custom Batch Sampler to balance classes during batching
class BalancedBatchSampler(BatchSampler):
    """
    Custom BatchSampler for datasets with imbalanced classes.
    Samples a fixed number of classes (n_classes) and a fixed number of examples (n_samples) per class.
    Returns batches of size n_classes * n_samples.
    """

    def __init__(self, labels, n_classes, n_samples, class_dists, class_probs, probs_num):
        """
        Args:
            labels (list or tensor): The class labels for the dataset.
            n_classes (int): Number of classes to sample in a batch.
            n_samples (int): Number of samples to pick from each class.
            class_dists (dict): Class distances for "nearby" sampling.
            class_probs (dict): Probabilities for class-based sampling.
            probs_num (int): Index to select a probability distribution for batch creation.
        """
        self.labels = torch.tensor(labels)  # Convert labels to tensor
        self.labels_set = list(set(self.labels.numpy()))  # Unique class labels
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] 
                                 for label in self.labels_set}  # Map label to its indices
        
        # Shuffle indices for each label to randomize batches
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        
        self.used_label_indices_count = {label: 0 for label in self.labels_set}  # Track used indices
        self.count = 0  # Counter to track dataset iteration
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)  # Total dataset size
        self.batch_size = self.n_samples * self.n_classes  # Batch size
        self.class_dists = class_dists  # Class distance mapping for "nearby" sampling
        self.class_probs = class_probs  # Class probabilities for weighted sampling
        
        # Predefined probability distributions for sampling strategies
        self.probs = [
            [1., 0., 0.],  # Uniform sampling
            [0., 1., 0.],  # Nearby sampling
            [0., 0., 1.],  # Probability-based sampling
            [0.5, 0.5, 0.],  # Mixed uniform and nearby
            [0.5, 0., 0.5],  # Mixed uniform and probability
            [0., 0.5, 0.5],  # Mixed nearby and probability
            [0.33, 0.33, 0.33]  # Equal weight for all three
        ]
        self.probs = self.probs[probs_num]  # Select the desired probability distribution

    def pick_nearby(self, label_set, n_classes, class_dists):
        """
        Picks a combination of labels based on nearby distance.
        Args:
            label_set (list): Available labels.
            n_classes (int): Number of classes to sample.
            class_dists (dict): Mapping of class distances.
        Returns:
            numpy.ndarray: Selected labels.
        """
        first_labels = np.random.choice(label_set, n_classes // 2, replace=False)
        second_labels = []
        for label in first_labels:
            # Pick a nearby label for each selected label
            for sec_label in class_dists[label][np.random.randint(3):]:
                if sec_label in label_set:
                    second_labels.append(sec_label)
                    break
        return np.concatenate([first_labels, np.array(second_labels)])

    def pick_probs(self, label_set, n_classes, class_probs):
        """
        Picks labels based on class probabilities.
        Args:
            label_set (list): Available labels.
            n_classes (int): Number of classes to sample.
            class_probs (dict): Probability weights for classes.
        Returns:
            numpy.ndarray: Selected labels.
        """
        return np.random.choice(label_set, n_classes, p=class_probs[label_set] / np.sum(class_probs[label_set]))

    def __iter__(self):
        """
        Creates an iterator for the sampler.
        """
        self.count = 0  # Reset counter
        while self.count + self.batch_size < self.n_dataset:
            # Randomly choose a sampling strategy
            chance = np.random.rand()
            if chance < self.probs[0]:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            elif chance < self.probs[0] + self.probs[1]:
                classes = self.pick_nearby(self.labels_set, self.n_classes, self.class_dists)
            else:
                classes = self.pick_probs(self.labels_set, self.n_classes, self.class_probs)

            # Collect indices for the selected classes
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield torch.tensor(list(map(int, indices)))  # Yield batch indices
            self.count += self.batch_size

    def __len__(self):
        """
        Returns the number of batches.
        """
        return self.n_dataset // self.batch_size


class BalancedAudioToSpeechLabelDataLayer(DataLayerNM):
    """
    Data Layer for speech classification tasks, loading audio data and corresponding labels.
    Parses JSON manifests describing audio file paths, durations, and labels.
    """

    @property
    def output_ports(self):
        """
        Defines the outputs of the Data Layer.
        """
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),  # Batched audio signals
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),  # Lengths of audio signals
            'label': NeuralType(tuple('B'), LabelsType()),  # Corresponding labels
            'label_length': NeuralType(tuple('B'), LengthsType()),  # Lengths of labels
        }

    def __init__(self, *, manifest_filepath: str, labels: List[str], batch_size: int, sample_rate: int = 16000,
                 int_values: bool = False, num_workers: int = 0, shuffle: bool = True, min_duration: Optional[float] = 0.1,
                 max_duration: Optional[float] = None, trim_silence: bool = False, drop_last: bool = False,
                 load_audio: bool = True, augmentor: Optional[Union[AudioAugmentor, Dict[str, Dict[str, Any]]]] = None,
                 num_classes: int = 35, class_dists=None, class_probs=None, probs_num=0):
        """
        Initializes the Data Layer.

        Args:
            manifest_filepath (str): Path to the JSON manifest file.
            labels (list): List of possible class labels.
            batch_size (int): Batch size.
            sample_rate (int): Target sample rate for audio data.
            int_values (bool): If True, interpret audio as int data.
            num_workers (int): Number of worker threads for data loading.
            shuffle (bool): If True, shuffle the data.
            min_duration (float): Minimum duration of audio files to load.
            max_duration (float): Maximum duration of audio files to load.
            trim_silence (bool): If True, trim silence from audio signals.
            drop_last (bool): If True, drop the last incomplete batch.
            load_audio (bool): If True, load the audio data.
            augmentor (AudioAugmentor or dict): Augmentor for data augmentation.
            num_classes (int): Number of classes to sample in each batch.
            class_dists (dict): Class distances for "nearby" sampling.
            class_probs (dict): Probability weights for class sampling.
            probs_num (int): Index to select a probability distribution.
        """
        super(BalancedAudioToSpeechLabelDataLayer, self).__init__()

        # Initialize member variables
        self._manifest_filepath = manifest_filepath
        self._labels = labels
        self._sample_rate = sample_rate

        # Process augmentations if provided
        if augmentor is not None:
            augmentor = self._process_augmentations(augmentor)

        # Initialize waveform featurizer
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)

        # Dataset parameters
        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'trim': trim_silence,
            'load_audio': load_audio,
        }
        # Initialize the dataset
        self._dataset = AudioLabelDataset(**dataset_params)

        # Extract labels for the dataset
        labels = []
        for sample in self._dataset.collection:
            labels.append(self._dataset.label2id[sample.label])

        # Initialize the data loader with the BalancedBatchSampler
        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_sampler=BalancedBatchSampler(labels, n_classes=num_classes,
                                               n_samples=batch_size // num_classes,
                                               class_dists=class_dists, class_probs=class_probs, probs_num=probs_num),
            collate_fn=partial(seq_collate_fn, token_pad_value=0),  # Collate function for batching
            num_workers=num_workers,
        )

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self._dataset)

    def _process_augmentations(self, augmentor) -> AudioAugmentor:
        """
        Processes augmentation configurations and initializes an AudioAugmentor.
        Args:
            augmentor (dict): Dictionary of augmentations and their parameters.
        Returns:
            AudioAugmentor: Initialized augmentor with specified perturbations.
        """
        augmentations = []
        for augment_name, augment_kwargs in augmentor.items():
            prob = augment_kwargs.get('prob', None)

            if prob is None:
                logging.error(
                    f'Augmentation "{augment_name}" will not be applied as '
                    f'keyword argument "prob" was not defined for this augmentation.'
                )
            else:
                _ = augment_kwargs.pop('prob')
                try:
                    augmentation = perturbation_types[augment_name](**augment_kwargs)
                    augmentations.append([prob, augmentation])
                except KeyError:
                    logging.error(f"Invalid perturbation name. Allowed values : {perturbation_types.keys()}")

        return AudioAugmentor(perturbations=augmentations)

    @property
    def dataset(self):
        """
        Placeholder for dataset property (not implemented).
        """
        return None

    @property
    def data_iterator(self):
        """
        Returns the data loader.
        """
        return self._dataloader
