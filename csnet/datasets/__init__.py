from copy import deepcopy
import csnet.datasets.NUDTMIRSDTDataset as NUDTMIRSDTDataset
from csnet.datasets.MISTDataset import MISTDataset
from csnet.datasets.SatVideoIRSTDDataset import SatVideoIRSTDDataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from csnet.datasets.SyntheticDataset import SyntheticDataset


def unpack_batch(data):
    """Return the five fields shared by standard and visualization datasets."""
    frames, mask, height, width, name, *_ = data
    return frames, mask, height, width, name


def build_dataset(dataset_cfg, mode='train'):
    dataset_cfg = deepcopy(dataset_cfg)
    dataset_name = dataset_cfg.pop('name')
    if dataset_name == 'MIST':
        train_dataset = MISTDataset(**dataset_cfg, split='train')
        val_dataset = MISTDataset(**dataset_cfg, split='val_all')
        val_hard_dataset = MISTDataset(**dataset_cfg, split='val_hard')
    elif dataset_name == 'SatVideoIRSTD':
        train_dataset = SatVideoIRSTDDataset(**dataset_cfg, split='train')
        val_dataset = SatVideoIRSTDDataset(**dataset_cfg, split='val_all')
        val_hard_dataset = SatVideoIRSTDDataset(**dataset_cfg, split='val_hard')
    elif dataset_name == 'NUDT-MIRSDT':
        train_dataset = NUDTMIRSDTDataset.TrainDataset(dataset_cfg)
        val_dataset = NUDTMIRSDTDataset.ValDataset(dataset_cfg, split='all')
        val_hard_dataset = NUDTMIRSDTDataset.ValDataset(dataset_cfg, split='lSCR')
    # elif dataset_name == 'Synthetic':
    #     val_dataset = SyntheticDataset(**dataset_cfg, split='val_all')
    else:
        raise NotImplementedError(
            f"Invalid dataset name '{dataset_name}'. Only MIST and NUDT-MIRSDT are supported.")

    if mode == 'train':
        return train_dataset, val_dataset, val_hard_dataset
    elif mode == 'val':
        return val_dataset, val_hard_dataset
    else:
        raise ValueError(f"Invalid mode '{mode}'. It must be 'train' or 'val'.")


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
