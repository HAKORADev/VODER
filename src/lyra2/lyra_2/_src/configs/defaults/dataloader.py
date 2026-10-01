from hydra.core.config_store import ConfigStore
from lyra_2._src.datasets.config_dataverse import DATAVERSE_CONFIG
from lyra_2._src.datasets.depth_warp_dataloader import get_gen3c_multiple_video_dataloader


def lyra_register_dataloaders():
    """Register lyra_2 dataloaders."""
    cs = ConfigStore.instance()

    for dataset_name in DATAVERSE_CONFIG:
        dataloader = get_gen3c_multiple_video_dataloader(
            dataset_list=[dataset_name],
            dataset_weight_list=[1],
            num_workers=2,
            prefetch_factor=2,
        )
        cs.store(group="data_train", package="dataloader_train", name=dataset_name, node=dataloader)
        cs.store(group="data_val", package="dataloader_val", name=dataset_name, node=dataloader)
