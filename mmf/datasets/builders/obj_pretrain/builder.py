# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.builders.obj_pretrain.dataset import ObjPretrainDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("obj_pretrain")
class ObjPretrainBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="obj_pretrain", dataset_class=ObjPretrainDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/obj_pretrain/defaults.yaml"

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
            registry.register(
                f"{self.dataset_name}_text_processor", self.dataset.text_processor
            )
        if hasattr(self.dataset, "answer_processor"):
            # registry.register(
            #     self.dataset_name + "_num_final_outputs",
            #     self.dataset.answer_processor.get_vocab_size(),
            # )
            registry.register(
                f"{self.dataset_name}_answer_processor", self.dataset.answer_processor
            )
