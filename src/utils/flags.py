# Copyright (c) Facebook, Inc. and its affiliates.
import argparse


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        # TODO: Add Help flag here describing MMF Configuration
        # and point to configuration documentation
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        # This is needed to support torch.distributed.launch
        self.parser.add_argument(
            "--local_rank", type=int, default=None, help="Local rank of the argument"
        )
        self.parser.add_argument(
            "--config", type=str, default=None, help="config yaml"
        )
        self.parser.add_argument(
            "--datasets", type=str, default=None, help="textvqa"
        )
        self.parser.add_argument(
            "--model", type=str, default=None, help="m4c"
        )
        self.parser.add_argument(
            "--run_type", type=str, default=None, help="train/val"
        )
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )

flags = Flags()
