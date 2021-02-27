# Copyright (c) Facebook, Inc. and its affiliates.
imdb_version = 1
FASTTEXT_WIKI_URL = (
    "https://dl.fbaipublicfiles.com/pythia/pretrained_models/fasttext/wiki.en.bin"
)

VISUAL_GENOME_CONSTS = {
    "imdb_url": "https://dl.fbaipublicfiles.com/pythia/data/imdb/visual_genome.tar.gz",
    "features_url": "https://dl.fbaipublicfiles.com/pythia/features/visual_genome.tar.gz",  # noqa
    "synset_file": "vg_synsets.txt",
    "vocabs": "https://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz",
}


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
