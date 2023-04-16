import omegaconf


def get_cfg(path) -> omegaconf:
    return omegaconf.OmegaConf.load(path)