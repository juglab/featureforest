from pathlib import Path
import pooch


MODELS_CACHE_DIR = Path.home().joinpath(".featureforest").joinpath("models")
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def is_model_exists(model_name: str) -> bool:
    model_file = MODELS_CACHE_DIR.joinpath(model_name)
    return model_file.exists()


def download_model(
    model_url: str, model_name: str,
    cache_dir: Path = MODELS_CACHE_DIR, is_archived: bool = False
) -> str:
    """Download a model weights from a given url.

    Args:
        model_url (str): the model weights' url.
        model_name (str): model's name that will be saved in cache.
        cache_dir (Path, optional): download directory. Defaults to CACHE_DIR.
        is_archived (bool, optional):   set to True to unzip the downloaded file.
                                        Defaults to False.

    Returns:
        str: full path of the downloaded file.
    """
    if is_model_exists(model_name):
        return str(MODELS_CACHE_DIR.joinpath(model_name).absolute())

    try:
        downloaded_file = pooch.retrieve(
            url=model_url,
            fname=model_name,
            path=cache_dir,
            known_hash=None,
            processor=pooch.Unzip() if is_archived else None
        )
        return downloaded_file

    except Exception as err:
        print(f"\nError while downloading the model:\n{err}")
        return None


if __name__ == "__main__":
    model_file = download_model(
        model_url="https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        model_name="mobile_sam.pt"
    )
    print(model_file)
