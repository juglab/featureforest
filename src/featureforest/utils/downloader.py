from pathlib import Path
import pooch


MODELS_CACHE_DIR = Path.home().joinpath(".featureforest", "models")


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
    try:
        downloaded_file = pooch.retrieve(
            url=model_url,
            fname=model_name,
            path=cache_dir,
            known_hash=None,
            processor=pooch.Unzip() if is_archived else None
        )
        # for zip files, get the file ending with "pt" or "pth" as model weights file.
        if is_archived:
            pytorch_files = [
                f for f in downloaded_file
                if Path(f).suffix in ["pt", "pth"]
            ]
            downloaded_file = pytorch_files[0] if len(pytorch_files) > 0 else None

        return downloaded_file

    except Exception as err:
        print(f"\nError while downloading the model:\n{err}")
        return None
