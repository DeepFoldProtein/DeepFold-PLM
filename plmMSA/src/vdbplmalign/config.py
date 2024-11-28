import os
from threading import Lock

from dotenv import load_dotenv


class SingletonMeta(type):
    """
    Singleton Metaclass. All classes using this metaclass will be singleton classes.
    """

    _instances = {}
    _lock = Lock()  # Lock object to synchronize threads during the first access

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class GlobalConfig(metaclass=SingletonMeta):
    def __init__(
        self,
        ankh_device=None,
        esm1b_device=None,
        prottrans_device=None,
        vdbplmalign_api_port=None,
        faiss_api_url=None,
        destroy_models_after_search=None,
    ):
        self.load_settings(
            ankh_device,
            esm1b_device,
            prottrans_device,
            vdbplmalign_api_port,
            faiss_api_url,
            destroy_models_after_search,
        )

    def load_settings(
        self,
        ankh_device=None,
        esm1b_device=None,
        prottrans_device=None,
        vdbplmalign_api_port=None,
        faiss_api_url=None,
        destroy_models_after_search=None,
    ):

        # Device Map
        self.ANKH_DEVICE = ankh_device
        self.ESM1B_DEVICE = esm1b_device
        self.PROTTRANS_DEVICE = prottrans_device

        # API settings
        self.vdbplmalign_API_PORT = vdbplmalign_api_port
        self.FAISS_API_URL = faiss_api_url
        self.DESTROY_MODELS_AFTER_SEARCH = destroy_models_after_search

    def reinit(
        self,
        ankh_device=None,
        esm1b_device=None,
        prottrans_device=None,
        vdbplmalign_api_port=None,
        faiss_api_url=None,
        destroy_models_after_search=None,
    ):
        """
        Reinitialize the configuration with new values or reload from environment variables.
        """
        self.load_settings(
            ankh_device,
            esm1b_device,
            prottrans_device,
            vdbplmalign_api_port,
            faiss_api_url,
            destroy_models_after_search,
        )
